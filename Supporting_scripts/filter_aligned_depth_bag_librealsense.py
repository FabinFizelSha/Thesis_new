#!/usr/bin/env python3
"""Rewrite a ROS 2 bag's aligned depth stream using Intel's real librealsense
post-processing filters, instead of the hand-rolled reimplementation in
filter_aligned_depth_bag_realsense_v2.py.

Numpy depth frames decoded from the bag are pushed through a
rs.software_device (a synthetic depth sensor), which hands back real
rs.depth_frame objects that Intel's own compiled spatial/temporal/
hole-filling filters can process. This runs the same code path used by an
actual D400-series camera, rather than a Python re-derivation of the
Grunnet-Jepsen/Tong paper.

Processing order per frame (Intel's documented post-processing pipeline):
  1. Invalidate pixels outside [min_depth_m, max_depth_m].
  2. depth -> disparity
  3. spatial_filter   (edge-preserving smoothing; its own light hole fill)
  4. temporal_filter  (cross-frame persistence; needs frames in order,
                       one persistent filter instance for the whole bag)
  5. disparity -> depth
  6. hole_filling_filter (fills any depth still missing from neighbors)

Unlike filter_aligned_depth_bag_realsense_v2.py and the online RSG
depth-refinement design, this script's hole_filling_filter step
DELIBERATELY invents values for pixels that were never measured, by
extrapolating from valid spatial/temporal neighbors. That is its documented
job and is exactly what "aggressive" offline filtering was asked for here;
it would not be appropriate for a live per-frame runtime filter, but is a
reasonable one-time bake for a bag everyone downstream will replay.

RGB is not involved: librealsense's own filters operate purely on the
depth channel, matching how the real sensor's post-processing works.

Supported depth encoding: 16UC1 (D455 Z16), matching this project's bags.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyrealsense2 as rs
import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CameraInfo, Image

IMAGE_TYPE = "sensor_msgs/msg/Image"
CAMERA_INFO_TYPE = "sensor_msgs/msg/CameraInfo"


@dataclass
class FilterStats:
    input_bag: str
    output_bag: str
    storage_id: str
    depth_topic: str
    min_depth_m: float
    max_depth_m: float
    depth_scale: float
    spatial_magnitude: float
    spatial_smooth_alpha: float
    spatial_smooth_delta: float
    spatial_holes_fill: float
    temporal_smooth_alpha: float
    temporal_smooth_delta: float
    temporal_holes_fill: float
    hole_fill_mode: float
    total_messages: int = 0
    copied_messages: int = 0
    depth_frames: int = 0
    depth_pixels: int = 0
    valid_pixels_before: int = 0
    valid_pixels_after_range_gate: int = 0
    valid_pixels_after_filtering: int = 0
    already_invalid_pixels: int = 0
    below_min_pixels: int = 0
    above_max_pixels: int = 0
    pixels_filled_by_hole_filling: int = 0


class DepthFilterError(RuntimeError):
    """Raised for an invalid bag, topic, or depth image."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a ROS 2 bag and replace its aligned-depth stream with the "
            "output of Intel's real librealsense spatial/temporal/"
            "hole-filling post-processing filters."
        )
    )
    parser.add_argument("input_bag", type=Path, help="Input ROS 2 bag directory (rectified+aligned)")
    parser.add_argument("output_bag", type=Path, help="New output bag directory")
    parser.add_argument(
        "--depth-topic",
        default=None,
        help=(
            "Aligned depth sensor_msgs/msg/Image topic. If omitted, the script "
            "auto-selects the only Image topic containing both 'aligned' and 'depth'."
        ),
    )
    parser.add_argument(
        "--depth-info-topic",
        default="/camera/aligned_depth_to_color/camera_info",
        help="CameraInfo topic matching --depth-topic, used only for width/height",
    )
    parser.add_argument("--min-depth-m", type=float, default=0.30)
    parser.add_argument("--max-depth-m", type=float, default=6.00)
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=0.001,
        help="Metres per raw uint16 unit (D455 Z16: 0.001)",
    )
    parser.add_argument(
        "--spatial-magnitude", type=float, default=5.0,
        help="Spatial filter iteration count, range [1,5]. Aggressive default: 5 (max).",
    )
    parser.add_argument(
        "--spatial-smooth-alpha", type=float, default=0.25,
        help="Spatial filter current-pixel weight, range [0.25,1.0]. Lower = more "
             "smoothing. Aggressive default: 0.25 (min/max-smoothing).",
    )
    parser.add_argument(
        "--spatial-smooth-delta", type=float, default=50.0,
        help="Spatial filter edge threshold in disparity levels, range [1,50]. "
             "Higher = more willing to blend across depth jumps. Aggressive default: 50 (max).",
    )
    parser.add_argument(
        "--spatial-holes-fill", type=float, default=3.0,
        help="Spatial filter's own light hole-fill extrapolation degree, range [0,5].",
    )
    parser.add_argument(
        "--temporal-smooth-alpha", type=float, default=0.2,
        help="Temporal filter current-frame weight, range [0,1]. Lower = more "
             "historical smoothing. Kept off the true minimum (0) because the Go1 "
             "is a moving platform; too much temporal weight risks smearing "
             "geometry across frames where the scene genuinely changed.",
    )
    parser.add_argument(
        "--temporal-smooth-delta", type=float, default=40.0,
        help="Temporal filter edge threshold, range [1,100]. Moderated for the same "
             "moving-platform reason as --temporal-smooth-alpha.",
    )
    parser.add_argument(
        "--temporal-holes-fill", type=float, default=5.0,
        help="Temporal filter persistency mode, range [0,8]. Higher = holds onto "
             "old valid values longer to fill new holes.",
    )
    parser.add_argument(
        "--hole-fill-mode", type=float, default=2.0,
        help="Standalone hole_filling_filter mode: 0=fill_from_left, "
             "1=farest_from_around (library default), 2=nearest_from_around "
             "(default here: fills occlusion holes with foreground depth, which "
             "is the safer bias for obstacle/mesh geometry).",
    )
    parser.add_argument(
        "--storage-id",
        default=None,
        help="Override input/output storage plugin, e.g. sqlite3 or mcap",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output_bag first if it already exists",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N filtered depth frames; 0 disables progress",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_bag.exists():
        raise DepthFilterError(f"Input bag does not exist: {args.input_bag}")
    if args.input_bag.resolve() == args.output_bag.resolve():
        raise DepthFilterError("Input and output bag paths must be different")
    if not math.isfinite(args.min_depth_m) or args.min_depth_m < 0.0:
        raise DepthFilterError("--min-depth-m must be finite and >= 0")
    if not math.isfinite(args.max_depth_m) or args.max_depth_m <= args.min_depth_m:
        raise DepthFilterError("--max-depth-m must be finite and > --min-depth-m")
    if not math.isfinite(args.depth_scale) or args.depth_scale <= 0.0:
        raise DepthFilterError("--depth-scale must be finite and > 0")
    if not 1.0 <= args.spatial_magnitude <= 5.0:
        raise DepthFilterError("--spatial-magnitude must be in [1,5]")
    if not 0.25 <= args.spatial_smooth_alpha <= 1.0:
        raise DepthFilterError("--spatial-smooth-alpha must be in [0.25,1.0]")
    if not 1.0 <= args.spatial_smooth_delta <= 50.0:
        raise DepthFilterError("--spatial-smooth-delta must be in [1,50]")
    if not 0.0 <= args.spatial_holes_fill <= 5.0:
        raise DepthFilterError("--spatial-holes-fill must be in [0,5]")
    if not 0.0 <= args.temporal_smooth_alpha <= 1.0:
        raise DepthFilterError("--temporal-smooth-alpha must be in [0,1]")
    if not 1.0 <= args.temporal_smooth_delta <= 100.0:
        raise DepthFilterError("--temporal-smooth-delta must be in [1,100]")
    if not 0.0 <= args.temporal_holes_fill <= 8.0:
        raise DepthFilterError("--temporal-holes-fill must be in [0,8]")
    if args.hole_fill_mode not in (0.0, 1.0, 2.0):
        raise DepthFilterError("--hole-fill-mode must be 0, 1, or 2")
    if args.progress_every < 0:
        raise DepthFilterError("--progress-every must be >= 0")

    if args.output_bag.exists():
        if not args.overwrite:
            raise DepthFilterError(
                f"Output already exists: {args.output_bag}. Use --overwrite to replace it."
            )
        if args.output_bag.is_dir():
            shutil.rmtree(args.output_bag)
        else:
            args.output_bag.unlink()


def detect_storage_id(input_bag: Path) -> str:
    metadata_path = input_bag / "metadata.yaml"
    if not metadata_path.exists():
        return "sqlite3"
    for line in metadata_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("storage_identifier:"):
            value = stripped.split(":", 1)[1].strip().strip("'\"")
            if value:
                return value
    return "sqlite3"


def topic_map(topics: Iterable[object]) -> dict[str, object]:
    return {topic.name: topic for topic in topics}


def choose_depth_topic(topics: Iterable[object], requested: str | None) -> str:
    metadata = topic_map(topics)
    if requested:
        if requested not in metadata:
            available = "\n  ".join(sorted(metadata))
            raise DepthFilterError(
                f"Depth topic not found: {requested}\nAvailable topics:\n  {available}"
            )
        if metadata[requested].type != IMAGE_TYPE:
            raise DepthFilterError(
                f"{requested} has type {metadata[requested].type}, expected {IMAGE_TYPE}"
            )
        return requested

    candidates = [
        topic.name for topic in metadata.values()
        if topic.type == IMAGE_TYPE and "depth" in topic.name.lower() and "aligned" in topic.name.lower()
    ]
    if len(candidates) == 1:
        return candidates[0]
    all_depth_images = [
        topic.name for topic in metadata.values()
        if topic.type == IMAGE_TYPE and "depth" in topic.name.lower()
    ]
    if len(all_depth_images) == 1:
        return all_depth_images[0]
    shown = candidates if candidates else all_depth_images
    formatted = "\n  ".join(sorted(shown)) if shown else "(none found)"
    raise DepthFilterError(
        "Could not uniquely auto-select the aligned depth topic. "
        "Pass --depth-topic explicitly. Candidate depth Image topics:\n  " + formatted
    )


def find_first_camera_info(bag_dir: Path, storage_id: str, topic: str, topic_types: dict[str, str]) -> CameraInfo:
    if topic not in topic_types:
        raise DepthFilterError(f"CameraInfo topic not found in bag: {topic}")
    msg_type = get_message(topic_types[topic])
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage_id),
        rosbag2_py.ConverterOptions("", ""),
    )
    while reader.has_next():
        current_topic, data, _timestamp = reader.read_next()
        if current_topic == topic:
            return deserialize_message(data, msg_type)
    raise DepthFilterError(f"No CameraInfo messages found on topic: {topic}")


def _depth_view(msg: Image) -> tuple[bytearray, np.ndarray]:
    dtype = np.dtype(">u2" if bool(msg.is_bigendian) else "<u2")
    itemsize = dtype.itemsize
    expected_row_bytes = int(msg.width) * itemsize
    if msg.height <= 0 or msg.width <= 0:
        raise DepthFilterError(f"Invalid depth dimensions: width={msg.width}, height={msg.height}")
    if msg.step < expected_row_bytes:
        raise DepthFilterError(f"Image step {msg.step} is smaller than row payload {expected_row_bytes}")
    expected_bytes = int(msg.height) * int(msg.step)
    raw = bytearray(msg.data)
    if len(raw) < expected_bytes:
        raise DepthFilterError(f"Depth data has {len(raw)} bytes but dimensions require {expected_bytes}")
    depth = np.ndarray(
        shape=(int(msg.height), int(msg.width)),
        dtype=dtype,
        buffer=raw,
        strides=(int(msg.step), itemsize),
    )
    return raw, depth


class LibrealsenseDepthPipeline:
    """Wraps a software_device + the real spatial/temporal/hole-filling chain.

    One instance must be reused across every frame in a bag: the temporal
    filter keeps state between .process() calls, and the software_device's
    callback plumbing is set up once and fed frames one at a time.
    """

    def __init__(self, width: int, height: int, depth_scale: float, args: argparse.Namespace) -> None:
        self.width = width
        self.height = height
        self._device = rs.software_device()
        sensor = self._device.add_sensor("Depth")

        video_stream = rs.video_stream()
        video_stream.type = rs.stream.depth
        video_stream.fmt = rs.format.z16
        video_stream.width = width
        video_stream.height = height
        video_stream.fps = 30
        video_stream.bpp = 2
        video_stream.index = 0
        video_stream.uid = 0
        intrinsics = rs.intrinsics()
        intrinsics.width = width
        intrinsics.height = height
        intrinsics.fx = float(width)
        intrinsics.fy = float(width)
        intrinsics.ppx = width / 2.0
        intrinsics.ppy = height / 2.0
        intrinsics.model = rs.distortion.none
        intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
        video_stream.intrinsics = intrinsics

        self._profile = sensor.add_video_stream(video_stream)
        sensor.add_read_only_option(rs.option.depth_units, depth_scale)
        self._out_queue: "queue.Queue" = queue.Queue()
        sensor.open(self._profile)
        sensor.start(self._out_queue.put)
        self._sensor = sensor

        self._depth_to_disparity = rs.disparity_transform(True)
        self._disparity_to_depth = rs.disparity_transform(False)
        self._spatial = rs.spatial_filter()
        self._spatial.set_option(rs.option.filter_magnitude, args.spatial_magnitude)
        self._spatial.set_option(rs.option.filter_smooth_alpha, args.spatial_smooth_alpha)
        self._spatial.set_option(rs.option.filter_smooth_delta, args.spatial_smooth_delta)
        self._spatial.set_option(rs.option.holes_fill, args.spatial_holes_fill)
        self._temporal = rs.temporal_filter()
        self._temporal.set_option(rs.option.filter_smooth_alpha, args.temporal_smooth_alpha)
        self._temporal.set_option(rs.option.filter_smooth_delta, args.temporal_smooth_delta)
        self._temporal.set_option(rs.option.holes_fill, args.temporal_holes_fill)
        self._hole_filling = rs.hole_filling_filter()
        self._hole_filling.set_option(rs.option.holes_fill, args.hole_fill_mode)

        self._frame_number = 0

    def process(self, depth_uint16: np.ndarray, timestamp_ms: float) -> np.ndarray:
        sw_frame = rs.software_video_frame()
        sw_frame.pixels = np.ascontiguousarray(depth_uint16).tobytes()
        sw_frame.stride = self.width * 2
        sw_frame.bpp = 2
        sw_frame.timestamp = timestamp_ms
        sw_frame.domain = rs.timestamp_domain.hardware_clock
        sw_frame.frame_number = self._frame_number
        sw_frame.profile = self._profile.as_video_stream_profile()
        self._frame_number += 1

        self._sensor.on_video_frame(sw_frame)
        raw_frame = self._out_queue.get(timeout=5.0)
        depth_frame = raw_frame.as_depth_frame()

        f = self._depth_to_disparity.process(depth_frame)
        f = self._spatial.process(f)
        f = self._temporal.process(f)
        f = self._disparity_to_depth.process(f)
        f = self._hole_filling.process(f)
        return np.asanyarray(f.get_data())

    def close(self) -> None:
        self._sensor.stop()
        self._sensor.close()


def copy_and_filter(args: argparse.Namespace) -> FilterStats:
    storage_id = args.storage_id or detect_storage_id(args.input_bag)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.input_bag), storage_id=storage_id),
        rosbag2_py.ConverterOptions("", ""),
    )
    topics = reader.get_all_topics_and_types()
    topic_types = {t.name: t.type for t in topics}
    depth_topic = choose_depth_topic(topics, args.depth_topic)
    depth_info = find_first_camera_info(args.input_bag, storage_id, args.depth_info_topic, topic_types)
    width, height = int(depth_info.width), int(depth_info.height)
    if width <= 0 or height <= 0:
        raise DepthFilterError(f"Invalid depth dimensions from {args.depth_info_topic}: {width}x{height}")

    pipeline = LibrealsenseDepthPipeline(width, height, args.depth_scale, args)

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(args.output_bag), storage_id=storage_id),
        rosbag2_py.ConverterOptions("", ""),
    )
    for metadata in topics:
        writer.create_topic(metadata)

    stats = FilterStats(
        input_bag=str(args.input_bag),
        output_bag=str(args.output_bag),
        storage_id=storage_id,
        depth_topic=depth_topic,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        depth_scale=args.depth_scale,
        spatial_magnitude=args.spatial_magnitude,
        spatial_smooth_alpha=args.spatial_smooth_alpha,
        spatial_smooth_delta=args.spatial_smooth_delta,
        spatial_holes_fill=args.spatial_holes_fill,
        temporal_smooth_alpha=args.temporal_smooth_alpha,
        temporal_smooth_delta=args.temporal_smooth_delta,
        temporal_holes_fill=args.temporal_holes_fill,
        hole_fill_mode=args.hole_fill_mode,
    )

    try:
        while reader.has_next():
            topic_name, serialized_data, timestamp_ns = reader.read_next()
            stats.total_messages += 1

            if topic_name != depth_topic:
                writer.write(topic_name, serialized_data, timestamp_ns)
                stats.copied_messages += 1
                continue

            msg = deserialize_message(serialized_data, Image)
            raw, depth = _depth_view(msg)

            already_invalid = depth == 0
            depth_m = depth.astype(np.float32) * np.float32(args.depth_scale)
            below = (~already_invalid) & (depth_m < args.min_depth_m)
            above = (~already_invalid) & (depth_m > args.max_depth_m)
            range_rejected = already_invalid | below | above
            depth[range_rejected] = 0

            filtered = pipeline.process(depth, timestamp_ms=timestamp_ns / 1.0e6)
            depth[:, :] = filtered

            msg.data = bytes(raw)
            writer.write(topic_name, serialize_message(msg), timestamp_ns)

            stats.depth_frames += 1
            stats.depth_pixels += width * height
            stats.valid_pixels_before += int(np.count_nonzero(~already_invalid))
            stats.valid_pixels_after_range_gate += int(np.count_nonzero(~range_rejected))
            stats.valid_pixels_after_filtering += int(np.count_nonzero(filtered))
            stats.already_invalid_pixels += int(np.count_nonzero(already_invalid))
            stats.below_min_pixels += int(np.count_nonzero(below))
            stats.above_max_pixels += int(np.count_nonzero(above))
            stats.pixels_filled_by_hole_filling += int(
                np.count_nonzero((filtered != 0) & range_rejected)
            )

            if args.progress_every and stats.depth_frames % args.progress_every == 0:
                print(
                    f"Processed {stats.depth_frames} depth frames; "
                    f"valid before={stats.valid_pixels_before:,} "
                    f"after range-gate={stats.valid_pixels_after_range_gate:,} "
                    f"after filtering={stats.valid_pixels_after_filtering:,}",
                    flush=True,
                )
    finally:
        pipeline.close()
        del writer
        del reader

    return stats


def write_report(stats: FilterStats, output_bag: Path) -> Path:
    report_path = output_bag.parent / f"{output_bag.name}_librealsense_filter_report.json"
    data = asdict(stats)
    if stats.depth_pixels:
        data["valid_fraction_before"] = stats.valid_pixels_before / stats.depth_pixels
        data["valid_fraction_after_range_gate"] = stats.valid_pixels_after_range_gate / stats.depth_pixels
        data["valid_fraction_after_filtering"] = stats.valid_pixels_after_filtering / stats.depth_pixels
    report_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        stats = copy_and_filter(args)
        report_path = write_report(stats, args.output_bag)
    except (DepthFilterError, RuntimeError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if stats.depth_frames == 0:
        print("ERROR: No depth frames were found; output bag was created without filtering.")
        return 3

    print("\nlibrealsense depth filtering complete")
    print(f"  Input bag:                    {stats.input_bag}")
    print(f"  Output bag:                   {stats.output_bag}")
    print(f"  Depth topic:                  {stats.depth_topic}")
    print(f"  Range retained:               {stats.min_depth_m:.3f} to {stats.max_depth_m:.3f} m")
    print(f"  Spatial:  magnitude={stats.spatial_magnitude:.0f} alpha={stats.spatial_smooth_alpha:.3f} "
          f"delta={stats.spatial_smooth_delta:.0f} holes_fill={stats.spatial_holes_fill:.0f}")
    print(f"  Temporal: alpha={stats.temporal_smooth_alpha:.3f} delta={stats.temporal_smooth_delta:.0f} "
          f"holes_fill={stats.temporal_holes_fill:.0f}")
    print(f"  Hole-filling mode:            {stats.hole_fill_mode:.0f}")
    print(f"  Depth frames:                 {stats.depth_frames:,}")
    print(f"  Pixels valid before:          {stats.valid_pixels_before:,} "
          f"({stats.valid_pixels_before / stats.depth_pixels:.1%})")
    print(f"  Pixels valid after range gate:{stats.valid_pixels_after_range_gate:,} "
          f"({stats.valid_pixels_after_range_gate / stats.depth_pixels:.1%})")
    print(f"  Pixels valid after filtering: {stats.valid_pixels_after_filtering:,} "
          f"({stats.valid_pixels_after_filtering / stats.depth_pixels:.1%})")
    print(f"  Pixels filled by hole-filling:{stats.pixels_filled_by_hole_filling:,}")
    print(f"  Report:                       {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
