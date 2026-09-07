#!/usr/bin/env python3
"""Quantify what Supporting_scripts/filter_aligned_depth_bag_librealsense.py
changed, by comparing its input (rectified+aligned, pre-filter depth) against
its output (final, post-filter depth) frame-by-frame across the whole
go1_d455_20260802_164826 bag.

Two passes over the same frame stream:
  1. Stats pass: every one of the 2,678 depth frames, cheap per-frame metrics
     only (no RGB decode). Produces results/stats.json.
  2. Example pass: re-reads the same two bags to pull full-resolution
     RGB + depth-before + depth-after PNGs for a stratified sample of frames
     (chosen by pre-filter valid-pixel percentile, so the gallery shows the
     full range of conditions in the bag, not just the most flattering ones).

Both bags carry the depth topic with identical frame count and order (the
filter stage processes messages strictly in place, one in one out; the
odom-timestamp-fix stage after it never touches the depth topic at all), so
frame N in one bag is always frame N in the other -- no timestamp matching
needed, just parallel iteration.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.ndimage import binary_erosion, median_filter

BASE = Path("/home/student/datasets/rebased_bags/go1_d455_20260802_164826")
BEFORE_BAG = BASE / "go1_d455_20260802_164826_ros2_rectified_aligned"
AFTER_BAG = BASE / "go1_d455_20260802_164826_ros2_rectified_aligned_smoothed_odom_time_fixed"
RGB_TOPIC = "/camera/color/image_rect_raw"
DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"
DEPTH_SCALE_M = 0.001  # D455 Z16: metres per raw uint16 unit
COLORMAP_MAX_MM = 5000  # fixed ceiling so all example images share one color scale

RESULTS_DIR = Path(__file__).resolve().parent / "results"
EXAMPLES_DIR = RESULTS_DIR / "examples"
N_EXAMPLES = 10


def make_reader(bag_dir: Path) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    return reader


def iter_topic(bag_dir: Path, topic: str):
    reader = make_reader(bag_dir)
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_type = get_message(topic_types[topic])
    while reader.has_next():
        name, data, _ts = reader.read_next()
        if name == topic:
            yield deserialize_message(data, msg_type)


def decode_depth16(msg) -> np.ndarray:
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    rows = raw[: msg.height * msg.step].reshape(msg.height, msg.step)
    return rows[:, : msg.width * 2].copy().view("<u2").reshape(msg.height, msg.width)


def decode_rgb(msg) -> np.ndarray:
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    rows = raw[: msg.height * msg.step].reshape(msg.height, msg.step)
    return rows[:, : msg.width * 3].reshape(msg.height, msg.width, 3)


def depth_to_colormap(depth_u16: np.ndarray, max_mm: int = COLORMAP_MAX_MM) -> np.ndarray:
    depth = depth_u16.astype(np.float32)
    valid = depth > 0
    norm = np.zeros_like(depth, dtype=np.uint8)
    norm[valid] = np.clip(depth[valid] / max_mm * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colored[~valid] = (30, 30, 30)
    return colored


def local_roughness(depth_m: np.ndarray, score_mask: np.ndarray, med: np.ndarray | None = None) -> float:
    """Mean |value - 3x3 median| over an arbitrary mask of pixel locations.
    Caller picks the mask so the 3x3 window at every scored location is
    known to be fully valid in whichever frame `depth_m` came from."""
    if not np.any(score_mask):
        return float("nan")
    if med is None:
        med = median_filter(depth_m, size=3)
    return float(np.mean(np.abs(depth_m - med)[score_mask]))


def frame_stats(depth_before_u16: np.ndarray, depth_after_u16: np.ndarray) -> dict:
    before_m = depth_before_u16.astype(np.float32) * DEPTH_SCALE_M
    after_m = depth_after_u16.astype(np.float32) * DEPTH_SCALE_M
    valid_before = depth_before_u16 > 0
    valid_after = depth_after_u16 > 0
    both_valid = valid_before & valid_after
    newly_filled = (~valid_before) & valid_after
    still_invalid = (~valid_before) & (~valid_after)
    total = depth_before_u16.size

    if np.any(both_valid):
        abs_change = np.abs(after_m[both_valid] - before_m[both_valid])
        mean_abs_change_m = float(np.mean(abs_change))
        max_abs_change_m = float(np.max(abs_change))
    else:
        mean_abs_change_m = float("nan")
        max_abs_change_m = float("nan")

    # "interior_before": pixels whose full neighborhood was already valid
    # pre-filter -- i.e. genuine sensor measurements, nowhere near a hole.
    # Computed at two window scales: 3x3 (pixel-to-pixel dither) and 9x9
    # (coarser structural waviness/bias), since a filter can plausibly move
    # noise between scales rather than removing it outright.
    result = {
        "valid_before_frac": float(np.count_nonzero(valid_before)) / total,
        "valid_after_frac": float(np.count_nonzero(valid_after)) / total,
        "newly_filled_frac": float(np.count_nonzero(newly_filled)) / total,
        "still_invalid_frac": float(np.count_nonzero(still_invalid)) / total,
        "mean_abs_change_m": mean_abs_change_m,
        "max_abs_change_m": max_abs_change_m,
    }
    for size, tag in ((3, "fine"), (9, "coarse")):
        footprint = np.ones((size, size), dtype=bool)
        interior_before = binary_erosion(valid_before, structure=footprint)
        interior_after = binary_erosion(valid_after, structure=footprint)
        filled_interior = interior_after & (~valid_before)
        after_med = median_filter(after_m, size=size)
        # Fair pre/post comparison: SAME pixel footprint (clean before the
        # filter ran at all), scored on the before values and on the after
        # values respectively. Isolates "did smoothing help or hurt real
        # measurements" from any hole-filling effect entirely.
        result[f"roughness_{tag}_before_m"] = local_roughness(before_m, interior_before)
        result[f"roughness_{tag}_after_same_footprint_m"] = local_roughness(after_m, interior_before, med=after_med)
        # Roughness strictly inside hole-filled (extrapolated, never-measured)
        # territory -- expected to be rougher, since it's invented, not sensed.
        result[f"roughness_{tag}_filled_region_m"] = local_roughness(after_m, filled_interior, med=after_med)
        if tag == "fine":
            result["filled_interior_frac"] = float(np.count_nonzero(filled_interior)) / total
    return result


def summarize(values: list) -> dict:
    arr = np.array([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None, "std": None, "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
        "n": int(arr.size),
    }


def run_stats_pass() -> list[dict]:
    print("Pass 1/2: computing per-frame statistics over the full bag...", flush=True)
    per_frame = []
    for i, (before_msg, after_msg) in enumerate(zip(iter_topic(BEFORE_BAG, DEPTH_TOPIC), iter_topic(AFTER_BAG, DEPTH_TOPIC))):
        before = decode_depth16(before_msg)
        after = decode_depth16(after_msg)
        stats = frame_stats(before, after)
        stats["index"] = i
        per_frame.append(stats)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1} frames analyzed", flush=True)
    print(f"Stats pass complete: {len(per_frame)} frames", flush=True)
    return per_frame


def choose_example_indices(per_frame: list[dict], n: int) -> list[int]:
    """Stratified sample by pre-filter valid-pixel fraction, spanning the
    full range of conditions actually present in the bag (worst coverage to
    best), not just the most dramatic-looking improvements."""
    ordered = sorted(per_frame, key=lambda s: s["valid_before_frac"])
    positions = np.linspace(0, len(ordered) - 1, n)
    seen = set()
    indices = []
    for p in positions:
        candidate = ordered[int(round(p))]
        idx = candidate["index"]
        while idx in seen:
            idx += 1
        seen.add(idx)
        indices.append(idx)
    return sorted(indices)


def run_example_pass(example_indices: list[int]) -> list[dict]:
    print(f"Pass 2/2: exporting {len(example_indices)} example frames...", flush=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    wanted = set(example_indices)
    examples = []

    rgb_iter = iter_topic(BEFORE_BAG, RGB_TOPIC)
    before_iter = iter_topic(BEFORE_BAG, DEPTH_TOPIC)
    after_iter = iter_topic(AFTER_BAG, DEPTH_TOPIC)

    for i, (rgb_msg, before_msg, after_msg) in enumerate(zip(rgb_iter, before_iter, after_iter)):
        if i not in wanted:
            continue
        rgb = decode_rgb(rgb_msg)
        before = decode_depth16(before_msg)
        after = decode_depth16(after_msg)

        tag = f"frame_{i:04d}"
        rgb_path = EXAMPLES_DIR / f"{tag}_rgb.jpg"
        before_path = EXAMPLES_DIR / f"{tag}_depth_before.png"
        after_path = EXAMPLES_DIR / f"{tag}_depth_after.png"

        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(before_path), depth_to_colormap(before))
        cv2.imwrite(str(after_path), depth_to_colormap(after))

        stats = frame_stats(before, after)
        stats["index"] = i
        stats["files"] = {
            "rgb": rgb_path.name,
            "depth_before": before_path.name,
            "depth_after": after_path.name,
        }
        examples.append(stats)
        print(f"  saved {tag}  valid_before={stats['valid_before_frac']:.1%}  valid_after={stats['valid_after_frac']:.1%}", flush=True)

        if len(examples) == len(wanted):
            break

    return sorted(examples, key=lambda e: e["index"])


def main() -> None:
    per_frame = run_stats_pass()

    aggregate = {
        "n_frames": len(per_frame),
        "valid_before_frac": summarize([f["valid_before_frac"] for f in per_frame]),
        "valid_after_frac": summarize([f["valid_after_frac"] for f in per_frame]),
        "newly_filled_frac": summarize([f["newly_filled_frac"] for f in per_frame]),
        "still_invalid_frac": summarize([f["still_invalid_frac"] for f in per_frame]),
        "mean_abs_change_m": summarize([f["mean_abs_change_m"] for f in per_frame]),
        "max_abs_change_m": summarize([f["max_abs_change_m"] for f in per_frame]),
        "filled_interior_frac": summarize([f["filled_interior_frac"] for f in per_frame]),
    }
    for tag in ("fine", "coarse"):
        aggregate[f"roughness_{tag}_before_m"] = summarize([f[f"roughness_{tag}_before_m"] for f in per_frame])
        aggregate[f"roughness_{tag}_after_same_footprint_m"] = summarize([f[f"roughness_{tag}_after_same_footprint_m"] for f in per_frame])
        aggregate[f"roughness_{tag}_filled_region_m"] = summarize([f[f"roughness_{tag}_filled_region_m"] for f in per_frame])
        rb = aggregate[f"roughness_{tag}_before_m"]["mean"]
        ra = aggregate[f"roughness_{tag}_after_same_footprint_m"]["mean"]
        aggregate[f"roughness_{tag}_reduction_pct"] = (rb - ra) / rb * 100.0 if rb else None

    example_indices = choose_example_indices(per_frame, N_EXAMPLES)
    examples = run_example_pass(example_indices)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "before_bag": str(BEFORE_BAG),
        "after_bag": str(AFTER_BAG),
        "depth_topic": DEPTH_TOPIC,
        "rgb_topic": RGB_TOPIC,
        "aggregate": aggregate,
        "examples": examples,
        "per_frame": per_frame,
    }
    stats_path = RESULTS_DIR / "stats.json"
    stats_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {stats_path}")
    print(f"Wrote {len(examples)} example image sets to {EXAMPLES_DIR}")
    print("\n--- Aggregate summary ---")
    print(f"Valid before:   mean={aggregate['valid_before_frac']['mean']:.1%}")
    print(f"Valid after:    mean={aggregate['valid_after_frac']['mean']:.1%}")
    print(f"Newly filled:   mean={aggregate['newly_filled_frac']['mean']:.1%}")
    print(f"Still invalid:  mean={aggregate['still_invalid_frac']['mean']:.1%}")
    print(f"Mean abs change on already-valid pixels: {aggregate['mean_abs_change_m']['mean']*100:.2f} cm (max seen: {aggregate['max_abs_change_m']['max']*100:.1f} cm)")
    for tag, label in (("fine", "3x3"), ("coarse", "9x9")):
        b = aggregate[f"roughness_{tag}_before_m"]["mean"]
        a = aggregate[f"roughness_{tag}_after_same_footprint_m"]["mean"]
        fr = aggregate[f"roughness_{tag}_filled_region_m"]["mean"]
        print(f"Roughness [{label}] same-footprint: before={b*1000:.2f}mm after={a*1000:.2f}mm change={-aggregate[f'roughness_{tag}_reduction_pct']:.1f}%   filled-region={fr*1000:.2f}mm")
    print(f"Hole-filled interior fraction: {aggregate['filled_interior_frac']['mean']:.1%} of frame")


if __name__ == "__main__":
    main()
