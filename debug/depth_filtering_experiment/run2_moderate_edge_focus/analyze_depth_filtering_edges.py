#!/usr/bin/env python3
"""Run 2 of the depth-filtering experiment: moderate/edge-focused librealsense
parameters (library defaults + hole_fill_mode=1, instead of run 1's maxed-out
"aggressive" settings), on the *original* go1_d455_20180128_170537 recording,
restricted to depth <= 5m (the operationally relevant range for this robot).

Run 1 (../EXPERIMENT_REPORT.md) found that maxing every aggressiveness knob
recovers coverage dramatically but measurably increases local roughness on
already-good pixels -- most likely because hard edge-preserving thresholds,
pushed to their limit, create piecewise-flat "plateaus" instead of smooth
gradients. This run asks the follow-up question that report explicitly
flagged as unresolved: with moderate (library-default) thresholds instead,
are actual depth *edges* (not just flat-region noise) preserved better?

Two things differ from run 1's analysis:
  1. A 5m focus mask, applied everywhere: coverage/change/roughness/edges are
     all computed only where the PRE-filter frame had a valid reading within
     5m -- "what's relevant to us," independent of anything the filter does,
     avoiding circularity.
  2. A genuine edge-sharpness metric: Canny edges detected in RGB (a proxy
     for real object-boundary locations) are compared to the depth gradient
     magnitude (Sobel) at those same locations, before vs after filtering.
     Ratio ~1 = edge preserved; ratio < 1 = edge blurred/softened; ratio > 1
     = transition got steeper. Only evaluated at locations whose full 3x3
     neighborhood is valid in BOTH frames, so a hole boundary never gets
     mistaken for a real depth edge.
Coarse-scale (9x9) roughness from run 1 is intentionally not repeated here
-- it was the most expensive part of that run and the ask this time is
specifically about edges, not re-litigating flat-region noise.
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

BASE = Path("/home/student/datasets/rebased_bags/go1_d455_20180128_170537")
BEFORE_BAG = BASE / "20180128_170537_rebased_ros2_rectified_aligned"
AFTER_BAG = BASE / "20180128_170537_rebased_ros2_rectified_aligned_smoothed_moderate_edges_odom_time_fixed"
RGB_TOPIC = "/camera/color/image_rect_raw"
DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"
DEPTH_SCALE_M = 0.001
FOCUS_MAX_M = 5.0          # "what's relevant to us" -- matches this run's filter range gate
COLORMAP_MAX_MM = 5000     # shared visualization ceiling, matches FOCUS_MAX_M

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
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
    return rows[:, : msg.width * 3].reshape(msg.height, msg.width, 3).copy()


def depth_to_colormap(depth_u16: np.ndarray, max_mm: int = COLORMAP_MAX_MM) -> np.ndarray:
    depth = depth_u16.astype(np.float32)
    valid = depth > 0
    norm = np.zeros_like(depth, dtype=np.uint8)
    norm[valid] = np.clip(depth[valid] / max_mm * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colored[~valid] = (30, 30, 30)
    return colored


def local_roughness(depth_m: np.ndarray, score_mask: np.ndarray, med: np.ndarray | None = None) -> float:
    if not np.any(score_mask):
        return float("nan")
    if med is None:
        med = median_filter(depth_m, size=3)
    return float(np.mean(np.abs(depth_m - med)[score_mask]))


def detect_rgb_edges(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, 50, 150) > 0


def depth_gradient_magnitude(depth_m: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def frame_stats(rgb: np.ndarray, depth_before_u16: np.ndarray, depth_after_u16: np.ndarray) -> dict:
    before_m = depth_before_u16.astype(np.float32) * DEPTH_SCALE_M
    after_m = depth_after_u16.astype(np.float32) * DEPTH_SCALE_M
    total = depth_before_u16.size

    # Coverage: independent per frame, both restricted to the focus range.
    valid_before = (depth_before_u16 > 0) & (before_m <= FOCUS_MAX_M)
    valid_after = (depth_after_u16 > 0) & (after_m <= FOCUS_MAX_M)
    newly_filled = (~valid_before) & valid_after
    still_invalid = (~valid_before) & (~valid_after)

    # Change/roughness/edge comparisons: fixed footprint = valid, in-focus,
    # BEFORE pixels only. Avoids circularity (never lets the filter's own
    # output decide which pixels "count").
    roi = valid_before
    both_valid_in_roi = roi & valid_after

    if np.any(both_valid_in_roi):
        abs_change = np.abs(after_m[both_valid_in_roi] - before_m[both_valid_in_roi])
        mean_abs_change_m = float(np.mean(abs_change))
        max_abs_change_m = float(np.max(abs_change))
    else:
        mean_abs_change_m = float("nan")
        max_abs_change_m = float("nan")

    interior_roi = binary_erosion(roi, structure=np.ones((3, 3), dtype=bool))
    after_med = median_filter(after_m, size=3)
    roughness_before = local_roughness(before_m, interior_roi)
    roughness_after_same_footprint = local_roughness(after_m, interior_roi, med=after_med)

    # Edge sharpness: RGB-detected edges, evaluated only where the full 3x3
    # neighborhood is valid+in-focus in BOTH frames (never a hole boundary).
    interior_after = binary_erosion(valid_after, structure=np.ones((3, 3), dtype=bool))
    edge_eval_mask = detect_rgb_edges(rgb) & interior_roi & interior_after
    edge_pixel_count = int(np.count_nonzero(edge_eval_mask))
    if edge_pixel_count > 0:
        grad_before = depth_gradient_magnitude(before_m)
        grad_after = depth_gradient_magnitude(after_m)
        grad_before_mean = float(np.mean(grad_before[edge_eval_mask]))
        grad_after_mean = float(np.mean(grad_after[edge_eval_mask]))
        sharpness_ratio = (grad_after_mean / grad_before_mean) if grad_before_mean > 0 else float("nan")
    else:
        grad_before_mean = float("nan")
        grad_after_mean = float("nan")
        sharpness_ratio = float("nan")

    return {
        "valid_before_frac": float(np.count_nonzero(valid_before)) / total,
        "valid_after_frac": float(np.count_nonzero(valid_after)) / total,
        "newly_filled_frac": float(np.count_nonzero(newly_filled)) / total,
        "still_invalid_frac": float(np.count_nonzero(still_invalid)) / total,
        "mean_abs_change_m": mean_abs_change_m,
        "max_abs_change_m": max_abs_change_m,
        "roughness_before_m": roughness_before,
        "roughness_after_same_footprint_m": roughness_after_same_footprint,
        "edge_pixel_count": edge_pixel_count,
        "edge_grad_before_m_per_px": grad_before_mean,
        "edge_grad_after_m_per_px": grad_after_mean,
        "edge_sharpness_ratio": sharpness_ratio,
    }


def summarize(values: list) -> dict:
    arr = np.array([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None, "std": None, "n": 0}
    return {
        "mean": float(np.mean(arr)), "median": float(np.median(arr)),
        "min": float(np.min(arr)), "max": float(np.max(arr)),
        "std": float(np.std(arr)), "n": int(arr.size),
    }


def run_stats_pass() -> list[dict]:
    print("Pass 1/2: computing per-frame statistics over the full bag...", flush=True)
    per_frame = []
    rgb_iter = iter_topic(BEFORE_BAG, RGB_TOPIC)
    before_iter = iter_topic(BEFORE_BAG, DEPTH_TOPIC)
    after_iter = iter_topic(AFTER_BAG, DEPTH_TOPIC)
    for i, (rgb_msg, before_msg, after_msg) in enumerate(zip(rgb_iter, before_iter, after_iter)):
        rgb = decode_rgb(rgb_msg)
        before = decode_depth16(before_msg)
        after = decode_depth16(after_msg)
        stats = frame_stats(rgb, before, after)
        stats["index"] = i
        per_frame.append(stats)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1} frames analyzed", flush=True)
    print(f"Stats pass complete: {len(per_frame)} frames", flush=True)
    return per_frame


def choose_example_indices(per_frame: list[dict], n: int) -> list[int]:
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
        stats = frame_stats(rgb, before, after)
        stats["index"] = i
        stats["files"] = {"rgb": rgb_path.name, "depth_before": before_path.name, "depth_after": after_path.name}
        examples.append(stats)
        print(f"  saved {tag}  valid_before={stats['valid_before_frac']:.1%}  valid_after={stats['valid_after_frac']:.1%}  edge_ratio={stats['edge_sharpness_ratio']:.2f}", flush=True)
        if len(examples) == len(wanted):
            break
    return sorted(examples, key=lambda e: e["index"])


def main() -> None:
    per_frame = run_stats_pass()

    aggregate = {
        "n_frames": len(per_frame),
        "focus_max_m": FOCUS_MAX_M,
        "valid_before_frac": summarize([f["valid_before_frac"] for f in per_frame]),
        "valid_after_frac": summarize([f["valid_after_frac"] for f in per_frame]),
        "newly_filled_frac": summarize([f["newly_filled_frac"] for f in per_frame]),
        "still_invalid_frac": summarize([f["still_invalid_frac"] for f in per_frame]),
        "mean_abs_change_m": summarize([f["mean_abs_change_m"] for f in per_frame]),
        "max_abs_change_m": summarize([f["max_abs_change_m"] for f in per_frame]),
        "roughness_before_m": summarize([f["roughness_before_m"] for f in per_frame]),
        "roughness_after_same_footprint_m": summarize([f["roughness_after_same_footprint_m"] for f in per_frame]),
        "edge_pixel_count": summarize([f["edge_pixel_count"] for f in per_frame]),
        "edge_grad_before_m_per_px": summarize([f["edge_grad_before_m_per_px"] for f in per_frame]),
        "edge_grad_after_m_per_px": summarize([f["edge_grad_after_m_per_px"] for f in per_frame]),
        "edge_sharpness_ratio": summarize([f["edge_sharpness_ratio"] for f in per_frame]),
    }
    rb = aggregate["roughness_before_m"]["mean"]
    ra = aggregate["roughness_after_same_footprint_m"]["mean"]
    aggregate["roughness_change_pct"] = (ra - rb) / rb * 100.0 if rb else None

    example_indices = choose_example_indices(per_frame, N_EXAMPLES)
    examples = run_example_pass(example_indices)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "before_bag": str(BEFORE_BAG), "after_bag": str(AFTER_BAG),
        "depth_topic": DEPTH_TOPIC, "rgb_topic": RGB_TOPIC,
        "focus_max_m": FOCUS_MAX_M,
        "aggregate": aggregate, "examples": examples, "per_frame": per_frame,
    }
    stats_path = RESULTS_DIR / "stats.json"
    stats_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\nWrote {stats_path}")
    print(f"Wrote {len(examples)} example image sets to {EXAMPLES_DIR}")
    print("\n--- Aggregate summary (depth <= 5m only) ---")
    print(f"Valid before:   mean={aggregate['valid_before_frac']['mean']:.1%}")
    print(f"Valid after:    mean={aggregate['valid_after_frac']['mean']:.1%}")
    print(f"Newly filled:   mean={aggregate['newly_filled_frac']['mean']:.1%}")
    print(f"Still invalid:  mean={aggregate['still_invalid_frac']['mean']:.1%}")
    print(f"Mean abs change: {aggregate['mean_abs_change_m']['mean']*100:.2f} cm (max seen: {aggregate['max_abs_change_m']['max']*100:.1f} cm)")
    print(f"Roughness [3x3]: before={rb*1000:.2f}mm after={ra*1000:.2f}mm change={aggregate['roughness_change_pct']:+.1f}%")
    print(f"Edge sharpness ratio (after/before depth-gradient at RGB edges): mean={aggregate['edge_sharpness_ratio']['mean']:.3f} median={aggregate['edge_sharpness_ratio']['median']:.3f}")
    print(f"  (n edge pixels/frame: mean={aggregate['edge_pixel_count']['mean']:.0f})")


if __name__ == "__main__":
    main()
