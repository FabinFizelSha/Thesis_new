#!/usr/bin/env python3
"""Analyze tracking output for overlapping bboxes to optimize matching threshold."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


def analyze_overlaps(lifecycles_file: str, output_file: str = None) -> Dict[str, Any]:
    """Analyze bbox overlaps from tracking lifecycle data.

    Args:
        lifecycles_file: Path to tracking_lifecycles_*.json file
        output_file: Optional path to save analysis report

    Returns:
        Dictionary with overlap analysis and threshold recommendations
    """
    with open(lifecycles_file) as f:
        lifecycles = json.load(f)

    overlaps = []
    track_ids = list(lifecycles.keys())

    # Analyze each track pair for bbox overlap
    for i in range(len(track_ids)):
        for j in range(i + 1, len(track_ids)):
            tid_i = track_ids[i]
            tid_j = track_ids[j]

            data_i = lifecycles[tid_i]
            data_j = lifecycles[tid_j]

            # Get bbox from observations
            obs_i = data_i.get('observations', [])
            obs_j = data_j.get('observations', [])

            if not obs_i or not obs_j:
                continue

            # Use final observation bbox as track bbox
            # (accumulated bbox from entire track lifecycle)
            bbox_i = _get_track_bbox(obs_i)
            bbox_j = _get_track_bbox(obs_j)

            if bbox_i is None or bbox_j is None:
                continue

            min_i, max_i = bbox_i
            min_j, max_j = bbox_j

            # Check if bboxes overlap
            overlap_min = np.maximum(min_i, min_j)
            overlap_max = np.minimum(max_i, max_j)

            if not np.all(overlap_min < overlap_max):
                continue

            # Calculate metrics
            overlap_volume = float(np.prod(np.maximum(0, overlap_max - overlap_min)))
            vol_i = float(np.prod(max_i - min_i))
            vol_j = float(np.prod(max_j - min_j))
            overlap_pct = 100.0 * overlap_volume / min(vol_i, vol_j)

            # XY distance
            cent_i = np.mean([o['centroid_3d'] for o in obs_i], axis=0)
            cent_j = np.mean([o['centroid_3d'] for o in obs_j], axis=0)
            xy_dist = float(np.linalg.norm(cent_i[:2] - cent_j[:2]))

            # Z levels
            z_i = np.mean([o['centroid_3d'][2] for o in obs_i])
            z_j = np.mean([o['centroid_3d'][2] for o in obs_j])

            if overlap_pct > 0:
                overlaps.append({
                    'track_i': tid_i,
                    'track_j': tid_j,
                    'overlap_pct': overlap_pct,
                    'xy_distance_m': xy_dist,
                    'z_i': float(z_i),
                    'z_j': float(z_j),
                    'z_diff': float(abs(z_i - z_j)),
                    'obs_count_i': len(obs_i),
                    'obs_count_j': len(obs_j),
                })

    # Sort by overlap percentage
    overlaps = sorted(overlaps, key=lambda x: x['overlap_pct'], reverse=True)

    # Analysis
    report = {
        'total_tracks': len(track_ids),
        'track_pairs_analyzed': len(track_ids) * (len(track_ids) - 1) // 2,
        'overlapping_pairs': len(overlaps),
        'overlap_statistics': {
            'max_pct': max([o['overlap_pct'] for o in overlaps], default=0),
            'mean_pct': float(np.mean([o['overlap_pct'] for o in overlaps])) if overlaps else 0,
            'min_pct': min([o['overlap_pct'] for o in overlaps], default=0),
            'pairs_over_10pct': sum(1 for o in overlaps if o['overlap_pct'] > 10),
            'pairs_over_20pct': sum(1 for o in overlaps if o['overlap_pct'] > 20),
            'pairs_over_50pct': sum(1 for o in overlaps if o['overlap_pct'] > 50),
        },
        'recommendations': [],
        'top_overlaps': overlaps[:20],  # Top 20 most overlapping pairs
    }

    # Recommendations based on overlap analysis
    if report['overlap_statistics']['pairs_over_50pct'] > len(track_ids) * 0.1:
        report['recommendations'].append(
            "High fragmentation detected (>50% overlap). Consider reducing matching threshold."
        )
    if report['overlap_statistics']['pairs_over_10pct'] > len(track_ids) * 0.5:
        report['recommendations'].append(
            "Many pairs with >10% overlap. Threshold optimization recommended."
        )
    if report['overlap_statistics']['max_pct'] > 90:
        report['recommendations'].append(
            f"Critical fragmentation: {report['overlap_statistics']['max_pct']:.1f}% overlap detected. "
            "Two nearly identical tracks should have merged."
        )

    # Save report
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_file}")

    # Print summary
    print("\n" + "="*70)
    print("TRACKING OVERLAP ANALYSIS REPORT")
    print("="*70)
    print(f"Total tracks: {report['total_tracks']}")
    print(f"Overlapping pairs: {report['overlapping_pairs']}")
    print(f"\nOverlap Statistics:")
    print(f"  Max overlap: {report['overlap_statistics']['max_pct']:.1f}%")
    print(f"  Mean overlap: {report['overlap_statistics']['mean_pct']:.1f}%")
    print(f"  Pairs >10%: {report['overlap_statistics']['pairs_over_10pct']}")
    print(f"  Pairs >20%: {report['overlap_statistics']['pairs_over_20pct']}")
    print(f"  Pairs >50%: {report['overlap_statistics']['pairs_over_50pct']}")

    if report['recommendations']:
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")

    print("\nTop overlapping pairs:")
    for i, overlap in enumerate(overlaps[:5], 1):
        print(f"  {i}. {overlap['track_i']} ↔ {overlap['track_j']}: "
              f"{overlap['overlap_pct']:.1f}% overlap, "
              f"XY dist: {overlap['xy_distance_m']:.2f}m, "
              f"Z diff: {overlap['z_diff']:.2f}m")

    print("="*70 + "\n")

    return report


def _get_track_bbox(observations: List[Dict]) -> tuple:
    """Extract accumulated bbox from observation sequence."""
    if not observations:
        return None

    # Get first and last centroids to estimate bbox growth
    first_obs = observations[0]['centroid_3d']

    # Estimate bbox from centroid (would need actual bbox data for accuracy)
    # For now, use centroid as proxy
    bbox_min = np.array(first_obs, dtype=np.float32)
    bbox_max = np.array(first_obs, dtype=np.float32)

    # Expand bbox to encompass all observations
    for obs in observations:
        centroid = np.array(obs['centroid_3d'], dtype=np.float32)
        bbox_min = np.minimum(bbox_min, centroid)
        bbox_max = np.maximum(bbox_max, centroid)

    # Add small padding to represent object size
    padding = 0.1
    bbox_min -= padding
    bbox_max += padding

    return (bbox_min, bbox_max)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_tracking_overlaps.py <lifecycles_json> [output_json]")
        print("\nExample:")
        print("  python3 analyze_tracking_overlaps.py tracking_lifecycles.json report.json")
        sys.exit(1)

    lifecycles_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(lifecycles_file).exists():
        print(f"Error: File not found: {lifecycles_file}")
        sys.exit(1)

    report = analyze_overlaps(lifecycles_file, output_file)
    sys.exit(0 if not report['recommendations'] else 1)
