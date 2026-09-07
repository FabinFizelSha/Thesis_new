#!/usr/bin/env python3
"""Report duplicate object nodes per semantic slot in a saved DSG.

One semantic slot == one physical object in this pipeline (phase 1 allocates a
slot per tracked object), so more than one object node on a slot is a
duplicate. That is the signature of a resumed session where Hydra created a
fresh node stacked on the restored one instead of recognising it.

Run it against Hydra's save to see what the fuser's collapse has to deal with:

    python3 debug/check_duplicate_objects.py memory/hydra/backend/dsg.json

It also replays the fuser's collapse predicate (fuser.cpp
collapseDuplicateObjects) so the expected post-collapse node count can be
checked without launching the pipeline. The fuser's own report of the same
thing is `summary.duplicate_slot_count` in its object metadata export, which
should be near zero once the collapse is on.
"""

import argparse
import json
import math
import sys
from collections import defaultdict

OBJECT_LAYER = 2


def _vec(value):
    if isinstance(value, list):
        return value
    return [value["x"], value["y"], value["z"]]


def _bounds(attrs):
    """Return (min, max) corners, or None when the node has no valid box."""
    box = attrs.get("bounding_box") or {}
    if "dimensions" not in box or "world_P_center" not in box:
        return None
    center = _vec(box["world_P_center"])
    dims = _vec(box["dimensions"])
    return (
        [center[i] - dims[i] / 2.0 for i in range(3)],
        [center[i] + dims[i] / 2.0 for i in range(3)],
    )


def _contains(bounds, point):
    if not bounds:
        return False
    lo, hi = bounds
    return all(lo[i] - 1e-6 <= point[i] <= hi[i] + 1e-6 for i in range(3))


def _union(a, b):
    if not a:
        return b
    if not b:
        return a
    return (
        [min(a[0][i], b[0][i]) for i in range(3)],
        [max(a[1][i], b[1][i]) for i in range(3)],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dsg", help="path to a saved dsg.json / dsg_with_mesh.json")
    parser.add_argument("--max-distance", type=float, default=2.0,
                        help="centroid fallback, matching object_slot_collapse_max_distance_m")
    args = parser.parse_args()

    with open(args.dsg) as handle:
        graph = json.load(handle)

    objects = [
        node for node in graph.get("nodes", [])
        if node.get("layer") == OBJECT_LAYER
        and node.get("attributes", {}).get("type") == "ObjectNodeAttributes"
    ]

    by_slot = defaultdict(list)
    for node in objects:
        attrs = node["attributes"]
        by_slot[attrs["semantic_label"]].append(
            (node["id"], _vec(attrs["position"]), _bounds(attrs), attrs.get("is_active"))
        )

    duplicates = {slot: m for slot, m in by_slot.items() if len(m) > 1}
    print(f"{len(objects)} object nodes | {len(by_slot)} distinct slots | "
          f"{len(duplicates)} slot(s) with more than one node")

    absorbed = 0
    for slot, members in sorted(duplicates.items()):
        members.sort(key=lambda m: m[0])
        # An archived node always outranks an active one as survivor, and among
        # equals the lowest id (oldest) wins -- so a resumed run keeps the node
        # it restored rather than a freshly created one.
        survivor_id, survivor_pos, survivor_box, survivor_active = min(
            members, key=lambda m: (bool(m[3]), m[0])
        )
        notes = []
        for other_id, other_pos, other_box, other_active in members:
            if other_id == survivor_id:
                continue
            if not other_active:
                # Restored from a previous session: never removed.
                notes.append("archived->kept")
                continue
            distance = math.dist(survivor_pos, other_pos)
            # Same predicate as the fuser: either box holding the other's
            # centroid, else centroid distance.
            overlap = _contains(survivor_box, other_pos) or _contains(other_box, survivor_pos)
            if overlap or distance <= args.max_distance:
                absorbed += 1
                if survivor_active:
                    # Active-only slot: the survivor's box grows as it absorbs,
                    # making the collapse transitive. A restored survivor keeps
                    # its saved extent instead.
                    survivor_box = _union(survivor_box, other_box)
                notes.append(f"{distance:.2f}m {'bbox' if overlap else 'dist'}->collapse")
            else:
                notes.append(f"{distance:.2f}m KEPT APART")
        active = [m[3] for m in members]
        print(f"  slot {slot:4d}: {len(members)} nodes active={active} | " + ", ".join(notes))

    remaining = len(objects) - absorbed
    print(f"\ncollapse would absorb {absorbed} node(s) -> {remaining} object nodes")
    kept = sum(len(m) for m in duplicates.values()) - len(duplicates) - absorbed
    if kept:
        print(f"{kept} same-slot node(s) stay separate (archived, or beyond "
              f"{args.max_distance}m with no bbox overlap)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
