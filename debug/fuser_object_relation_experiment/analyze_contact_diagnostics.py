#!/usr/bin/env python3
"""Offline verification for fuser object-contact/object-segment edges.

Reads the JSONL log written by fuser.cpp's writeContactDiagnostics() (one
line per fusion cycle: every main object's aggregate bbox, every member's
own bbox, and the contact/segment edges the fuser actually produced) and
independently recomputes the *expected* edges straight from the raw
geometry, using a from-scratch Python reimplementation of the algorithm in
src/rsg/nodes/fuser.cpp (aabbContact, the main-object broad-phase +
nearest-centroid narrow phase, and the segment formation-order chain). It
never reads the fuser's own edge-producing code path -- only geometry -- so
an implementation bug in fuser.cpp (wrong filter, wrong tie-break, a stale
edge, a duplicate) shows up as a mismatch here instead of silently agreeing
with itself.

Usage:
    python3 analyze_contact_diagnostics.py <path/to/log.jsonl> [--verbose]
    python3 analyze_contact_diagnostics.py <path/to/log.jsonl> --csv              # -> findings.csv next to the log
    python3 analyze_contact_diagnostics.py <path/to/log.jsonl> --csv out.csv      # explicit path

Enable the log by setting this fuser node param (off by default; see
rsg_scene_graph_fuser.yaml, which also documents object_contact_diagnostics_path
for a fixed-path override):
    object_contact_diagnostics_enabled: true
"""
import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Geometry: a faithful port of aabbContact() / contact_footprint_overlaps
# from src/rsg/nodes/fuser.cpp. Keep this in lockstep with that file by hand
# -- there is no shared source of truth, which is exactly the point: this is
# meant to be an independent reimplementation, not a wrapper around the C++.
# ---------------------------------------------------------------------------

FOOTPRINT_OVERLAP_MARGIN_M = 0.001


@dataclass
class AabbContact:
    touching: bool = False
    gap_m: float = 0.0
    contact_axis: int = -1
    footprint_overlaps: bool = False
    iou_3d: float = 0.0
    iou_xz: float = 0.0
    iou_yz: float = 0.0
    centroid_distance_m: float = 0.0


def _planar_iou(overlap_u, overlap_v, size_a_u, size_a_v, size_b_u, size_b_v):
    overlap_area = overlap_u * overlap_v
    area_a = max(0.0, size_a_u) * max(0.0, size_a_v)
    area_b = max(0.0, size_b_u) * max(0.0, size_b_v)
    denom = area_a + area_b - overlap_area
    return overlap_area / denom if denom > 1e-9 else 0.0


def aabb_contact(center_a, size_a, center_b, size_b, tolerance_m) -> AabbContact:
    gap = [0.0, 0.0, 0.0]
    overlap = [0.0, 0.0, 0.0]
    for i in range(3):
        half_sum = 0.5 * (max(0.0, size_a[i]) + max(0.0, size_b[i]))
        sep = abs(center_a[i] - center_b[i])
        gap[i] = sep - half_sum
        overlap[i] = max(0.0, -gap[i])

    out = AabbContact()
    out.contact_axis = max(range(3), key=lambda i: gap[i])
    out.gap_m = gap[out.contact_axis]
    out.touching = out.gap_m <= tolerance_m
    out.centroid_distance_m = sum((center_a[i] - center_b[i]) ** 2 for i in range(3)) ** 0.5

    out.footprint_overlaps = True
    for i in range(3):
        if i == out.contact_axis:
            continue
        if gap[i] >= -FOOTPRINT_OVERLAP_MARGIN_M:
            out.footprint_overlaps = False
            break

    padded = [0.0, 0.0, 0.0]
    for i in range(3):
        max_possible = min(max(0.0, size_a[i]), max(0.0, size_b[i]))
        padded[i] = min(max_possible, max(0.0, tolerance_m - gap[i]))
    out.iou_xz = _planar_iou(padded[0], padded[2], size_a[0], size_a[2], size_b[0], size_b[2])
    out.iou_yz = _planar_iou(padded[1], padded[2], size_a[1], size_a[2], size_b[1], size_b[2])

    if all(g <= 0.0 for g in gap):
        overlap_volume = overlap[0] * overlap[1] * overlap[2]
        vol_a = max(0.0, size_a[0]) * max(0.0, size_a[1]) * max(0.0, size_a[2])
        vol_b = max(0.0, size_b[0]) * max(0.0, size_b[1]) * max(0.0, size_b[2])
        denom = vol_a + vol_b - overlap_volume
        out.iou_3d = overlap_volume / denom if denom > 1e-9 else 0.0

    return out


# ---------------------------------------------------------------------------
# Main-object contact: broad-phase aggregate gate + narrow-phase nearest-
# centroid matching, mirroring computeObjectContacts() in fuser.cpp (reworked
# 2026-09-05, then again same day to match from the smaller side only). Once
# the aggregates touch, every node on the side with FEWER real members
# connects to its single closest counterpart on the other side by plain 3D
# centroid distance -- no bbox-intersection test, no per-pair distance
# cutoff, and matching runs from the smaller side only (matching both
# directions would force every node on the larger side to match whenever the
# smaller side has very few nodes, e.g. a 1-node wall next to a 5-node floor
# producing 5 wall-floor edges instead of the one real contact point). The
# C++ side additionally uses a spatial-hash grid to narrow candidate
# main-object pairs before this test -- that's a pure performance
# optimization (verified earlier to be equivalence-preserving) that this
# oracle skips in favor of a simple exhaustive O(G^2) pair scan, which is
# fine offline and doesn't depend on the grid's own correctness.
# ---------------------------------------------------------------------------


def _centroid_distance(center_a, center_b):
    return sum((center_a[i] - center_b[i]) ** 2 for i in range(3)) ** 0.5


@dataclass
class Group:
    internal_object_id: str
    members: list = field(default_factory=list)  # list of dicts: node_id(int), semantic_slot, has_bbox, bbox_center, bbox_size


def expected_contact_edges(groups, tolerance_m):
    """Returns {(source_id, target_id): AabbContact} for every expected edge."""
    bboxed_groups = []
    for g in groups:
        members_with_bbox = [m for m in g.members if m["has_bbox"]]
        if not members_with_bbox:
            continue
        mn = [min(m["bbox_center"][i] - 0.5 * m["bbox_size"][i] for m in members_with_bbox) for i in range(3)]
        mx = [max(m["bbox_center"][i] + 0.5 * m["bbox_size"][i] for m in members_with_bbox) for i in range(3)]
        agg_center = [0.5 * (mn[i] + mx[i]) for i in range(3)]
        agg_size = [mx[i] - mn[i] for i in range(3)]
        bboxed_groups.append((g, agg_center, agg_size))

    expected = {}
    for i in range(len(bboxed_groups)):
        for j in range(i + 1, len(bboxed_groups)):
            group_a, agg_a_c, agg_a_s = bboxed_groups[i]
            group_b, agg_b_c, agg_b_s = bboxed_groups[j]
            # Canonical side assignment: whichever group has the smaller
            # minimum member node_id is "a", matching fuser.cpp's geom_key
            # convention, so recomputed source/target sides line up with
            # what's logged for exact-tuple diffing.
            min_id_a = min(m["node_id"] for m in group_a.members)
            min_id_b = min(m["node_id"] for m in group_b.members)
            if min_id_b < min_id_a:
                group_a, group_b = group_b, group_a
                agg_a_c, agg_b_c = agg_b_c, agg_a_c
                agg_a_s, agg_b_s = agg_b_s, agg_a_s

            broad = aabb_contact(agg_a_c, agg_a_s, agg_b_c, agg_b_s, tolerance_m)
            if not broad.touching:
                continue

            members_a = [m for m in group_a.members if m["has_bbox"]]
            members_b = [m for m in group_b.members if m["has_bbox"]]
            if not members_a or not members_b:
                continue

            # (min_id, max_id) -> (ma, mb), ma always from group_a and mb
            # always from group_b regardless of which direction found the
            # pair -- matches fuser.cpp's ClosestPair convention so
            # source/target sides line up with what's logged.
            # Only the side with FEWER real members drives the matching (a
            # tie keeps group_a, the side with the smaller minimum NodeId).
            # Matching from both sides would force every node on the larger
            # side to match whenever the smaller side has very few nodes --
            # e.g. a 1-node wall next to a 5-node floor would produce 5
            # wall-floor edges (every floor node "finds" the wall, since
            # it's the only candidate), even though only the one floor node
            # actually nearest the wall is a sensible contact point.
            a_is_smaller = len(members_a) <= len(members_b)
            smaller_members = members_a if a_is_smaller else members_b
            larger_members = members_b if a_is_smaller else members_a

            selected = []
            for small in smaller_members:
                best = min(larger_members, key=lambda large: _centroid_distance(small["bbox_center"], large["bbox_center"]))
                # ma always from group_a, mb always from group_b, regardless
                # of which side turned out to be "smaller".
                selected.append((small, best) if a_is_smaller else (best, small))

            for ma, mb in selected:
                c = aabb_contact(ma["bbox_center"], ma["bbox_size"], mb["bbox_center"], mb["bbox_size"], tolerance_m)
                expected[(ma["node_id"], mb["node_id"])] = c
    return expected


def expected_segment_edges(groups):
    """Returns set of (source_id, target_id) for the formation-order chain."""
    expected = set()
    for g in groups:
        if not g.internal_object_id or len(g.members) < 2:
            continue
        members = sorted(g.members, key=lambda m: m["node_id"])
        for i in range(len(members) - 1):
            expected.add((members[i]["node_id"], members[i + 1]["node_id"]))
    return expected


# ---------------------------------------------------------------------------
# Log parsing + per-line verification
# ---------------------------------------------------------------------------


def parse_groups(line):
    groups = []
    for g in line["main_objects"]:
        members = []
        for m in g["members"]:
            member = {
                "node_id": int(m["node_id"]),
                "semantic_slot": m["semantic_slot_id"],
                "has_bbox": m["has_bbox"],
            }
            if m["has_bbox"]:
                member["bbox_center"] = m["bbox_center"]
                member["bbox_size"] = m["bbox_size"]
            members.append(member)
        groups.append(Group(internal_object_id=g["internal_object_id"], members=members))
    return groups


def _letter_suffix(index):
    """0->'a', 25->'z', 26->'aa', 27->'ab', ... (bijective base-26, Excel-column style,
    so a group with more than 26 members still gets distinct, readable suffixes)."""
    n = index + 1
    letters = []
    while n > 0:
        n -= 1
        letters.append(chr(ord("a") + n % 26))
        n //= 26
    return "".join(reversed(letters))


def build_display_ids(main_objects_json):
    """Maps node_id (int) -> a human-readable "1a"/"2b"-style id: main-object
    number = 1-based position in main_objects, member letter = position
    within that group's members list. Both arrays are already in a
    deterministic order on the C++ side (buildMainObjectGroups sorts groups
    by first-member NodeId and members by NodeId within each group), so no
    separate sort is needed here -- this just names the order that's
    already there, matching the "1a, 1b, 2a" shorthand used throughout the
    design discussion for this feature (see IMPLEMENTATION.md)."""
    display_id = {}
    for main_object_index, g in enumerate(main_objects_json):
        for member_index, m in enumerate(g["members"]):
            display_id[int(m["node_id"])] = f"{main_object_index + 1}{_letter_suffix(member_index)}"
    return display_id


def build_node_labels(main_objects_json):
    """Maps node_id (int) -> its resolved semantic label (e.g. "floor",
    "chair"; "" if phase 1/RAP hadn't labeled it yet this cycle). Purely for
    human identification -- never used by the verification logic, which
    only reasons about geometry. Requires a log written by a fuser build
    that includes the "label" field (added after the initial diagnostics
    feature); falls back to "" for older logs via .get()."""
    node_label = {}
    for g in main_objects_json:
        for m in g["members"]:
            node_label[int(m["node_id"])] = m.get("label", "")
    return node_label


@dataclass
class Finding:
    """One deviation between the logged edges and what the raw geometry
    says should be there. line_number/sequence/stamp_sec are filled in by
    main() after check_line() returns, since check_line() only sees one
    line at a time."""

    finding_type: str  # CONTACT_MISSING, CONTACT_EXTRA, SEGMENT_MISSING, SEGMENT_EXTRA, BOTH_TYPES, STRUCTURAL
    source: str = ""  # display id, e.g. "1b" ("" for a line-level finding like STRUCTURAL)
    target: str = ""  # display id, e.g. "2a"
    detail: str = ""  # free-text specifics (gap/IoU for a contact finding, the message for STRUCTURAL)
    line_number: int = 0
    sequence: object = None
    stamp_sec: object = None

    _LABELS = {
        "CONTACT_MISSING": "CONTACT missing: {pair} expected ({detail}) but not in logged contact_edges",
        "CONTACT_EXTRA": "CONTACT extra: {pair} logged but not expected from raw geometry ({detail})",
        "SEGMENT_MISSING": "SEGMENT missing: {pair} expected in the formation-order chain",
        "SEGMENT_EXTRA": "SEGMENT extra: {pair} logged but not expected",
        "BOTH_TYPES": "BOTH TYPES: {pair} is logged as both a contact edge and a segment edge",
        "SEGMENT_OVERLAP": "SEGMENT OVERLAP: {pair} ({detail})",
        "STRUCTURAL": "STRUCTURAL: {detail}",
    }

    def text(self):
        """Human-readable one-line rendering for the console (unchanged look
        from before CSV output existed)."""
        pair = f"{self.source}-{self.target}"
        template = self._LABELS.get(self.finding_type, "{finding_type}: {pair} {detail}")
        return template.format(pair=pair, detail=self.detail, finding_type=self.finding_type)


def check_line(line, verbose=False):
    """Returns a list of Finding objects (empty if the line is clean)."""
    findings = []
    groups = parse_groups(line)
    params = line["params"]
    display_id = build_display_ids(line["main_objects"])

    def fmt(node_id):
        return display_id.get(node_id, str(node_id))

    # --- Structural sanity on the log itself ---
    all_member_ids = {}
    for g in groups:
        for m in g.members:
            if m["node_id"] in all_member_ids:
                findings.append(Finding(
                    "STRUCTURAL",
                    detail=f"node {fmt(m['node_id'])} appears in two different main-object groups "
                           f"({all_member_ids[m['node_id']]!r} and {g.internal_object_id!r})",
                ))
            all_member_ids[m["node_id"]] = g.internal_object_id

    # --- Contact edges ---
    expected_contacts = expected_contact_edges(
        groups,
        params["object_contact_tolerance_m"],
    )
    logged_contacts = {}
    for edge in line.get("contact_edges", []):
        logged_contacts[(int(edge["source"]), int(edge["target"]))] = edge

    missing = set(expected_contacts) - set(logged_contacts)
    extra = set(logged_contacts) - set(expected_contacts)
    for pair in sorted(missing):
        c = expected_contacts[pair]
        findings.append(Finding(
            "CONTACT_MISSING", fmt(pair[0]), fmt(pair[1]),
            detail=f"gap={c.gap_m:.4f}, iou_xz={c.iou_xz:.3f}, iou_yz={c.iou_yz:.3f}",
        ))
    for pair in sorted(extra):
        findings.append(Finding(
            "CONTACT_EXTRA", fmt(pair[0]), fmt(pair[1]),
            detail=f"logged bbox_gap_m={logged_contacts[pair].get('bbox_gap_m')}",
        ))
    if verbose:
        for pair in sorted(set(expected_contacts) & set(logged_contacts)):
            print(f"    ok contact {fmt(pair[0])}-{fmt(pair[1])}")

    # --- Segment edges ---
    expected_segments = expected_segment_edges(groups)
    logged_segments = {(int(e["source"]), int(e["target"])) for e in line.get("segment_edges", [])}
    for pair in sorted(expected_segments - logged_segments):
        findings.append(Finding("SEGMENT_MISSING", fmt(pair[0]), fmt(pair[1])))
    for pair in sorted(logged_segments - expected_segments):
        findings.append(Finding("SEGMENT_EXTRA", fmt(pair[0]), fmt(pair[1])))

    # --- Cross-check: a pair should never be both a contact edge and a
    # segment edge (they're structurally disjoint -- contact edges only ever
    # connect different main-object groups, segment edges only ever connect
    # members of the same group). If this ever fires, it's exactly the
    # "dotted and solid line between the same two nodes" symptom.
    both = set(logged_contacts) & logged_segments
    for pair in sorted(both):
        findings.append(Finding("BOTH_TYPES", fmt(pair[0]), fmt(pair[1])))

    # --- Segment geometry cleanliness: two local segments of the SAME
    # physical object (same group) shouldn't substantially overlap in space
    # -- each should own a distinct section. Uses genuine 3D volumetric IoU
    # (all three axes must overlap), not the 2D plane IoUs -- two segments
    # that are the split object's dominant elongation axis apart (e.g. two
    # ends of a ceiling) can still share the *other* two dimensions (same
    # height/depth band) and register a high 2D-plane IoU on that unrelated
    # plane even though they don't actually overlap at all; iou_3d requires
    # every axis to overlap, so it only fires for a real spatial collision.
    # A little overlap right at a hand-off is normal (bounded by the
    # tracker's own gap tolerance); a large one is the "perfect seam"
    # regression this check exists to catch. Reuses aabb_contact purely for
    # its IoU math, not the touching/footprint semantics it was written for.
    for g in groups:
        bboxed = [m for m in g.members if m["has_bbox"]]
        for i in range(len(bboxed)):
            for j in range(i + 1, len(bboxed)):
                ma, mb = bboxed[i], bboxed[j]
                c = aabb_contact(ma["bbox_center"], ma["bbox_size"], mb["bbox_center"], mb["bbox_size"], 0.0)
                if c.iou_3d > 0.02:
                    findings.append(Finding(
                        "SEGMENT_OVERLAP", fmt(ma["node_id"]), fmt(mb["node_id"]),
                        detail=f"same physical object ({g.internal_object_id}), iou_3d={c.iou_3d:.3f} "
                               f"(gap_m={c.gap_m:.3f})",
                    ))

    return findings


CONTACT_EDGE_FIELDNAMES = [
    "sequence", "stamp_sec", "source", "target", "source_label", "target_label",
    "source_internal_object_id", "target_internal_object_id",
    "centroid_distance_m", "bbox_gap_m", "bbox_iou_3d", "bbox_iou_xz", "bbox_iou_yz",
    "bbox_iou_2d_max", "contact_axis", "source_slot_id", "target_slot_id",
    "source_group_size", "target_group_size", "source_node_id", "target_node_id",
]
SEGMENT_EDGE_FIELDNAMES = [
    "sequence", "stamp_sec", "source", "target", "source_label", "target_label",
    "internal_object_id", "group_size",
    "source_slot_id", "target_slot_id", "source_node_id", "target_node_id",
]


def export_raw_csv(log_path, contacts_csv_path, segments_csv_path, max_lines=None):
    """Raw data dump, not a verification report: every contact_edges/
    segment_edges entry logged across all cycles, flattened one row per
    edge-per-cycle into two CSV files, with the same readable 1a/1b/2a
    display ids used everywhere else in this script. Unlike check_line(),
    this doesn't recompute anything or compare against expectations -- it's
    just the logged data made browsable in a spreadsheet. Returns
    (contact_row_count, segment_row_count)."""
    contact_rows = 0
    segment_rows = 0
    with open(log_path, "r") as f, \
         open(contacts_csv_path, "w", newline="") as cf, \
         open(segments_csv_path, "w", newline="") as sf:
        contact_writer = csv.DictWriter(cf, fieldnames=CONTACT_EDGE_FIELDNAMES)
        contact_writer.writeheader()
        segment_writer = csv.DictWriter(sf, fieldnames=SEGMENT_EDGE_FIELDNAMES)
        segment_writer.writeheader()

        lines_seen = 0
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            if max_lines and lines_seen >= max_lines:
                break
            lines_seen += 1
            line = json.loads(raw_line)
            display_id = build_display_ids(line["main_objects"])
            node_label = build_node_labels(line["main_objects"])
            seq = line.get("sequence", "")
            stamp = line.get("stamp_sec", "")

            for edge in line.get("contact_edges", []):
                source_id = int(edge["source"])
                target_id = int(edge["target"])
                contact_writer.writerow({
                    "sequence": seq, "stamp_sec": stamp,
                    "source": display_id.get(source_id, source_id),
                    "target": display_id.get(target_id, target_id),
                    "source_label": node_label.get(source_id, ""),
                    "target_label": node_label.get(target_id, ""),
                    "source_internal_object_id": edge.get("source_internal_object_id"),
                    "target_internal_object_id": edge.get("target_internal_object_id"),
                    "centroid_distance_m": edge.get("centroid_distance_m"),
                    "bbox_gap_m": edge.get("bbox_gap_m"),
                    "bbox_iou_3d": edge.get("bbox_iou_3d"),
                    "bbox_iou_xz": edge.get("bbox_iou_xz"),
                    "bbox_iou_yz": edge.get("bbox_iou_yz"),
                    "bbox_iou_2d_max": edge.get("bbox_iou_2d_max"),
                    "contact_axis": edge.get("contact_axis"),
                    "source_slot_id": edge.get("source_slot_id"),
                    "target_slot_id": edge.get("target_slot_id"),
                    "source_group_size": edge.get("source_group_size"),
                    "target_group_size": edge.get("target_group_size"),
                    "source_node_id": source_id,
                    "target_node_id": target_id,
                })
                contact_rows += 1

            for edge in line.get("segment_edges", []):
                source_id = int(edge["source"])
                target_id = int(edge["target"])
                segment_writer.writerow({
                    "sequence": seq, "stamp_sec": stamp,
                    "source": display_id.get(source_id, source_id),
                    "target": display_id.get(target_id, target_id),
                    "source_label": node_label.get(source_id, ""),
                    "target_label": node_label.get(target_id, ""),
                    "internal_object_id": edge.get("internal_object_id"),
                    "group_size": edge.get("group_size"),
                    "source_slot_id": edge.get("source_slot_id"),
                    "target_slot_id": edge.get("target_slot_id"),
                    "source_node_id": source_id,
                    "target_node_id": target_id,
                })
                segment_rows += 1

    return contact_rows, segment_rows


CSV_FIELDNAMES = ["line_number", "sequence", "stamp_sec", "finding_type", "source", "target", "detail"]
_CSV_AUTO = object()  # sentinel: --csv given with no path -> auto-derive one next to the input log


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_path", help="Path to the JSONL diagnostics log")
    parser.add_argument("--verbose", action="store_true", help="Print every matched edge, not just mismatches")
    parser.add_argument("--max-lines", type=int, default=None, help="Stop after this many lines")
    parser.add_argument("--csv", metavar="PATH", nargs="?", const=_CSV_AUTO,
                        help="Also write every finding (a mismatch against the recomputed geometry) as a "
                             "CSV row, columns: " + ", ".join(CSV_FIELDNAMES) + ". Empty (header only) if "
                             "nothing was wrong. With no PATH, defaults to findings.csv next to the input log.")
    parser.add_argument("--export-csv", metavar="DIR", nargs="?", const=_CSV_AUTO,
                        help="Export the raw logged data (not a verification report) as two CSV files, "
                             "contact_edges.csv and segment_edges.csv, one row per edge observed per cycle "
                             "-- every contact/segment edge the fuser actually produced, browsable in a "
                             "spreadsheet, regardless of whether it matched expectations. With no DIR, "
                             "writes into the same directory as the input log.")
    args = parser.parse_args()
    csv_path = args.csv
    if csv_path is _CSV_AUTO:
        csv_path = os.path.join(os.path.dirname(args.log_path), "findings.csv")
    export_dir = args.export_csv
    if export_dir is _CSV_AUTO:
        export_dir = os.path.dirname(args.log_path)

    if args.export_csv is not None:
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
        contacts_csv_path = os.path.join(export_dir, "contact_edges.csv")
        segments_csv_path = os.path.join(export_dir, "segment_edges.csv")
        contact_rows, segment_rows = export_raw_csv(
            args.log_path, contacts_csv_path, segments_csv_path, max_lines=args.max_lines)
        print(f"exported {contact_rows} contact-edge rows to {contacts_csv_path}")
        print(f"exported {segment_rows} segment-edge rows to {segments_csv_path}")

    total_lines = 0
    clean_lines = 0
    all_findings = []
    with open(args.log_path, "r") as f:
        for line_number, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            if args.max_lines and total_lines >= args.max_lines:
                break
            total_lines += 1
            try:
                line = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                print(f"line {line_number}: JSON parse error: {exc}", file=sys.stderr)
                continue
            findings = check_line(line, verbose=args.verbose)
            if findings:
                seq = line.get("sequence", "?")
                stamp = line.get("stamp_sec", "?")
                print(f"line {line_number} (sequence={seq}, stamp_sec={stamp}):")
                for finding in findings:
                    finding.line_number = line_number
                    finding.sequence = seq
                    finding.stamp_sec = stamp
                    print(f"  {finding.text()}")
                all_findings.extend(findings)
            else:
                clean_lines += 1

    print()
    print(f"checked {total_lines} cycles: {clean_lines} clean, "
          f"{total_lines - clean_lines} with findings ({len(all_findings)} findings total)")

    if csv_path:
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for finding in all_findings:
                writer.writerow({
                    "line_number": finding.line_number,
                    "sequence": finding.sequence,
                    "stamp_sec": finding.stamp_sec,
                    "finding_type": finding.finding_type,
                    "source": finding.source,
                    "target": finding.target,
                    "detail": finding.detail,
                })
        print(f"wrote {len(all_findings)} findings to {csv_path}")

    sys.exit(1 if all_findings else 0)


if __name__ == "__main__":
    main()
