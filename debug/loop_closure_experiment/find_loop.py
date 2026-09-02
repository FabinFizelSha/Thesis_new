#!/usr/bin/env python3
"""Analyse /tesse/odom in the uHumans2 bag to locate spatial loop-closure
opportunities, and optionally cut a smaller loop-closure test bag.

Usage:
  python3 find_loop.py analyse [BAG]
  python3 find_loop.py extract T0 T1 OUT_BAG [BAG]     # bag-clock seconds
"""
import sys, os, math

import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from nav_msgs.msg import Odometry

DEFAULT_BAG = "/home/student/datasets/uhumans2/uHumans2_office_s1_00h_ros2"


def reader(bag):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id="sqlite3"),
           rosbag2_py.ConverterOptions("", ""))
    return r


def load_odom(bag):
    r = reader(bag)
    r.set_filter(rosbag2_py.StorageFilter(topics=["/tesse/odom"]))
    t0 = None
    rows = []
    while r.has_next():
        topic, data, t_ns = r.read_next()
        if topic != "/tesse/odom":
            continue
        m = deserialize_message(data, Odometry)
        t = t_ns * 1e-9
        if t0 is None:
            t0 = t
        p = m.pose.pose.position
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        rows.append((t - t0, p.x, p.y, p.z, yaw))
    return t0, rows


def analyse(bag):
    t0_abs, rows = load_odom(bag)
    print(f"bag: {bag}")
    print(f"/tesse/odom messages: {len(rows)}   duration: {rows[-1][0]:.1f}s   "
          f"abs start: {t0_abs:.3f}")

    # bin to ~5 Hz for the loop scan
    binned = []
    last_t = -1
    for (t, x, y, z, yaw) in rows:
        if t - last_t >= 0.2:
            binned.append((t, x, y, z, yaw))
            last_t = t
    xs = [b[1] for b in binned]; ys = [b[2] for b in binned]; zs = [b[3] for b in binned]
    print(f"extent  x:[{min(xs):.1f},{max(xs):.1f}]  y:[{min(ys):.1f},{max(ys):.1f}]  "
          f"z:[{min(zs):.1f},{max(zs):.1f}]  (m)")

    # path length
    L = 0.0
    for i in range(1, len(binned)):
        L += math.hypot(binned[i][1] - binned[i-1][1], binned[i][2] - binned[i-1][2])
    print(f"path length (xy): {L:.1f} m")

    # loop scan: for each pose i, earliest j with (t_i - t_j) > MIN_GAP and
    # xy distance < RADIUS.  Report distinct loop events (merge nearby hits).
    RADIUS = 1.5      # m  -- revisit proximity
    MIN_GAP = 25.0    # s  -- must be a real excursion, not just standing still
    events = []
    for i in range(len(binned)):
        ti, xi, yi = binned[i][0], binned[i][1], binned[i][2]
        for j in range(i):
            tj, xj, yj = binned[j][0], binned[j][1], binned[j][2]
            if ti - tj < MIN_GAP:
                break
            if math.hypot(xi - xj, yi - yj) <= RADIUS:
                events.append((tj, ti, math.hypot(xi - xj, yi - yj), ti - tj))
                break
    # collapse consecutive events into loop windows
    windows = []
    for (tj, ti, d, gap) in events:
        if windows and ti - windows[-1][1] < 5.0:
            w = windows[-1]
            windows[-1] = (min(w[0], tj), max(w[1], ti), min(w[2], d), max(w[3], gap))
        else:
            windows.append([tj, ti, d, gap])
    print(f"\n=== {len(windows)} candidate loop window(s) (radius {RADIUS} m, min gap {MIN_GAP} s) ===")
    for k, (tj, ti, d, gap) in enumerate(windows, 1):
        print(f"  L{k}: first-visit t~{tj:6.1f}s  revisit t~{ti:6.1f}s  "
              f"gap {gap:5.1f}s  closest approach {d:.2f} m")
        pad = 8.0
        c0 = max(0.0, tj - pad); c1 = min(rows[-1][0], ti + pad)
        print(f"      suggested cut  [{c0:.1f}, {c1:.1f}]  (bag-clock, {c1-c0:.1f}s)  "
              f"abs [{t0_abs + c0:.3f}, {t0_abs + c1:.3f}]")

    # ascii trajectory
    print("\n=== xy trajectory (start '@', end 'X', '.' path) ===")
    W, H = 78, 26
    def cell(x, y):
        cx = int((x - min(xs)) / max(1e-6, (max(xs) - min(xs))) * (W - 1))
        cy = int((y - min(ys)) / max(1e-6, (max(ys) - min(ys))) * (H - 1))
        return cx, H - 1 - cy
    grid = [[" "] * W for _ in range(H)]
    for b in binned:
        cx, cy = cell(b[1], b[2]); grid[cy][cx] = "."
    for (tj, ti, d, gap) in windows:
        for tt in (tj, ti):
            bb = min(binned, key=lambda r: abs(r[0] - tt))
            cx, cy = cell(bb[1], bb[2]); grid[cy][cx] = "O"
    b0 = binned[0]; b1 = binned[-1]
    cx, cy = cell(b0[1], b0[2]); grid[cy][cx] = "@"
    cx, cy = cell(b1[1], b1[2]); grid[cy][cx] = "X"
    for row in grid:
        print("  " + "".join(row))
    print("  (O = loop first-visit / revisit points)")


LATCHED_ALWAYS = ("/tf_static",)  # published once at t~0; must survive any cut


def extract(t0, t1, out, bag):
    """Write a new sqlite bag with all topics, keeping messages in [t0,t1]
    bag-clock.  Latched one-shot topics (``/tf_static``) are always copied and
    their timestamp is clamped forward to the window start so ``ros2 bag play``
    delivers them before the first real frame."""
    r = reader(bag)
    meta = r.get_metadata()
    base_ns = meta.starting_time.nanoseconds
    lo = base_ns + int(t0 * 1e9)
    hi = base_ns + int(t1 * 1e9)

    w = rosbag2_py.SequentialWriter()
    w.open(rosbag2_py.StorageOptions(uri=out, storage_id="sqlite3"),
           rosbag2_py.ConverterOptions("", ""))
    for tmeta in meta.topics_with_message_count:
        tm = tmeta.topic_metadata
        w.create_topic(rosbag2_py.TopicMetadata(
            name=tm.name, type=tm.type,
            serialization_format=tm.serialization_format,
            offered_qos_profiles=tm.offered_qos_profiles))

    n = 0
    latched = 0
    while r.has_next():
        topic, data, t_ns = r.read_next()
        if topic in LATCHED_ALWAYS and t_ns < lo:
            w.write(topic, data, lo)   # clamp forward into the window
            latched += 1
            continue
        if t_ns < lo:
            continue
        if t_ns > hi:
            break
        w.write(topic, data, t_ns)
        n += 1
    del w
    print(f"wrote {out}  ({n} messages + {latched} latched, bag-clock [{t0},{t1}])")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "analyse":
        analyse(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_BAG)
    elif len(sys.argv) >= 5 and sys.argv[1] == "extract":
        extract(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4],
                sys.argv[5] if len(sys.argv) > 5 else DEFAULT_BAG)
    else:
        print(__doc__)
