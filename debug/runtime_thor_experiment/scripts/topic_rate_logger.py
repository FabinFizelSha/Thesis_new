#!/usr/bin/env python3
"""Passive topic-rate logger (no publishing, no deserialisation of payloads).

Subscribes with raw=True (serialised CDR bytes) to the pipeline topics and
writes one CSV row per message: topic, receive wall-clock, receive ROS time
(sim time under use_sim_time), header stamp (if the message starts with
std_msgs/Header; blank otherwise) and payload size.

QoS mirrors the first discovered publisher (reliability/durability), so the
logger never forces a different QoS on the publisher side.

Usage: topic_rate_logger.py --out topic_rates.csv [--duration SEC]
"""
import argparse, csv, struct, time, threading, sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor
from rosidl_runtime_py.utilities import get_message

TOPICS = {
    "/rsg/preprocessed/frame": True,              # preprocessor -> Phase 1 (RsgFrame, Header first)
    "/rsg/hydra/rgb/image_raw": True,             # Phase 1 -> Hydra input (RGB)
    "/rsg/hydra/semantic/image_raw": True,        # Phase 1 -> Hydra input (semantic labels)
    "/hydra/backend/dsg": True,                   # Hydra output (DsgUpdate; stamp parse best-effort)
    "/rsg/hydra/fused_dsg": True,                 # Fuser (not published when publish_fused_dsg=false)
    "/rsg/scene_graph/markers": False,            # Fuser markers (MarkerArray, no header)
    "/rsg/objects/semantic_label_result": True,   # Phase 1 -> Fuser label events
    "/rsg/phase1/object_classifier/vlm_result": True,
}


def header_stamp(buf):
    # CDR little-endian: 4-byte encapsulation, then int32 sec, uint32 nanosec.
    if len(buf) < 12 or buf[1] != 1:
        return ""
    sec, nsec = struct.unpack_from("<iI", buf, 4)
    if sec < 0 or nsec >= 1_000_000_000:
        return ""
    return f"{sec + nsec * 1e-9:.9f}"


class Logger(Node):
    def __init__(self, out):
        super().__init__("rsg_topic_rate_logger", parameter_overrides=[
            rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)])
        self.f = open(out, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["topic", "wall_clock_unix_sec", "ros_time_sec", "header_stamp_sec", "bytes"])
        self.lock = threading.Lock()
        self.subs = {}
        self.counts = {t: 0 for t in TOPICS}
        self.create_timer(1.0, self.discover)

    def discover(self):
        names = dict(self.get_topic_names_and_types())
        for topic, has_header in TOPICS.items():
            if topic in self.subs or topic not in names:
                continue
            infos = self.get_publishers_info_by_topic(topic)
            rel, dur = ReliabilityPolicy.RELIABLE, DurabilityPolicy.VOLATILE
            if infos:
                q = infos[0].qos_profile
                rel, dur = q.reliability, q.durability
            qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1000, reliability=rel, durability=dur)
            msg_type = get_message(names[topic][0])
            self.subs[topic] = self.create_subscription(
                msg_type, topic, lambda m, t=topic, h=has_header: self.cb(t, m, h), qos, raw=True)
            self.get_logger().info(f"subscribed {topic} ({names[topic][0]}, {rel.name}/{dur.name})")

    def cb(self, topic, data, has_header):
        wall = time.time()
        ros = self.get_clock().now().nanoseconds * 1e-9
        buf = bytes(data)
        with self.lock:
            self.w.writerow([topic, f"{wall:.6f}", f"{ros:.6f}", header_stamp(buf) if has_header else "", len(buf)])
            self.counts[topic] += 1

    def close(self):
        with self.lock:
            self.f.flush(); self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=None)
    a = ap.parse_args()
    rclpy.init()
    node = Logger(a.out)
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(node)
    t0 = time.time()
    try:
        while rclpy.ok() and (a.duration is None or time.time() - t0 < a.duration):
            ex.spin_once(timeout_sec=0.5)
            if int(time.time() - t0) % 30 == 0:
                node.f.flush()
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        print("counts:", node.counts, file=sys.stderr)
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
