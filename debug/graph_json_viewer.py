#!/usr/bin/env python3
"""View a fuser graph_json snapshot in RViz.

Set RViz Fixed Frame to "graph_json_view" (a static identity child of the
snapshot frame, published by this script).

Loads a saved /rsg/scene_graph/graph_json message (file mode) or follows the
live topic (--live) and republishes it as a MarkerArray on
/rsg/graph_json_viewer/markers.
"""
import argparse
import json
import sys

import rclpy
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA, String
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

EDGE_STYLE = {  # type -> (r, g, b, a, width_m)
    "room_place_hierarchy": (0.0, 0.0, 0.0, 0.5, 0.02),
    "place_object_membership": (0.3, 0.3, 0.3, 0.5, 0.01),
    "place_connectivity": (0.0, 0.4, 0.9, 0.7, 0.03),
    "native_object_object": (0.0, 0.0, 0.0, 0.8, 0.02),
    "native_room_room": (0.0, 0.0, 0.0, 0.8, 0.04),
    "derived_object_contact": (0.9, 0.1, 0.1, 0.9, 0.03),
    "derived_object_segment": (0.9, 0.6, 0.0, 0.9, 0.02),
}
VIEW_FRAME = "graph_json_view"
NODE_SIZE = {"object": None, "place": 0.6, "room": 1.5, "building": 2.0}  # None -> display.diameter_m
MIN_OBJECT_SIZE = 0.3  # fuser spheres are ~0.07 m: invisible from a normal RViz distance


def marker(ns, mid, mtype, frame):
    m = Marker()
    m.header.frame_id = frame
    m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
    m.pose.orientation.w = 1.0
    return m


def build(graph, show_labels, hide_layers, node_scale=1.0):
    frame = graph.get("frame_id") or "world"
    out = MarkerArray()
    clear = Marker()
    clear.header.frame_id = frame
    clear.action = Marker.DELETEALL
    out.markers.append(clear)
    # One SPHERE_LIST per layer (uniform size, per-point colour) instead of one
    # SPHERE marker per node: lighter for RViz and rendered as point sprites.
    spheres = {}
    for i, node in enumerate(graph["nodes"]):
        if node["layer"] in hide_layers:
            continue
        pos, disp = node["display_position"], node.get("display", {})
        size = (NODE_SIZE.get(node["layer"]) or MIN_OBJECT_SIZE) * node_scale
        c = disp.get("color", {"r": 0.6, "g": 0.6, "b": 0.6, "a": 0.9})
        layer = spheres.get(node["layer"])
        if layer is None:
            layer = spheres[node["layer"]] = marker("graph_json_nodes", len(spheres), Marker.SPHERE_LIST, frame)
            layer.scale.x = layer.scale.y = layer.scale.z = float(size)
            layer.color.r = layer.color.g = layer.color.b = layer.color.a = 1.0
            out.markers.append(layer)
        layer.points.append(Point(x=pos["x"], y=pos["y"], z=pos["z"]))
        col = ColorRGBA(r=c["r"], g=c["g"], b=c["b"], a=max(c["a"], 0.5))
        layer.colors.append(col)
        label = disp.get("label") or node.get("name") or ""
        if show_labels and label:
            t = marker("graph_json_labels", i, Marker.TEXT_VIEW_FACING, frame)
            t.pose.position.x, t.pose.position.y = pos["x"], pos["y"]
            t.pose.position.z = pos["z"] + size / 2 + 0.1
            t.scale.z = 0.15
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = label
            out.markers.append(t)
    lines = {}
    for e in graph["edges"]:
        lines.setdefault(e["type"], []).append(e)
    for i, (etype, edges) in enumerate(sorted(lines.items())):
        r, g, b, a, w = EDGE_STYLE.get(etype, (0.5, 0.5, 0.5, 0.8, 0.02))
        m = marker("graph_json_edges", i, Marker.LINE_LIST, frame)
        m.scale.x = w
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
        for e in edges:
            for p in (e["source_position"], e["target_position"]):
                m.points.append(Point(x=p["x"], y=p["y"], z=p["z"]))
        out.markers.append(m)
    return out


class Viewer(Node):
    def __init__(self, args):
        super().__init__("graph_json_viewer")
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.args = args
        # RViz draws nothing while its Fixed Frame is missing from /tf. The
        # snapshot's frame (e.g. "world") only exists during a live run, so
        # publish a static identity transform to a frame RViz can be pointed at.
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.tf_parent = None
        self.pub = self.create_publisher(MarkerArray, args.out_topic, qos)
        if args.live:
            self.create_subscription(String, args.topic, self.on_msg, qos)
            self.get_logger().info(f"following {args.topic}")
        else:
            with open(args.file) as f:
                self.graph = json.load(f)
            self.publish()
            self.create_timer(2.0, self.publish)  # keep alive for late RViz starts

    def on_msg(self, msg):
        self.graph = json.loads(msg.data)
        self.publish()

    def publish_tf(self, frame):
        if frame == self.tf_parent:
            return
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = frame
        tf.child_frame_id = VIEW_FRAME
        tf.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(tf)
        self.tf_parent = frame

    def publish(self):
        self.publish_tf(self.graph.get("frame_id") or "world")
        arr = build(self.graph, not self.args.no_labels, set(self.args.hide), self.args.node_scale)
        self.pub.publish(arr)
        self.get_logger().info(
            f"published {len(arr.markers)} markers (frame '{arr.markers[0].header.frame_id}', "
            f"{self.graph['node_count']} nodes, {self.graph['edge_count']} edges)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file", nargs="?", help="saved graph_json message (JSON)")
    ap.add_argument("--live", action="store_true", help="follow --topic instead of a file")
    ap.add_argument("--topic", default="/rsg/scene_graph/graph_json")
    ap.add_argument("--out-topic", default="/rsg/graph_json_viewer/markers")
    ap.add_argument("--node-scale", type=float, default=1.0, help="multiply all sphere sizes")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--hide", nargs="*", default=[], choices=list(NODE_SIZE), help="node layers to hide")
    args = ap.parse_args()
    if not args.live and not args.file:
        ap.error("give a file, or use --live")
    rclpy.init()
    try:
        rclpy.spin(Viewer(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
