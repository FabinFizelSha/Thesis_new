#!/usr/bin/env python3
"""Inject a synthetic ``map -> odom`` loop-closure correction for testing.

The real pipeline runs on ground-truth odometry with ``map == odom == world``
and Hydra LCD off, so ``map -> odom`` is always identity and phase 1's
re-anchor path is never exercised.  This node stands in for the back-end: it
broadcasts ``map -> odom`` as identity until a scripted bag time ``t_jump_sec``
and then a fixed rigid step ``(dx, dy, dz, dyaw_deg)`` -- the same shape of
signal a pose-graph loop closure folds into that transform.

Run it instead of the static identity ``world -> odom`` bridge, with the
pipeline configured for split ``map`` / ``odom`` frames and
``phase1.loop_closure.enabled: true``.

    ros2 run <pkg> loop_jump_injector --ros-args \
        -p use_sim_time:=true -p t_jump_sec:=53.0 \
        -p dx:=-0.8 -p dy:=0.0 -p dyaw_deg:=3.0

or plainly:  python3 loop_jump_injector.py --ros-args -p use_sim_time:=true ...
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class LoopJumpInjector(Node):
    def __init__(self) -> None:
        super().__init__("loop_jump_injector")
        p = self.declare_parameters(
            "",
            [
                ("map_frame", "map"),
                ("odom_frame", "odom"),
                ("t_jump_sec", 53.0),      # bag-clock seconds from the first /clock stamp
                ("dx", -0.8),
                ("dy", 0.0),
                ("dz", 0.0),
                ("dyaw_deg", 3.0),
                ("ramp_sec", 0.0),         # 0 = instant step; >0 = linear over this long
                ("rate_hz", 50.0),
            ],
        )
        self.map_frame = self.get_parameter("map_frame").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.t_jump = float(self.get_parameter("t_jump_sec").value)
        self.dx = float(self.get_parameter("dx").value)
        self.dy = float(self.get_parameter("dy").value)
        self.dz = float(self.get_parameter("dz").value)
        self.dyaw = math.radians(float(self.get_parameter("dyaw_deg").value))
        self.ramp = max(0.0, float(self.get_parameter("ramp_sec").value))
        rate = max(1.0, float(self.get_parameter("rate_hz").value))

        self.br = TransformBroadcaster(self)
        self._t0 = None
        self._fired = False
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"loop_jump_injector: {self.map_frame}->{self.odom_frame} identity until "
            f"t={self.t_jump:.1f}s, then dt=({self.dx:.2f},{self.dy:.2f},{self.dz:.2f}) "
            f"dyaw={math.degrees(self.dyaw):.1f} deg over {self.ramp:.1f}s"
        )

    def _tick(self) -> None:
        now = self.get_clock().now()
        now_sec = now.nanoseconds * 1e-9
        if self._t0 is None:
            self._t0 = now_sec
        elapsed = now_sec - self._t0

        if elapsed < self.t_jump:
            frac = 0.0
        elif self.ramp <= 0.0 or elapsed >= self.t_jump + self.ramp:
            frac = 1.0
        else:
            frac = (elapsed - self.t_jump) / self.ramp
        if frac >= 1.0 and not self._fired:
            self._fired = True
            self.get_logger().warn(f"loop_jump_injector: correction fully applied at t={elapsed:.2f}s")

        tf = TransformStamped()
        tf.header.stamp = now.to_msg()
        tf.header.frame_id = self.map_frame
        tf.child_frame_id = self.odom_frame
        tf.transform.translation.x = self.dx * frac
        tf.transform.translation.y = self.dy * frac
        tf.transform.translation.z = self.dz * frac
        half = 0.5 * self.dyaw * frac
        tf.transform.rotation.z = math.sin(half)
        tf.transform.rotation.w = math.cos(half)
        self.br.sendTransform(tf)


def main() -> None:
    rclpy.init()
    node = LoopJumpInjector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
