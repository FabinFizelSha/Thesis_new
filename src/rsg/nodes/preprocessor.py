"""ROS 2 preprocessing node for the Risk Scene Graph pipeline.

This file intentionally contains only the main node and the data flow. Helper
logic is kept in ``nodes/support/preprocessor`` so the pipeline can be
understood by following the data through smaller modules.
"""

from __future__ import annotations

import copy
import json
import time
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import message_filters
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Imu
from std_msgs.msg import Float64MultiArray, String
from tf2_ros import TransformBroadcaster

from rsg.msg import RsgFrame

from nodes.support.preprocessor.config_loader import PreprocessorConfig
from nodes.support.preprocessor.frame_validator import FrameValidator
from nodes.support.preprocessor.image_converter import ImageConverter
from nodes.support.preprocessor.imu_buffer import ImuBuffer
from nodes.support.preprocessor.odom_buffer import OdomBuffer
from nodes.support.preprocessor.time_utils import stamp_to_float
from nodes.support.preprocessor.timing_excel_recorder import TimingExcelRecorder
from nodes.support.preprocessor.transform_math import TransformMath


class PreprocessorNode(Node):
    """Prepare synchronized RGB-D and robot-pose frames for downstream RSG nodes."""

    def __init__(self) -> None:
        """Initialize subscriptions, publishers, configuration, and buffers."""
        super().__init__("rsg_preprocessor")

        self.declare_parameter("config_file", "")
        config_file = self.get_parameter("config_file").get_parameter_value().string_value
        if not config_file:
            raise ValueError("Parameter 'config_file' must point to rsg_pipeline.yaml")

        self.config = PreprocessorConfig.from_yaml(config_file)
        self.set_parameters([
            rclpy.parameter.Parameter(
                "use_sim_time",
                rclpy.Parameter.Type.BOOL,
                self.config.use_sim_time,
            )
        ])

        self.bridge = CvBridge()
        self.converter = ImageConverter(self.bridge, self.config)
        self.validator = FrameValidator(self.config)
        self.odom_buffer = OdomBuffer(
            max_size=self.config.odom_buffer_size,
            tolerance_sec=self.config.odom_tolerance_sec,
            use_interpolation=self.config.use_odom_interpolation,
            assume_ordered=self.config.assume_ordered_messages,
        )
        self.imu_buffer: Optional[ImuBuffer] = None
        if self.config.use_camera_imu:
            self.imu_buffer = ImuBuffer(
                max_size=self.config.imu_buffer_size,
                tolerance_sec=self.config.imu_tolerance_sec,
                assume_ordered=self.config.assume_ordered_messages,
            )

        base_rotation = TransformMath.rpy_to_matrix(*self.config.base_to_camera_rpy)
        base_translation = np.array(self.config.base_to_camera_translation, dtype=np.float64)
        self.t_base_camera = TransformMath.make_transform(base_rotation, base_translation)

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self.config.sensor_qos_depth,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self.config.output_qos_depth,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        odom_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self.config.odom_qos_depth,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self._camera_info_lock = Lock()
        self._latest_camera_info: Optional[CameraInfo] = None

        self._rgb_received_count = 0
        self._depth_received_count = 0
        self._camera_info_received_count = 0
        self._synchronized_pair_count = 0
        self._published_frame_count = 0
        self._explicit_rejected_count = 0

        self.rgb_sub = message_filters.Subscriber(
            self,
            Image,
            self.config.rgb_topic,
            qos_profile=sensor_qos,
        )
        self.depth_sub = message_filters.Subscriber(
            self,
            Image,
            self.config.depth_topic,
            qos_profile=sensor_qos,
        )
        self.rgb_sub.registerCallback(self.rgb_received_callback)
        self.depth_sub.registerCallback(self.depth_received_callback)

        self.info_sub = self.create_subscription(
            CameraInfo,
            self.config.camera_info_topic,
            self.camera_info_callback,
            sensor_qos,
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=self.config.sync_queue_size,
            slop=self.config.rgb_depth_slop_sec,
        )
        self.sync.registerCallback(self.synced_callback)

        self.odom_sub = self.create_subscription(Odometry, self.config.odom_topic, self.odom_callback, odom_qos)
        self.imu_sub = None
        if self.config.use_camera_imu:
            self.imu_sub = self.create_subscription(Imu, self.config.imu_topic, self.imu_callback, sensor_qos)

        self.frame_pub = self.create_publisher(RsgFrame, self.config.frame_topic, output_qos)
        self.tf_broadcaster = TransformBroadcaster(self) if self.config.publish_camera_tf else None
        self.status_pub = self.create_publisher(String, self.config.status_topic, output_qos)
        self.timing_pub = None
        if self.config.timing_enabled and self.config.publish_timing_topic:
            self.timing_pub = self.create_publisher(Float64MultiArray, self.config.timing_topic, output_qos)

        self.timing_recorder = TimingExcelRecorder(
            enabled=(
                self.config.write_timing_excel
                and (self.config.timing_enabled or self.config.latency_record_dropped_to_excel)
            ),
            output_path=self.config.timing_excel_path,
            autosave_every=self.config.timing_excel_autosave_every,
            logger=self.get_logger(),
            sheet_name=self.config.timing_sheet_name,
        )

        self.sequence = 0
        self._last_published_rgb_time: Optional[float] = None
        self._rate_limited_count = 0
        self._warned_odom_frame_contract = False
        self._log_startup_summary()

    def _log_startup_summary(self) -> None:
        """Print the most useful startup settings."""
        self.get_logger().info("rsg_preprocessor started.")
        self.get_logger().info(f"RGB topic: {self.config.rgb_topic}")
        self.get_logger().info(f"Depth topic: {self.config.depth_topic}")
        self.get_logger().info(f"CameraInfo topic: {self.config.camera_info_topic}")
        self.get_logger().info(f"Odom topic: {self.config.odom_topic}")
        if self.config.use_camera_imu:
            self.get_logger().info(f"Camera IMU topic: {self.config.imu_topic}")
            self.get_logger().info(
                f"Camera IMU enabled: require_for_frame={self.config.require_imu_for_frame}, "
                f"tolerance={self.config.imu_tolerance_sec}s"
            )
        else:
            self.get_logger().info("Camera IMU support disabled in configuration.")
        self.get_logger().info(f"Output frame topic: {self.config.frame_topic}")
        self.get_logger().info(
            "Synchronization: RGB + depth only; CameraInfo cached separately; "
            f"sync_queue_size={self.config.sync_queue_size}, "
            f"sensor_qos_depth={self.config.sensor_qos_depth}"
        )
        self.get_logger().info(
            "Camera-pose composition: "
            f"{self.config.world_frame} -> {self.config.base_frame} -> {self.config.camera_frame}; "
            f"translation={self.config.base_to_camera_translation}, "
            f"rpy={self.config.base_to_camera_rpy}, "
            f"publish_camera_tf={self.config.publish_camera_tf}"
        )
        self.get_logger().info(
            "Validation checks: "
            f"resolution={self.config.check_resolution}, "
            f"depth_processing={self.config.single_pass_depth_processing_enabled}, "
            f"depth_ratio_check={self.config.check_invalid_depth_ratio}, "
            f"compute_depth_ratio={self.config.compute_invalid_depth_ratio}, "
            f"depth_ratio_every_n={self.config.invalid_depth_ratio_every_n_frames}, "
            f"require_odom={self.config.require_odom_for_frame}, "
            f"require_imu={self.config.require_imu_for_frame}"
        )
        self.get_logger().info(
            f"Timing enabled: {self.config.timing_enabled} "
            f"(measure={self.config.timing_measurement_enabled}, "
            f"write_excel={self.config.write_timing_excel}, "
            f"publish_topic={self.config.publish_timing_topic})"
        )
        self.get_logger().info(
            f"Latency guard: enabled={self.config.latency_guard_enabled}, "
            f"warn={self.config.latency_warn_delay_ms} ms, "
            f"drop={self.config.latency_drop_delay_ms} ms, "
            f"drop_exceeded={self.config.latency_drop_exceeded_frames}"
        )
        if self.config.max_publish_frequency_hz > 0.0:
            self.get_logger().info(
                f"Publish rate limiter: max={self.config.max_publish_frequency_hz:.3f} Hz "
                f"(min_interval={1.0 / self.config.max_publish_frequency_hz:.3f} s)"
            )
        else:
            self.get_logger().info("Publish rate limiter: disabled")
        if self.config.timing_enabled or (self.config.latency_guard_enabled and self.config.latency_record_dropped_to_excel):
            self.get_logger().info(f"Timing/debug Excel file: {self.config.timing_excel_path}")

    def should_skip_for_publish_rate(self, rgb_time: float) -> bool:
        """Return True when publishing this frame would exceed the configured max output rate.

        The limiter uses message timestamps instead of wall time so it also works
        for rosbag playback with ``use_sim_time``. A value <= 0 disables the
        limiter. If timestamps move backwards, the limiter resets and accepts
        the frame.
        """
        max_hz = float(self.config.max_publish_frequency_hz)
        if max_hz <= 0.0:
            return False
        last_time = self._last_published_rgb_time
        if last_time is None:
            return False
        if rgb_time < last_time:
            self._last_published_rgb_time = None
            return False
        min_interval = 1.0 / max_hz
        return (rgb_time - last_time) < (min_interval - 1e-6)

    def odom_callback(self, msg: Odometry) -> None:
        """Store incoming odometry and verify the configured Tesse frame contract once."""
        if not self._warned_odom_frame_contract:
            frame_ok = msg.header.frame_id == self.config.world_frame
            child_ok = msg.child_frame_id == self.config.base_frame
            if not frame_ok or not child_ok:
                self.get_logger().warn(
                    "Odometry frame contract differs from config: "
                    f"received {msg.header.frame_id!r} -> {msg.child_frame_id!r}, "
                    f"configured {self.config.world_frame!r} -> {self.config.base_frame!r}. "
                    "Camera pose composition may be geometrically invalid."
                )
            else:
                self.get_logger().info(
                    "Odometry frame contract verified: "
                    f"{self.config.world_frame} -> {self.config.base_frame}."
                )
            self._warned_odom_frame_contract = True
        self.odom_buffer.add(msg)

    def imu_callback(self, msg: Imu) -> None:
        """Store incoming camera IMU messages for timestamp association."""
        if self.imu_buffer is not None:
            self.imu_buffer.add(msg)

    def rgb_received_callback(self, _msg: Image) -> None:
        """Count raw RGB arrivals before message-filter synchronization."""
        self._rgb_received_count += 1

    def depth_received_callback(self, _msg: Image) -> None:
        """Count raw depth arrivals before message-filter synchronization."""
        self._depth_received_count += 1

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """Cache the latest camera calibration outside the image synchronizer."""
        self._camera_info_received_count += 1
        with self._camera_info_lock:
            self._latest_camera_info = msg

    def get_cached_camera_info(self) -> Optional[CameraInfo]:
        """Return the latest cached CameraInfo message."""
        with self._camera_info_lock:
            return self._latest_camera_info

    def synced_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        """Prepare and publish one synchronized RGB-D-pose frame."""
        processing_start = time.perf_counter() if (self.config.timing_enabled or self.config.latency_guard_enabled) else None

        self._synchronized_pair_count += 1
        self.sequence += 1
        rgb_time = stamp_to_float(rgb_msg.header.stamp)
        depth_time = stamp_to_float(depth_msg.header.stamp)
        info_msg = self.get_cached_camera_info()
        info_time = stamp_to_float(info_msg.header.stamp) if info_msg is not None else rgb_time
        rgb_depth_dt = abs(rgb_time - depth_time)
        rgb_camera_info_dt = abs(rgb_time - info_time)
        frame_id = self.make_frame_id(rgb_msg)

        if info_msg is None:
            self.reject_frame(
                frame_id=frame_id,
                reason="camera_info_unavailable",
                rgb_time=rgb_time,
                processing_start=processing_start,
                extra={"rgb_depth_dt_sec": rgb_depth_dt},
                rgb_depth_dt_sec=rgb_depth_dt,
            )
            return

        if self.should_skip_for_publish_rate(rgb_time):
            self._rate_limited_count += 1
            if self._rate_limited_count % self.config.publish_published_status_every_n_frames == 0:
                self.publish_status(
                    "skipped",
                    frame_id,
                    "publish_rate_limited",
                    rgb_time,
                    {
                        "max_publish_frequency_hz": self.config.max_publish_frequency_hz,
                        "last_published_rgb_time": self._last_published_rgb_time,
                        "rate_limited_count": self._rate_limited_count,
                    },
                )
            return

        resolution_error = self.validator.validate_resolution(rgb_msg, depth_msg)
        if resolution_error:
            self.reject_frame(
                frame_id=frame_id,
                reason=resolution_error,
                rgb_time=rgb_time,
                processing_start=processing_start,
                extra={"rgb_depth_dt_sec": rgb_depth_dt, "rgb_camera_info_dt_sec": rgb_camera_info_dt},
                rgb_depth_dt_sec=rgb_depth_dt,
            )
            return

        try:
            rgb_out = self.converter.forward_rgb(rgb_msg)
            if self.drop_if_latency_exceeded(
                frame_id=frame_id,
                rgb_time=rgb_time,
                processing_start=processing_start,
                stage="rgb_conversion",
                rgb_depth_dt_sec=rgb_depth_dt,
            ):
                return

            depth_out, invalid_depth_ratio = self.converter.prepare_depth(depth_msg, self.sequence)
            if self.drop_if_latency_exceeded(
                frame_id=frame_id,
                rgb_time=rgb_time,
                processing_start=processing_start,
                stage="optional_depth_processing",
                rgb_depth_dt_sec=rgb_depth_dt,
                invalid_depth_ratio=None if "invalid_depth_ratio" not in locals() else invalid_depth_ratio,
            ):
                return
        except Exception as exc:
            self.reject_frame(
                frame_id=frame_id,
                reason=str(exc),
                rgb_time=rgb_time,
                processing_start=processing_start,
                extra={"rgb_depth_dt_sec": rgb_depth_dt, "rgb_camera_info_dt_sec": rgb_camera_info_dt},
                rgb_depth_dt_sec=rgb_depth_dt,
            )
            return

        depth_error = self.validator.validate_depth_ratio(invalid_depth_ratio)
        if depth_error:
            self.reject_frame(
                frame_id=frame_id,
                reason=depth_error,
                rgb_time=rgb_time,
                processing_start=processing_start,
                extra={
                    "invalid_depth_ratio": invalid_depth_ratio,
                    "rgb_depth_dt_sec": rgb_depth_dt,
                    "rgb_camera_info_dt_sec": rgb_camera_info_dt,
                },
                rgb_depth_dt_sec=rgb_depth_dt,
                invalid_depth_ratio=invalid_depth_ratio,
            )
            return

        if self.drop_if_latency_exceeded(
            frame_id=frame_id,
            rgb_time=rgb_time,
            processing_start=processing_start,
            stage="depth_ratio_check",
            rgb_depth_dt_sec=rgb_depth_dt,
            invalid_depth_ratio=invalid_depth_ratio,
        ):
            return

        t_odom_base, odom_delta, odom_status = self.odom_buffer.lookup(rgb_time)
        if t_odom_base is None:
            if self.config.require_odom_for_frame:
                self.reject_frame(
                    frame_id=frame_id,
                    reason=odom_status,
                    rgb_time=rgb_time,
                    processing_start=processing_start,
                    extra={
                        "odom_delta_sec": odom_delta,
                        "rgb_depth_dt_sec": rgb_depth_dt,
                        "rgb_camera_info_dt_sec": rgb_camera_info_dt,
                        "odom_status": odom_status,
                        "invalid_depth_ratio": invalid_depth_ratio,
                    },
                    rgb_depth_dt_sec=rgb_depth_dt,
                    rgb_odom_dt_sec=odom_delta,
                    odom_status=odom_status,
                    invalid_depth_ratio=invalid_depth_ratio,
                )
                return
            t_odom_base = np.eye(4, dtype=np.float64)
            odom_delta = -1.0
            odom_status = f"{odom_status}_identity_pose_used"

        if self.drop_if_latency_exceeded(
            frame_id=frame_id,
            rgb_time=rgb_time,
            processing_start=processing_start,
            stage="odom_lookup",
            rgb_depth_dt_sec=rgb_depth_dt,
            rgb_odom_dt_sec=odom_delta,
            odom_status=odom_status,
            invalid_depth_ratio=invalid_depth_ratio,
        ):
            return

        imu_msg, imu_delta, imu_status = self.lookup_camera_imu(rgb_time)
        has_camera_imu = imu_msg is not None
        if self.config.use_camera_imu and self.config.require_imu_for_frame and not has_camera_imu:
            self.reject_frame(
                frame_id=frame_id,
                reason=imu_status,
                rgb_time=rgb_time,
                processing_start=processing_start,
                extra={
                    "imu_delta_sec": imu_delta,
                    "rgb_imu_dt_sec": imu_delta,
                    "imu_status": imu_status,
                    "rgb_depth_dt_sec": rgb_depth_dt,
                    "rgb_camera_info_dt_sec": rgb_camera_info_dt,
                    "rgb_odom_dt_sec": odom_delta,
                    "odom_status": odom_status,
                    "invalid_depth_ratio": invalid_depth_ratio,
                },
                rgb_depth_dt_sec=rgb_depth_dt,
                rgb_odom_dt_sec=odom_delta,
                odom_status=odom_status,
                invalid_depth_ratio=invalid_depth_ratio,
                imu_status=imu_status,
                rgb_imu_dt_sec=imu_delta,
            )
            return

        if self.drop_if_latency_exceeded(
            frame_id=frame_id,
            rgb_time=rgb_time,
            processing_start=processing_start,
            stage="imu_lookup",
            rgb_depth_dt_sec=rgb_depth_dt,
            rgb_odom_dt_sec=odom_delta,
            odom_status=odom_status,
            invalid_depth_ratio=invalid_depth_ratio,
            imu_status=imu_status,
            rgb_imu_dt_sec=imu_delta,
        ):
            return

        t_odom_camera = t_odom_base @ self.t_base_camera
        tx = t_odom_camera[:3, 3].astype(np.float64)
        rot_m = t_odom_camera[:3, :3].astype(np.float64)

        pose_msg = self.make_pose_msg(rgb_msg, tx, rot_m)
        self.publish_camera_tf(pose_msg)
        camera_info_out = self.make_camera_info_output(info_msg, rgb_msg)

        rgb_out.header.stamp = rgb_msg.header.stamp
        rgb_out.header.frame_id = self.config.camera_frame
        depth_out.header.stamp = rgb_msg.header.stamp
        depth_out.header.frame_id = self.config.camera_frame

        msg = self.make_rsg_frame_msg(
            rgb_msg=rgb_msg,
            rgb_out=rgb_out,
            depth_out=depth_out,
            camera_info_out=camera_info_out,
            imu_msg=imu_msg,
            imu_delta=imu_delta,
            has_camera_imu=has_camera_imu,
            pose_msg=pose_msg,
            tx=tx,
            rot_m=rot_m,
            frame_id=frame_id,
            invalid_depth_ratio=invalid_depth_ratio,
            rgb_depth_dt=rgb_depth_dt,
            rgb_camera_info_dt=rgb_camera_info_dt,
            odom_delta=odom_delta,
            odom_status=odom_status,
            imu_status=imu_status,
        )
        if self.drop_if_latency_exceeded(
            frame_id=frame_id,
            rgb_time=rgb_time,
            processing_start=processing_start,
            stage="message_assembly_before_publish",
            rgb_depth_dt_sec=rgb_depth_dt,
            rgb_odom_dt_sec=odom_delta,
            odom_status=odom_status,
            invalid_depth_ratio=invalid_depth_ratio,
            imu_status=imu_status,
            rgb_imu_dt_sec=imu_delta,
        ):
            return

        self.frame_pub.publish(msg)
        self._published_frame_count += 1
        self._last_published_rgb_time = rgb_time

        processing_delay_ms: Optional[float] = None
        if self.config.timing_enabled and processing_start is not None:
            processing_delay_ms = (time.perf_counter() - processing_start) * 1000.0
            self.publish_timing(
                frame_id=frame_id,
                rgb_time=rgb_time,
                processing_delay_ms=processing_delay_ms,
                rgb_depth_dt_sec=rgb_depth_dt,
                rgb_odom_dt_sec=float(odom_delta if odom_delta is not None else -1.0),
                odom_status=odom_status,
                invalid_depth_ratio=invalid_depth_ratio,
                imu_status=imu_status,
                rgb_imu_dt_sec=imu_delta,
            )

        status_extra: Dict[str, Any] = {
            "invalid_depth_ratio": invalid_depth_ratio,
            "rgb_depth_dt_sec": rgb_depth_dt,
            "rgb_camera_info_dt_sec": rgb_camera_info_dt,
            "rgb_odom_dt_sec": odom_delta,
            "odom_status": odom_status,
            "imu_enabled": self.config.use_camera_imu,
            "has_camera_imu": has_camera_imu,
            "rgb_imu_dt_sec": imu_delta,
            "imu_status": imu_status,
        }
        if processing_delay_ms is not None:
            status_extra["processing_delay_ms"] = processing_delay_ms

        if self.should_publish_success_status():
            self.publish_status("published", frame_id, "ok", rgb_time, status_extra)

    def make_camera_info_output(self, info_msg: CameraInfo, rgb_msg: Image) -> CameraInfo:
        """Create a timestamped CameraInfo view without deep-copying calibration arrays."""
        output = CameraInfo()
        output.header = copy.copy(info_msg.header)
        output.header.stamp = rgb_msg.header.stamp
        output.header.frame_id = self.config.camera_frame
        output.height = info_msg.height
        output.width = info_msg.width
        output.distortion_model = info_msg.distortion_model
        output.d = info_msg.d
        output.k = info_msg.k
        output.r = info_msg.r
        output.p = info_msg.p
        output.binning_x = info_msg.binning_x
        output.binning_y = info_msg.binning_y
        output.roi = copy.copy(info_msg.roi)
        return output

    def publish_camera_tf(self, pose_msg: PoseStamped) -> None:
        """Broadcast a synthetic world -> camera transform only when explicitly enabled.

        For the Tesse bag this is disabled: rosbag TF already publishes the
        authoritative world -> base_link_gt -> left_cam chain. Enabling this
        method for that bag would create a competing world -> left_cam authority.
        """
        if self.tf_broadcaster is None:
            return
        if not pose_msg.header.frame_id or not self.config.camera_frame:
            return

        tf_msg = TransformStamped()
        tf_msg.header.stamp = pose_msg.header.stamp
        tf_msg.header.frame_id = pose_msg.header.frame_id
        tf_msg.child_frame_id = self.config.camera_frame
        tf_msg.transform.translation.x = pose_msg.pose.position.x
        tf_msg.transform.translation.y = pose_msg.pose.position.y
        tf_msg.transform.translation.z = pose_msg.pose.position.z
        tf_msg.transform.rotation = pose_msg.pose.orientation
        try:
            self.tf_broadcaster.sendTransform(tf_msg)
        except Exception as exc:
            self.get_logger().warn(f"Failed to publish camera TF: {exc}")

    def make_rsg_frame_msg(
        self,
        rgb_msg: Image,
        rgb_out: Image,
        depth_out: Image,
        camera_info_out: CameraInfo,
        imu_msg: Optional[Imu],
        imu_delta: Optional[float],
        has_camera_imu: bool,
        pose_msg: PoseStamped,
        tx: np.ndarray,
        rot_m: np.ndarray,
        frame_id: str,
        invalid_depth_ratio: float,
        rgb_depth_dt: float,
        rgb_camera_info_dt: float,
        odom_delta: Optional[float],
        odom_status: str,
        imu_status: str,
    ) -> RsgFrame:
        """Create the output RsgFrame message."""
        msg = RsgFrame()
        msg.header.stamp = rgb_msg.header.stamp
        msg.header.frame_id = self.config.world_frame
        msg.rsg_frame_id = frame_id
        msg.source = self.config.source
        msg.sequence = self.sequence
        msg.rgb = rgb_out
        msg.depth_m = depth_out
        msg.camera_info = camera_info_out
        msg.has_camera_imu = bool(has_camera_imu)
        msg.camera_imu = imu_msg if imu_msg is not None else self.make_empty_imu_msg(rgb_msg)
        msg.rgb_imu_dt_sec = float(imu_delta if imu_delta is not None else -1.0)
        msg.camera_pose = pose_msg
        msg.tx = [float(v) for v in tx]
        msg.rot_m = [float(v) for v in rot_m.reshape(9)]
        msg.invalid_depth_ratio = float(invalid_depth_ratio)
        msg.rgb_depth_dt_sec = float(rgb_depth_dt)
        msg.rgb_camera_info_dt_sec = float(rgb_camera_info_dt)
        msg.rgb_odom_dt_sec = float(odom_delta if odom_delta is not None else -1.0)
        msg.metadata_json = json.dumps({
            "odom_status": odom_status,
            "imu_enabled": self.config.use_camera_imu,
            "imu_required": self.config.require_imu_for_frame,
            "imu_status": imu_status,
            "has_camera_imu": has_camera_imu,
            "imu_topic": self.config.imu_topic,
            "rgb_encoding_in": rgb_msg.encoding,
            "depth_encoding_in": depth_out.encoding,
            "rgb_frame_id_in": rgb_msg.header.frame_id,
            "camera_frame_out": self.config.camera_frame,
            "world_frame": self.config.world_frame,
            "base_frame": self.config.base_frame,
            "session_date": self.config.session_date,
            "checks": {
                "resolution": self.config.check_resolution,
                "invalid_depth_ratio": self.config.check_invalid_depth_ratio,
                "single_pass_depth_processing": self.config.single_pass_depth_processing_enabled,
                "require_odom": self.config.require_odom_for_frame,
                "require_imu": self.config.require_imu_for_frame,
            },
        })
        return msg

    def reject_frame(
        self,
        frame_id: str,
        reason: str,
        rgb_time: float,
        processing_start: Optional[float],
        extra: Optional[Dict[str, Any]] = None,
        rgb_depth_dt_sec: Optional[float] = None,
        rgb_odom_dt_sec: Optional[float] = None,
        odom_status: str = "",
        invalid_depth_ratio: Optional[float] = None,
        imu_status: Optional[str] = None,
        rgb_imu_dt_sec: Optional[float] = None,
    ) -> None:
        """Record and publish a rejected-frame debug entry."""
        self._explicit_rejected_count += 1
        payload_extra: Dict[str, Any] = dict(extra or {})
        processing_delay_ms: Optional[float] = None

        if self.config.timing_enabled and processing_start is not None:
            processing_delay_ms = (time.perf_counter() - processing_start) * 1000.0
            payload_extra["processing_delay_ms"] = processing_delay_ms
            if rgb_depth_dt_sec is None:
                rgb_depth_dt_sec = payload_extra.get("rgb_depth_dt_sec")
            if rgb_odom_dt_sec is None:
                rgb_odom_dt_sec = payload_extra.get("rgb_odom_dt_sec", payload_extra.get("odom_delta_sec"))
            if not odom_status:
                odom_status = str(payload_extra.get("odom_status", reason))
            if invalid_depth_ratio is None:
                invalid_depth_ratio = payload_extra.get("invalid_depth_ratio")
            if imu_status is None:
                imu_status = payload_extra.get("imu_status")
            if rgb_imu_dt_sec is None:
                rgb_imu_dt_sec = payload_extra.get("rgb_imu_dt_sec", payload_extra.get("imu_delta_sec"))

            self.record_timing_event(
                frame_id=frame_id,
                rgb_time=rgb_time,
                processing_delay_ms=processing_delay_ms,
                rgb_depth_dt_sec=rgb_depth_dt_sec,
                rgb_odom_dt_sec=rgb_odom_dt_sec,
                odom_status=odom_status,
                status="rejected",
                reason=reason,
                invalid_depth_ratio=invalid_depth_ratio,
                imu_status=imu_status,
                rgb_imu_dt_sec=rgb_imu_dt_sec,
                publish_timing_topic=False,
            )

        if self.config.publish_status_for_rejected_frames:
            self.publish_status("rejected", frame_id, reason, rgb_time, payload_extra)

    def lookup_camera_imu(self, rgb_time: float) -> Tuple[Optional[Imu], Optional[float], str]:
        """Look up the camera IMU sample associated with the RGB timestamp."""
        if not self.config.use_camera_imu:
            return None, None, "imu_disabled"
        if self.imu_buffer is None:
            return None, None, "imu_buffer_not_initialized"
        return self.imu_buffer.lookup(rgb_time)

    def make_empty_imu_msg(self, rgb_msg: Image) -> Imu:
        """Create a timestamped empty IMU message when IMU is unavailable."""
        imu_msg = Imu()
        imu_msg.header.stamp = rgb_msg.header.stamp
        imu_msg.header.frame_id = self.config.camera_frame
        return imu_msg

    def publish_timing(
        self,
        frame_id: str,
        rgb_time: float,
        processing_delay_ms: float,
        rgb_depth_dt_sec: float,
        rgb_odom_dt_sec: float,
        odom_status: str,
        invalid_depth_ratio: Optional[float] = None,
        imu_status: Optional[str] = None,
        rgb_imu_dt_sec: Optional[float] = None,
    ) -> None:
        """Publish and record one successful preprocessing timing sample."""
        self.record_timing_event(
            frame_id=frame_id,
            rgb_time=rgb_time,
            processing_delay_ms=processing_delay_ms,
            rgb_depth_dt_sec=rgb_depth_dt_sec,
            rgb_odom_dt_sec=rgb_odom_dt_sec,
            odom_status=odom_status,
            status="published",
            reason="ok",
            invalid_depth_ratio=invalid_depth_ratio,
            imu_status=imu_status,
            rgb_imu_dt_sec=rgb_imu_dt_sec,
            publish_timing_topic=True,
        )

    def record_timing_event(
        self,
        frame_id: str,
        rgb_time: float,
        processing_delay_ms: float,
        rgb_depth_dt_sec: Optional[float],
        rgb_odom_dt_sec: Optional[float],
        odom_status: str,
        status: str,
        reason: str,
        invalid_depth_ratio: Optional[float] = None,
        imu_status: Optional[str] = None,
        rgb_imu_dt_sec: Optional[float] = None,
        publish_timing_topic: bool = True,
        force_record_to_excel: bool = False,
    ) -> None:
        """Publish timing data and/or record an Excel timing/debug row."""
        if not self.config.timing_enabled and not force_record_to_excel:
            return
        if self.config.timing_enabled and publish_timing_topic and self.timing_pub is not None:
            timing_msg = Float64MultiArray()
            timing_msg.data = [
                float(self.sequence),
                float(processing_delay_ms),
                float(rgb_time),
                float(rgb_depth_dt_sec if rgb_depth_dt_sec is not None else -1.0),
                float(rgb_odom_dt_sec if rgb_odom_dt_sec is not None else -1.0),
            ]
            self.timing_pub.publish(timing_msg)

        if not self.timing_recorder.enabled:
            return

        self.timing_recorder.add_sample(
            sequence=self.sequence,
            frame_id=frame_id,
            rgb_time=rgb_time,
            processing_delay_ms=processing_delay_ms,
            rgb_depth_dt_sec=rgb_depth_dt_sec,
            rgb_odom_dt_sec=rgb_odom_dt_sec,
            odom_status=odom_status,
            status=status,
            reason=reason,
            invalid_depth_ratio=invalid_depth_ratio,
            imu_status=imu_status,
            rgb_imu_dt_sec=rgb_imu_dt_sec,
        )

    def elapsed_processing_ms(self, processing_start: Optional[float]) -> Optional[float]:
        """Return elapsed preprocessing time in milliseconds for the current callback."""
        if processing_start is None:
            return None
        return (time.perf_counter() - processing_start) * 1000.0

    def drop_if_latency_exceeded(
        self,
        frame_id: str,
        rgb_time: float,
        processing_start: Optional[float],
        stage: str,
        rgb_depth_dt_sec: Optional[float] = None,
        rgb_odom_dt_sec: Optional[float] = None,
        odom_status: str = "",
        invalid_depth_ratio: Optional[float] = None,
        imu_status: Optional[str] = None,
        rgb_imu_dt_sec: Optional[float] = None,
    ) -> bool:
        """Drop the current frame before publishing if it exceeds the latency budget.

        The guard is cooperative: it is checked at safe points between
        processing stages. It cannot interrupt a single blocking NumPy/cv_bridge
        operation while that operation is already running, but it prevents a
        frame that has already exceeded the configured budget from reaching the
        downstream worker.
        """
        if not self.config.latency_guard_enabled or not self.config.latency_drop_exceeded_frames:
            return False

        elapsed_ms = self.elapsed_processing_ms(processing_start)
        if elapsed_ms is None or elapsed_ms <= self.config.latency_drop_delay_ms:
            return False

        reason = "exceeded_processing_time_threshold"
        extra: Dict[str, Any] = {
            "latency_stage": stage,
            "processing_delay_ms": elapsed_ms,
            "latency_drop_threshold_ms": self.config.latency_drop_delay_ms,
            "latency_warn_threshold_ms": self.config.latency_warn_delay_ms,
            "rgb_depth_dt_sec": rgb_depth_dt_sec,
            "rgb_odom_dt_sec": rgb_odom_dt_sec,
            "odom_status": odom_status,
            "invalid_depth_ratio": invalid_depth_ratio,
            "imu_status": imu_status,
            "rgb_imu_dt_sec": rgb_imu_dt_sec,
        }
        self.record_timing_event(
            frame_id=frame_id,
            rgb_time=rgb_time,
            processing_delay_ms=elapsed_ms,
            rgb_depth_dt_sec=rgb_depth_dt_sec,
            rgb_odom_dt_sec=rgb_odom_dt_sec,
            odom_status=odom_status or reason,
            status="rejected",
            reason=reason,
            invalid_depth_ratio=invalid_depth_ratio,
            imu_status=imu_status,
            rgb_imu_dt_sec=rgb_imu_dt_sec,
            publish_timing_topic=False,
            force_record_to_excel=self.config.latency_record_dropped_to_excel,
        )
        if self.config.publish_status_for_rejected_frames:
            self.publish_status("rejected", frame_id, reason, rgb_time, extra)
        self.get_logger().warn(
            f"Dropped {frame_id}: preprocessing delay {elapsed_ms:.2f} ms exceeded "
            f"threshold {self.config.latency_drop_delay_ms:.2f} ms at stage '{stage}'."
        )
        return True

    def close(self) -> None:
        """Flush debug outputs before the node is destroyed."""
        self.timing_recorder.save()

    def should_publish_success_status(self) -> bool:
        """Return whether this accepted frame should produce a status message."""
        if not self.config.publish_status_for_published_frames:
            return False
        return self.sequence % self.config.publish_published_status_every_n_frames == 0

    def make_frame_id(self, rgb_msg: Image) -> str:
        """Create a repeat-safe frame id from date, data timestamp, and sequence."""
        stamp = rgb_msg.header.stamp
        return (
            f"{self.config.frame_id_prefix}_"
            f"{self.config.session_date}_"
            f"{stamp.sec}_{stamp.nanosec:09d}_"
            f"{self.sequence:06d}"
        )

    def make_pose_msg(self, rgb_msg: Image, tx: np.ndarray, rot_m: np.ndarray) -> PoseStamped:
        """Create a camera pose message from translation and rotation matrix."""
        qx, qy, qz, qw = TransformMath.matrix_to_quaternion(rot_m)
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rgb_msg.header.stamp
        pose_msg.header.frame_id = self.config.world_frame
        pose_msg.pose.position.x = float(tx[0])
        pose_msg.pose.position.y = float(tx[1])
        pose_msg.pose.position.z = float(tx[2])
        pose_msg.pose.orientation.x = float(qx)
        pose_msg.pose.orientation.y = float(qy)
        pose_msg.pose.orientation.z = float(qz)
        pose_msg.pose.orientation.w = float(qw)
        return pose_msg

    def publish_status(self, status: str, frame_id: str, reason: str, rgb_time: float, extra: Optional[Dict[str, Any]] = None) -> None:
        """Publish lightweight JSON status for debugging and bag inspection."""
        payload: Dict[str, Any] = {
            "status": status,
            "frame_id": frame_id,
            "reason": reason,
            "rgb_time": rgb_time,
            "sequence": self.sequence,
            "rgb_received_count": self._rgb_received_count,
            "depth_received_count": self._depth_received_count,
            "camera_info_received_count": self._camera_info_received_count,
            "synchronized_pair_count": self._synchronized_pair_count,
            "published_frame_count": self._published_frame_count,
            "explicit_rejected_count": self._explicit_rejected_count,
            "rate_limited_count": self._rate_limited_count,
        }
        if extra:
            payload.update(extra)
        self.status_pub.publish(String(data=json.dumps(payload)))


def main(args: Optional[List[str]] = None) -> None:
    """Run the :class:`PreprocessorNode` with graceful Ctrl+C handling."""
    rclpy.init(args=args)
    node: Optional[PreprocessorNode] = None
    try:
        node = PreprocessorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("Shutdown requested by user.")
    except rclpy.executors.ExternalShutdownException:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
