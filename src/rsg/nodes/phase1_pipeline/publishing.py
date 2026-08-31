"""Hydra message publishing pipeline stage."""

from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String
from rsg.msg import RsgFrame, RsgHydraFrame
from nodes.support.phase1.time_utils import stamp_to_float
from nodes.support.phase1.json_utils import safe_json_loads, safe_json_dumps


class PublishingStage:
    """Wraps Hydra message publishing logic."""

    def __init__(self, config: Any, logger: Any, bridge: Any = None, tf_broadcaster: Any = None):
        """Initialize publishing stage.

        Args:
            config: Phase1 configuration
            logger: ROS logger
            bridge: CvBridge for image conversion
            tf_broadcaster: TF broadcaster for camera TF
        """
        self.config = config
        self.logger = logger
        self.bridge = bridge or CvBridge()
        self.tf_broadcaster = tf_broadcaster

    def build_hydra_frame(
        self, frame: RsgFrame, result: Any, callback_start: float, cached: Any
    ) -> Tuple[RsgHydraFrame, Dict[str, float]]:
        """Create the combined Hydra-ready frame message."""
        hydra_msg = RsgHydraFrame()
        hydra_msg.header = frame.header
        hydra_msg.rsg_frame_id = frame.rsg_frame_id
        hydra_msg.source = frame.source
        hydra_msg.sequence = frame.sequence
        hydra_msg.rgb = frame.rgb

        depth_filter_start = time.perf_counter()
        hydra_depth, hydra_semantic, hydra_instance = self.apply_hydra_depth_range_filter(
            frame.depth_m, result.semantic_labels, result.instance_labels
        )
        hydra_depth_filter_ms = (time.perf_counter() - depth_filter_start) * 1000.0

        hydra_msg.depth_m = hydra_depth
        hydra_msg.camera_info = frame.camera_info
        hydra_msg.camera_pose = frame.camera_pose
        hydra_msg.tx = frame.tx
        hydra_msg.rot_m = frame.rot_m
        hydra_msg.semantic_labels = hydra_semantic
        hydra_msg.instance_labels = hydra_instance
        hydra_msg.label_table_json = result.label_table_json
        hydra_msg.object_metadata_json = result.object_metadata_json
        hydra_msg.unknown_candidates_json = result.unknown_candidates_json
        hydra_msg.perception_metadata_json = result.metadata_json

        metadata_start = time.perf_counter()
        metadata = {
            "phase": "phase1_hydra_input",
            "node": "rsg_object_detection",
            "classifier_success": bool(result.success),
            "classifier_status": result.status,
            "classifier_reason": result.reason,
            "num_masks": int(result.num_masks),
            "num_known": int(result.num_known),
            "num_unknown": int(result.num_unknown),
        }
        if self.config.include_frame_relation_metadata:
            metadata["source_preprocessor_metadata"] = safe_json_loads(frame.metadata_json, default={})
        hydra_msg.metadata_json = safe_json_dumps(metadata)
        hydra_metadata_build_ms = (time.perf_counter() - metadata_start) * 1000.0
        hydra_msg.coordinator_delay_ms = (time.perf_counter() - callback_start) * 1000.0
        hydra_msg.classifier_delay_ms = float(result.classifier_delay_ms)
        timing_valid = cached is not None and float(getattr(cached, "received_monotonic", 0.0) or 0.0) > 0.0
        hydra_msg.total_delay_ms = (time.perf_counter() - cached.received_monotonic) * 1000.0 if timing_valid else 0.0

        return hydra_msg, {
            "hydra_depth_filter_ms": hydra_depth_filter_ms,
            "hydra_metadata_build_ms": hydra_metadata_build_ms,
        }


    def apply_hydra_depth_range_filter(
        self, depth_msg: Any, semantic_msg: Any, instance_msg: Any
    ) -> Tuple[Any, Any, Any]:
        """Make out-of-range pixels invalid before publishing Hydra inputs.

        This is deliberately a depth gate. ID 0 in the semantic and instance
        maps denotes no usable observation but is not the reason the mesh is
        excluded; the zero depth value is.
        """
        if not self.config.hydra_depth_range_filter_enabled:
            return depth_msg, semantic_msg, instance_msg
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").astype(np.float32, copy=False)
            valid = (
                np.isfinite(depth)
                & (depth >= float(self.config.hydra_depth_min_range_m))
                & (depth <= float(self.config.hydra_depth_max_range_m))
            )
            if bool(np.all(valid)):
                return depth_msg, semantic_msg, instance_msg
            depth_filtered = np.where(valid, depth, 0.0).astype(np.float32, copy=False)
            semantic = self.bridge.imgmsg_to_cv2(semantic_msg, desired_encoding="passthrough")
            instance = self.bridge.imgmsg_to_cv2(instance_msg, desired_encoding="passthrough")
            semantic_filtered = np.where(valid, semantic, 0).astype(semantic.dtype, copy=False)
            instance_filtered = np.where(valid, instance, 0).astype(instance.dtype, copy=False)
            depth_out = self.bridge.cv2_to_imgmsg(depth_filtered, encoding="32FC1")
            semantic_out = self.bridge.cv2_to_imgmsg(semantic_filtered, encoding=self.config.semantic_label_encoding)
            instance_out = self.bridge.cv2_to_imgmsg(instance_filtered, encoding=self.config.instance_label_encoding)
            depth_out.header = depth_msg.header
            semantic_out.header = semantic_msg.header
            instance_out.header = instance_msg.header
            return depth_out, semantic_out, instance_out
        except Exception as exc:
            self.logger.warn(f"Hydra depth range filter failed; publishing original frame: {exc}")
            return depth_msg, semantic_msg, instance_msg

    def publish_hydra_camera_tf(self, hydra_msg: RsgHydraFrame) -> None:
        """Publish a synthetic camera TF only for deployments without an authoritative TF tree."""
        if self.tf_broadcaster is None:
            return
        pose = hydra_msg.camera_pose
        if not pose.header.frame_id or not hydra_msg.rgb.header.frame_id:
            return
        tf_msg = TransformStamped()
        tf_msg.header.stamp = hydra_msg.rgb.header.stamp
        tf_msg.header.frame_id = pose.header.frame_id
        tf_msg.child_frame_id = hydra_msg.rgb.header.frame_id
        tf_msg.transform.translation.x = pose.pose.position.x
        tf_msg.transform.translation.y = pose.pose.position.y
        tf_msg.transform.translation.z = pose.pose.position.z
        tf_msg.transform.rotation = pose.pose.orientation
        try:
            self.tf_broadcaster.sendTransform(tf_msg)
        except Exception as exc:
            self.logger.warn(f"Failed to publish Hydra camera TF: {exc}")

    def publish_separate_hydra_topics(self, hydra_msg: RsgHydraFrame, publishers: Dict[str, Any]) -> bool:
        """Republish the combined Hydra frame fields as separate topics."""
        self.publish_hydra_camera_tf(hydra_msg)
        ok = True
        if publishers.get("hydra_rgb_pub") is not None:
            ok = self._safe_publish(publishers["hydra_rgb_pub"], hydra_msg.rgb) and ok
        if publishers.get("hydra_depth_pub") is not None:
            ok = self._safe_publish(publishers["hydra_depth_pub"], hydra_msg.depth_m) and ok
        if publishers.get("hydra_camera_info_pub") is not None:
            ok = self._safe_publish(publishers["hydra_camera_info_pub"], hydra_msg.camera_info) and ok
        if publishers.get("hydra_pose_pub") is not None:
            ok = self._safe_publish(publishers["hydra_pose_pub"], hydra_msg.camera_pose) and ok
        if publishers.get("hydra_semantic_pub") is not None:
            ok = self._safe_publish(publishers["hydra_semantic_pub"], hydra_msg.semantic_labels) and ok
        if publishers.get("hydra_instance_pub") is not None:
            ok = self._safe_publish(publishers["hydra_instance_pub"], hydra_msg.instance_labels) and ok
        if publishers.get("hydra_metadata_pub") is not None:
            ok = self._safe_publish(publishers["hydra_metadata_pub"], String(data=hydra_msg.metadata_json)) and ok
        return ok

    def add_evidence_record(self, hydra_msg: RsgHydraFrame, result: Any, evidence_buffer: Any) -> None:
        """Store compact frame metadata for future risk-annotation retrieval."""
        if not self.config.store_evidence_frames:
            return
        objects = safe_json_loads(result.object_metadata_json, default=[])
        record = {
            "frame_id": hydra_msg.rsg_frame_id,
            "sequence": int(hydra_msg.sequence),
            "timestamp_sec": stamp_to_float(hydra_msg.header.stamp),
            "num_objects": len(objects) if isinstance(objects, list) else 0,
            "object_ids": [obj.get("candidate_id", "") for obj in objects] if isinstance(objects, list) else [],
            "total_delay_ms": float(hydra_msg.total_delay_ms),
        }
        evidence_buffer.add(record)

    def _safe_publish(self, publisher: Any, msg: Any) -> bool:
        """Safely publish message with error handling."""
        try:
            publisher.publish(msg)
            return True
        except Exception as e:
            self.logger.error(f"Publish error: {e}")
            return False
