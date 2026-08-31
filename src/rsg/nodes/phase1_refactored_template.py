"""TEMPLATE: Refactored Phase1 Coordinator (~500 lines)

This shows the pattern for applying the full refactoring.
Copy this structure and fill in with actual logic from phase1.py methods.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64MultiArray, String
from tf2_ros import TransformBroadcaster

from rsg.msg import Phase1ClassificationResult, Phase1VlmResult, RsgFrame, RsgHydraFrame

# Backend imports
from nodes.support.phase1.backends import SamMask, make_rap_backend, make_sam_backend, make_vlm_backend
from nodes.support.phase1.frame_cache import BoundedFrameCache, CachedFrame, EvidenceBuffer
from nodes.support.phase1.tracking_quality_recorder import TrackingQualityRecorder
from nodes.support.phase1.tracking_crop_manager import TrackingCropManager
from nodes.support.phase1.phase1_config import Phase1Config
from nodes.support.phase1.phase1_timing_recorder import Phase1TimingRecorder
from nodes.support.phase1.persistent_object_tracker import PersistentObjectTracker
from nodes.support.phase1.unknown_tracker import UnknownObjectTracker
from nodes.support.phase1.object_geometry import ObjectGeometryEstimator, filter_metadata
from nodes.support.phase1.label_map_builder import LabelMapBuilder
from nodes.support.phase1.rap_memory import RapMemoryUpdater
from nodes.support.phase1.time_utils import stamp_to_float

# NEW: Modular stages
from nodes.phase1_pipeline import (
    SegmentationStage,
    TrackingStage,
    SemanticsStage,
    PublishingStage,
)


class Phase1SemanticCoordinator(Node):
    """Refactored Phase1 Coordinator - Pure Orchestration (~500 lines)

    Responsibilities:
    - ROS node lifecycle management
    - Frame input and queueing
    - Worker thread orchestration
    - RAP/VLM queue management
    - Minimal state tracking

    Delegates to stages:
    - SegmentationStage: SAM processing
    - TrackingStage: Object association
    - SemanticsStage: RAP/VLM labeling
    - PublishingStage: Hydra publishing
    """

    def __init__(self) -> None:
        super().__init__("rsg_phase1_semantic_coordinator")

        # Config
        self.declare_parameter("config_file", "")
        config_file = self.get_parameter("config_file").get_parameter_value().string_value
        if not config_file:
            raise ValueError("Parameter 'config_file' must point to rsg_pipeline.yaml")

        self.config = Phase1Config.from_yaml(config_file, node_key="rsg_object_detection")
        self.set_parameters([
            rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, self.config.use_sim_time)
        ])

        # Essential infrastructure
        self.bridge = CvBridge()
        self.geometry_estimator = ObjectGeometryEstimator(self.config)
        self.label_map_builder = LabelMapBuilder(self.config)

        # Backends
        self.sam_backend = make_sam_backend(self.config, self.get_logger())
        self.rap_backend = make_rap_backend(self.config, self.get_logger())
        self.vlm_backend = make_vlm_backend(self.config)

        # Trackers
        self.persistent_tracker = PersistentObjectTracker(self.config, self.get_logger(), coordinator=self)
        self.unknown_tracker = UnknownObjectTracker(self.config, self.get_logger())

        # Utilities
        self.rap_memory_updater = RapMemoryUpdater(
            enabled=self.config.rap_update_enabled,
            output_path=self.config.rap_memory_path,
            min_confidence=self.config.rap_update_min_confidence,
            logger=self.get_logger(),
        )
        self.tf_broadcaster = TransformBroadcaster(self) if self.config.publish_hydra_tf else None

        # ===== NEW: Initialize modular stages =====
        self.seg_stage = SegmentationStage(self.sam_backend, self.config, self.get_logger())
        self.track_stage = TrackingStage(self.persistent_tracker, self.config, self.get_logger())
        self.sem_stage = SemanticsStage(self.config, self.get_logger())
        self.pub_stage = PublishingStage(self.config, self.get_logger())

        # Frame queues
        self.frame_fifo: "queue.Queue[CachedFrame]" = queue.Queue(maxsize=self.config.request_queue_size)
        self.sam_output_fifo: "queue.Queue[Tuple[Dict[str, Any], CachedFrame]]" = queue.Queue(maxsize=1)
        self.frame_cache = BoundedFrameCache(self.config.frame_cache_size)
        self.evidence_buffer = EvidenceBuffer(self.config.evidence_buffer_size)

        # Worker queues (STAY IN COORDINATOR - essential for orchestration)
        self.rap_queue: "queue.Queue[str]" = queue.Queue(maxsize=self.config.rap_queue_size)
        self.vlm_queue: "queue.Queue[Any]" = queue.Queue(maxsize=self.config.vlm_queue_size)
        self._rap_deferred_track_ids = deque()
        self._rap_deferred_track_id_set: set[str] = set()

        # Tracking state (STAY IN COORDINATOR)
        self._track_best_crops: Dict[str, Dict[str, Any]] = {}
        self._track_crop_lock = threading.RLock()
        self._semantic_label_pending_track_ids: set[str] = set()
        self._semantic_label_lock = threading.Lock()

        # Diagnostics (optional)
        self.timing_recorder = Phase1TimingRecorder(...) if self.config.diagnostics.timing_enabled else None
        self.tracking_quality_recorder = TrackingQualityRecorder(...) if self.config.diagnostics.tracking_quality_enabled else None
        self.tracking_crop_manager = TrackingCropManager(...) if self.config.diagnostics.crop_extraction_enabled else None

        # Publishers
        self.hydra_pub = self.create_publisher(RsgHydraFrame, "/rsg/hydra", ...)
        self.classifications_pub = self.create_publisher(Phase1ClassificationResult, "/rsg/classifications", ...)
        self.vlm_results_pub = self.create_publisher(Phase1VlmResult, "/rsg/vlm_results", ...)

        # Subscribers
        self.frame_sub = self.create_subscription(RsgFrame, "/rsg/frames", self.frame_callback, ...)

        # Threading
        self.seg_thread = threading.Thread(target=self._segmentation_loop, daemon=True)
        self.tracking_thread = threading.Thread(target=self._tracking_publish_loop, daemon=True)
        self.rap_thread = threading.Thread(target=self._rap_worker_loop, daemon=True)
        self.vlm_thread = threading.Thread(target=self._vlm_worker_loop, daemon=True)

        self.seg_thread.start()
        self.tracking_thread.start()
        self.rap_thread.start()
        self.vlm_thread.start()

        self._log_startup_summary()

    # ===== FRAME INPUT PIPELINE =====

    def frame_callback(self, msg: RsgFrame) -> None:
        """Callback: new RGB-D frame arrived."""
        try:
            self.frame_fifo.put_nowait(self._cache_frame(msg))
        except queue.Full:
            # Drop oldest
            try:
                self.frame_fifo.get_nowait()
                self.frame_fifo.put_nowait(self._cache_frame(msg))
            except queue.Empty:
                pass

    def _cache_frame(self, msg: RsgFrame) -> CachedFrame:
        """Convert RsgFrame to cached format."""
        rgb = self.bridge.imgmsg_to_cv2(msg.rgb_image)
        depth = self.bridge.imgmsg_to_cv2(msg.depth_image)
        return CachedFrame(frame=msg, rgb=rgb, depth=depth, timestamp_sec=stamp_to_float(msg.header.stamp))

    # ===== SEGMENTATION LOOP =====

    def _segmentation_loop(self) -> None:
        """Segmentation worker: runs SAM on frames."""
        while rclpy.ok():
            try:
                cached = self.frame_fifo.get(timeout=0.1)
            except queue.Empty:
                continue

            # ===== PHASE 1 EXTRACTION: Use SegmentationStage =====
            try:
                masks, prep_info = self.seg_stage.run(cached.rgb, cached.depth)
            except Exception as e:
                self.get_logger().error(f"SAM error: {e}")
                continue

            # Enqueue for tracking/publishing
            try:
                self.sam_output_fifo.put_nowait(({"masks": masks, "prep_info": prep_info}, cached))
            except queue.Full:
                try:
                    self.sam_output_fifo.get_nowait()
                    self.sam_output_fifo.put_nowait(({"masks": masks, "prep_info": prep_info}, cached))
                except queue.Empty:
                    pass

    # ===== TRACKING & PUBLISHING LOOP =====

    def _tracking_publish_loop(self) -> None:
        """Tracking worker: associates masks with tracks, publishes Hydra."""
        while rclpy.ok():
            try:
                stage_output, cached = self.sam_output_fifo.get(timeout=0.1)
            except queue.Empty:
                continue

            masks = stage_output.get("masks", [])
            cached_frame = cached.frame

            # ===== PHASE 2 EXTRACTION: Use TrackingStage =====
            track_records = self.track_stage.associate(
                masks, cached_frame, stamp_to_float(cached_frame.header.stamp)
            )

            # ===== PHASE 3 EXTRACTION: Use SemanticsStage for crops =====
            for record in track_records:
                track_id = record.get("persistent_track_id")
                bbox_2d = record.get("bbox_2d")
                if track_id and bbox_2d:
                    rap_crop = self.sem_stage.build_rap_crop(cached.rgb, None, bbox_2d)
                    vlm_crop = self.sem_stage.build_vlm_crop(cached.rgb, None, bbox_2d)
                    # Store for RAP/VLM workers
                    with self._track_crop_lock:
                        self._track_best_crops[track_id] = {
                            "rap": rap_crop,
                            "vlm": vlm_crop,
                        }

            # ===== PHASE 4 EXTRACTION: Use PublishingStage =====
            hydra_msg = self.pub_stage.build_hydra_frame(cached_frame, track_records)

            # Publish
            self._safe_publish(self.hydra_pub, hydra_msg)

    # ===== UTILITY METHODS =====

    def _safe_publish(self, publisher: Any, msg: Any) -> bool:
        """Safely publish message with error handling."""
        try:
            publisher.publish(msg)
            return True
        except Exception as e:
            self.get_logger().error(f"Publish error: {e}")
            return False

    def _log_startup_summary(self) -> None:
        """Log initialization summary."""
        self.get_logger().info("=== Phase1 Coordinator Started ===")
        self.get_logger().info(f"SAM: {self.config.sam_model_name}")
        self.get_logger().info(f"RAP: {'enabled' if self.config.rap_enabled else 'disabled'}")
        self.get_logger().info(f"VLM: {'enabled' if self.config.vlm_enabled else 'disabled'}")

    def destroy_node(self) -> None:
        """Cleanup on shutdown."""
        if self.timing_recorder:
            self.timing_recorder.save()
        if self.tracking_quality_recorder:
            self.tracking_quality_recorder.save_snapshots(suffix="_final")
        super().destroy_node()

    # ===== RAP/VLM WORKERS (STUB) =====

    def _rap_worker_loop(self) -> None:
        """RAP classification worker (simplified)."""
        while rclpy.ok():
            try:
                track_id = self.rap_queue.get(timeout=1.0)
                # Actual RAP processing would go here
            except queue.Empty:
                pass

    def _vlm_worker_loop(self) -> None:
        """VLM description worker (simplified)."""
        while rclpy.ok():
            try:
                track_id = self.vlm_queue.get(timeout=1.0)
                # Actual VLM processing would go here
            except queue.Empty:
                pass


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = Phase1SemanticCoordinator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
