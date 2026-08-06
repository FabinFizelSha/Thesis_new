"""Single-process Phase 1 object detection node.

This node implements Option A from the design discussion: the coordinator
and object-classifier worker run inside one ROS 2 Python process.  This avoids
the expensive ROS round trip of sending full RGB-D frames from coordinator to
classifier and then sending label maps back to the coordinator.

The node still keeps the same logical separation:

- a FIFO frame queue before SAM/RAP, used as a cushion for occasional slow
  SAM/RAP frames;
- SAM + persistent-slot association on the frame-to-Hydra path;
- a separate FIFO RAP worker that publishes ``slot_id + label`` later;
- a separate FIFO VLM queue after asynchronous RAP-unknown results;
- direct Hydra-ready output for every processed frame.
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

from nodes.support.phase1.backends import SamMask, make_rap_backend, make_sam_backend, make_vlm_backend
from nodes.support.phase1.frame_cache import BoundedFrameCache, CachedFrame, EvidenceBuffer
from nodes.support.phase1.json_utils import safe_json_dumps, safe_json_loads
from nodes.support.phase1.label_map_builder import ClassifiedMask, LabelMapBuilder
from nodes.support.phase1.object_geometry import ObjectGeometryEstimator, filter_metadata
from nodes.support.phase1.phase1_config import Phase1Config
from nodes.support.phase1.phase1_timing_recorder import Phase1TimingRecorder
from nodes.support.phase1.persistent_object_tracker import PersistentObjectTracker
from nodes.support.phase1.rap_memory import RapMemoryUpdater
from nodes.support.phase1.semantic_crop import (
    build_rap_target_only_crop,
    build_vlm_target_focus_crop,
    context_bbox_xywh,
    prepare_target_mask,
)
from nodes.support.phase1.time_utils import stamp_to_float
from nodes.support.phase1.unknown_tracker import UnknownObjectTracker
from nodes.support.phase1.vlm_result import infer_mobility_from_label


class Phase1SemanticCoordinator(Node):
    """Coordinate Phase 1 segmentation, tracking, RAP retrieval, and VLM fallback.

    The class is intentionally a single ROS 2 node. SAM, RAP, and VLM workers
    remain internal threads so RGB-D crops do not cross ROS process boundaries.
    Final slot-to-label events are published only after RAP or VLM reaches a
    terminal decision.
    """


    def __init__(self) -> None:
        super().__init__("rsg_phase1_semantic_coordinator")

        self.declare_parameter("config_file", "")
        config_file = self.get_parameter("config_file").get_parameter_value().string_value
        if not config_file:
            raise ValueError("Parameter 'config_file' must point to rsg_pipeline.yaml")

        self.config = Phase1Config.from_yaml(config_file, node_key="rsg_object_detection")  # Baseline YAML compatibility key.
        self.set_parameters([
            rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, self.config.use_sim_time)
        ])

        self.bridge = CvBridge()
        self.sam_backend = make_sam_backend(self.config, self.get_logger())
        self.rap_backend = make_rap_backend(self.config, self.get_logger())
        self.vlm_backend = make_vlm_backend(self.config)
        self.rap_memory_updater = RapMemoryUpdater(
            enabled=self.config.rap_update_enabled,
            output_path=self.config.rap_memory_path,
            min_confidence=self.config.rap_update_min_confidence,
            logger=self.get_logger(),
        )
        self.geometry_estimator = ObjectGeometryEstimator(self.config)
        self.label_map_builder = LabelMapBuilder(self.config)
        self.unknown_tracker = UnknownObjectTracker(self.config, self.get_logger())
        self.persistent_tracker = PersistentObjectTracker(self.config, self.get_logger())
        self.tf_broadcaster = TransformBroadcaster(self) if self.config.publish_hydra_tf else None

        # Hydra receives one fixed slot ID per physical object.  Semantic names
        # are applied by the downstream scene-graph fuser, therefore Phase 1 does not rewrite or
        # reserve Hydra label-space entries across sessions.
        self.static_hydra_label_ids = dict(self.config.hydra_label_lookup)
        self.static_hydra_label_names = dict(self.config.hydra_label_names)
        self.persistent_tracker.set_reserved_slot_ids(set())
        self.semantic_reuse_enabled = False
        # RAP/VLM scheduling is intentionally decoupled from Hydra output.  A
        # persistent track receives a unique Hydra slot immediately, while the
        # workers receive only its track ID.  The current best crop remains
        # mutable until the relevant worker dequeues that ID; an immutable crop
        # snapshot is then used for that worker's single inference request.
        self.rap_runs_async = bool(self.config.rap_enabled)
        self.rap_runs_synchronously = False
        self._track_best_crops: Dict[str, Dict[str, Any]] = {}
        self._track_crop_lock = threading.RLock()
        self._semantic_label_pending_track_ids: set[str] = set()
        self._semantic_label_lock = threading.Lock()
        self._latest_processed_timestamp_sec = 0.0

        # One application-level frame FIFO before SAM. A second one-slot FIFO
        # sits between SAM and tracking/publish so the two run as a pipeline:
        # the segmentation thread can start the next frame's SAM inference
        # while the tracking/publish thread is still finishing geometry,
        # slot assignment, and the Hydra publish for the previous frame. Both
        # queues keep the same drop-oldest bias so the pipeline always
        # prefers the newest available observation over completeness.
        self.frame_fifo: "queue.Queue[CachedFrame]" = queue.Queue(maxsize=self.config.request_queue_size)
        self.sam_output_fifo: "queue.Queue[Tuple[Dict[str, Any], CachedFrame]]" = queue.Queue(maxsize=1)
        self.sam_output_dropped_count = 0
        self.frame_cache = BoundedFrameCache(self.config.frame_cache_size)
        self.evidence_buffer = EvidenceBuffer(self.config.evidence_buffer_size)

        # The bounded worker FIFOs store only persistent track IDs.  When a
        # worker FIFO is full, the ID is retained in a small deferred registry
        # rather than being dropped.  Crops are never held in either queue.
        self.rap_queue: "queue.Queue[str]" = queue.Queue(maxsize=self.config.rap_queue_size)
        self.rap_queue_dropped_count = 0
        self.rap_queue_deferred_count = 0
        self.rap_completed_count = 0
        self._rap_task_keys: set[str] = set()
        self._rap_deferred_track_ids = deque()
        self._rap_deferred_track_id_set: set[str] = set()
        self._rap_enqueued_monotonic: Dict[str, float] = {}
        self._rap_task_lock = threading.Lock()

        # VLM uses the same ID-only/deferred scheduling policy.  A legacy
        # dictionary task remains supported only for the RAP-disabled fallback
        # path; normal RAP-enabled operation always queues a track ID.
        self.vlm_queue: "queue.Queue[Any]" = queue.Queue(maxsize=self.config.vlm_queue_size)
        self.vlm_queue_dropped_count = 0
        self.vlm_queue_deferred_count = 0
        self._vlm_task_keys: set[str] = set()
        self._vlm_deferred_track_ids = deque()
        self._vlm_deferred_track_id_set: set[str] = set()
        self._vlm_enqueued_monotonic: Dict[str, float] = {}
        self._vlm_task_lock = threading.Lock()
        # Tracks that reached RAP but still have a weak VLM crop are retained
        # outside the VLM FIFO while later observations may improve the crop.
        # The defer-start timestamp uses recorded bag time. A bounded timeout
        # prevents small or partly visible real objects from waiting forever.
        self._vlm_quality_deferred_track_ids: set[str] = set()
        self._vlm_quality_deferred_since_timestamp_sec: Dict[str, float] = {}
        self._vlm_quality_force_track_ids: set[str] = set()
        self._vlm_quality_deferred_lock = threading.Lock()

        self._stop_event = threading.Event()
        # Segmentation (GPU-bound SAM) and tracking/publish (CPU-bound) run on
        # separate threads so they can overlap: SAM backends release the GIL
        # for most of their wall time while waiting on the GPU, the same
        # property the RAP/VLM worker threads below already rely on.
        self._segmentation_thread = threading.Thread(target=self._segmentation_loop, daemon=True)
        self._tracking_publish_thread = threading.Thread(target=self._tracking_publish_loop, daemon=True)
        self._rap_thread = threading.Thread(target=self._rap_loop, daemon=True)
        self._vlm_thread = threading.Thread(target=self._vlm_loop, daemon=True)

        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self.config.input_qos_depth,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self.config.output_qos_depth,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        # Final slot-to-label events are compact and must not be lost while the
        # fuser coalesces expensive graph redraws. Keep a separate deep queue
        # instead of increasing image-topic buffering.
        semantic_label_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self.config.semantic_label_qos_depth,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.frame_sub = self.create_subscription(RsgFrame, self.config.preprocessed_frame_topic, self.frame_callback, input_qos)

        self.hydra_frame_pub = self.create_publisher(RsgHydraFrame, self.config.hydra_frame_topic, output_qos)
        self.vlm_result_pub = self.create_publisher(Phase1VlmResult, self.config.vlm_result_topic, output_qos)
        # JSON payload: persistent_track_id, hydra_slot_id, label, confidence,
        # is_known, timestamp/frame provenance. This is the RAP/RSG boundary.
        self.rap_result_pub = self.create_publisher(String, self.config.rap_result_topic, output_qos)
        # One asynchronous semantic result keyed by a persistent Hydra slot.
        # The RAP fuser joins this label to Hydra object nodes.
        self.semantic_label_result_pub = self.create_publisher(
            String,
            self.config.semantic_labeling_publish_topic,
            semantic_label_qos,
        )
        self.active_segments_pub = self.create_publisher(
            String,
            self.config.persistent_active_segments_topic,
            output_qos,
        )
        self.status_pub = self.create_publisher(String, self.config.status_topic, output_qos)
        self.unknown_pub = self.create_publisher(String, self.config.unknown_candidates_topic, output_qos)
        self.timing_pub = None
        if self.config.timing_enabled and self.config.publish_timing_topic:
            self.timing_pub = self.create_publisher(Float64MultiArray, self.config.timing_topic, output_qos)

        self.hydra_rgb_pub = None
        self.hydra_depth_pub = None
        self.hydra_camera_info_pub = None
        self.hydra_pose_pub = None
        self.hydra_semantic_pub = None
        self.hydra_instance_pub = None
        self.hydra_metadata_pub = None
        if self.config.publish_hydra_separate_topics:
            self.hydra_rgb_pub = self.create_publisher(Image, self.config.hydra_rgb_topic, output_qos)
            self.hydra_depth_pub = self.create_publisher(Image, self.config.hydra_depth_topic, output_qos)
            self.hydra_camera_info_pub = self.create_publisher(CameraInfo, self.config.hydra_camera_info_topic, output_qos)
            self.hydra_pose_pub = self.create_publisher(PoseStamped, self.config.hydra_pose_topic, output_qos)
            self.hydra_semantic_pub = self.create_publisher(Image, self.config.hydra_semantic_topic, output_qos)
            self.hydra_instance_pub = self.create_publisher(Image, self.config.hydra_instance_topic, output_qos)
            self.hydra_metadata_pub = self.create_publisher(String, self.config.hydra_metadata_topic, output_qos)

        self.timing_recorder = Phase1TimingRecorder(
            enabled=self.config.timing_enabled and self.config.write_timing_csv,
            output_path=self.config.timing_csv_path,
            autosave_every=self.config.timing_excel_autosave_every,
            logger=self.get_logger(),
            sheet_name=self.config.timing_sheet_name,
        )

        self.received_count = 0
        self.processed_count = 0
        self.failed_count = 0
        self.dropped_count = 0
        self.hydra_published_count = 0
        self.unknown_vlm_count = 0

        self._segmentation_thread.start()
        self._tracking_publish_thread.start()
        if self.rap_runs_async:
            self._rap_thread.start()
        if self.config.vlm_enabled:
            self._vlm_thread.start()
        self._log_startup_summary()

    def _log_startup_summary(self) -> None:
        self.get_logger().info("rsg_phase1_semantic_coordinator started in single-process Option-A mode.")
        self.get_logger().info(f"Input frame topic: {self.config.preprocessed_frame_topic}")
        self.get_logger().info(f"Hydra combined output topic: {self.config.hydra_frame_topic}")
        if self.config.publish_hydra_separate_topics:
            self.get_logger().info("Hydra direct separated topics enabled:")
            self.get_logger().info(f"  RGB: {self.config.hydra_rgb_topic}")
            self.get_logger().info(f"  Depth: {self.config.hydra_depth_topic}")
            self.get_logger().info(f"  CameraInfo: {self.config.hydra_camera_info_topic}")
            self.get_logger().info(f"  Semantic labels: {self.config.hydra_semantic_topic}")
            self.get_logger().info(f"  Instance labels: {self.config.hydra_instance_topic}")
            self.get_logger().info(f"  Pose: {self.config.hydra_pose_topic}")
        else:
            self.get_logger().info("Hydra direct separated topics disabled.")
        self.get_logger().info(f"RAP result topic: {self.config.rap_result_topic}")
        self.get_logger().info(f"VLM result topic: {self.config.vlm_result_topic}")
        self.get_logger().info(
            f"Profile={self.config.profile}, allow_dummy_fallback={self.config.allow_dummy_fallback}"
        )
        self.get_logger().info(
            f"Frame FIFO size={self.config.request_queue_size}, frame_cache_size={self.config.frame_cache_size}, "
            f"RAP FIFO size={self.config.rap_queue_size}, VLM FIFO size={self.config.vlm_queue_size}"
        )
        self.get_logger().info(
            f"SAM backend={self.config.sam_backend}, RAP backend={self.config.rap_backend}, "
            f"RAP execution={'async' if self.rap_runs_async else 'synchronous'}, "
            f"slot ontology=physical-instance-only, VLM enabled={self.config.vlm_enabled}, "
            f"VLM mode={self.config.vlm_mode}, profile={self.config.vlm_active_profile or 'default'}, "
            f"model={self.config.vlm_model}, endpoint={self.config.vlm_endpoint}"
        )
        self.get_logger().info(
            f"SAM input scale={self.config.sam_input_scale_ratio:.3f}, "
            f"depth_filter_enabled={self.config.sam_depth_filter_enabled}, "
            f"depth_range=[{self.config.sam_depth_filter_min_m:.2f}, {self.config.sam_depth_filter_max_m:.2f}] m, "
            f"crop_to_valid_roi={self.config.sam_depth_filter_crop_to_roi}"
        )
        self.get_logger().info(
            "Persistent object tracking: "
            f"enabled={self.config.persistent_tracking_enabled}, "
            f"mode={'hydra_slots' if self.config.persistent_use_hydra_slots else 'instance_only'}, "
            f"max_tracks={self.config.persistent_max_tracks}"
        )
        if self.config.persistent_tracking_enabled and self.config.persistent_use_hydra_slots:
            first_slot = self.config.persistent_slot_first_label_id
            last_slot = first_slot + self.config.persistent_slot_count - 1
            self.get_logger().info(
                f"Hydra slot range=[{first_slot}, {last_slot}], "
                "RAP scheduling=immediate_async_once_per_track"
            )

    def frame_callback(self, msg: RsgFrame) -> None:
        """Receive preprocessed frames and enqueue them for SAM/RAP.

        This callback does not run SAM/RAP. It only stores the frame in the
        bounded FIFO so the ROS subscription callback remains lightweight.

        Important lifecycle detail:
        the FIFO now carries the ``CachedFrame`` timing object itself. Earlier
        versions placed only the ROS message in the FIFO and looked up timing
        later from a small bounded cache. With real SAM/RAP/VLM, that cache can
        evict the frame before Hydra publishing, producing misleading
        ``total_delay_ms = 0`` samples. Keeping timing with the queued frame
        makes latency reporting independent of cache eviction.
        """
        # Shutdown has started; do not enqueue another frame.
        if self._stop_event.is_set() or not rclpy.ok():
            return

        now = time.perf_counter()
        self.received_count += 1
        frame_id = msg.rsg_frame_id
        rgb_time = stamp_to_float(msg.header.stamp)
        cached = CachedFrame(
            frame_id=frame_id,
            sequence=int(msg.sequence),
            received_monotonic=now,
            received_stamp_sec=rgb_time,
            msg=msg,
            status="received",
        )
        self.frame_cache.put(cached)
        self.record_frame_lifecycle_event(
            "received",
            cached,
            status="received",
            timing_valid=True,
            timing_source="fifo_cached_frame",
            total_delay_ms=0.0,
            reason="preprocessed_frame_received",
        )

        if self.frame_fifo.full():
            if self.config.drop_oldest_when_full:
                try:
                    dropped_cached = self.frame_fifo.get_nowait()
                    dropped_cached.status = "dropped_oldest"
                    self.dropped_count += 1
                    self.record_frame_lifecycle_event(
                        "dropped_oldest",
                        dropped_cached,
                        status="dropped",
                        timing_valid=True,
                        timing_source="fifo_cached_frame",
                        total_delay_ms=(time.perf_counter() - dropped_cached.received_monotonic) * 1000.0,
                        reason="frame_fifo_full",
                    )
                    self.frame_cache.remove(dropped_cached.frame_id)
                except queue.Empty:
                    pass
            else:
                cached.status = "dropped_newest"
                self.dropped_count += 1
                self.record_frame_lifecycle_event(
                    "dropped_newest",
                    cached,
                    status="dropped",
                    timing_valid=True,
                    timing_source="fifo_cached_frame",
                    total_delay_ms=(time.perf_counter() - cached.received_monotonic) * 1000.0,
                    reason="frame_fifo_full_drop_newest",
                )
                self.frame_cache.remove(frame_id)
                self.publish_status("dropped", frame_id, "frame_fifo_full_drop_newest")
                return

        try:
            cached.status = "enqueued"
            self.frame_fifo.put_nowait(cached)
            cached.enqueued_monotonic = time.perf_counter()
            cached.callback_enqueue_delay_ms = (
                cached.enqueued_monotonic - cached.received_monotonic
            ) * 1000.0
            self.record_frame_lifecycle_event(
                "enqueued",
                cached,
                status="queued",
                timing_valid=True,
                timing_source="fifo_cached_frame",
                total_delay_ms=(time.perf_counter() - cached.received_monotonic) * 1000.0,
                reason="preprocessed_frame_received",
            )
        except queue.Full:
            cached.status = "dropped_newest"
            self.dropped_count += 1
            self.record_frame_lifecycle_event(
                "dropped_newest",
                cached,
                status="dropped",
                timing_valid=True,
                timing_source="fifo_cached_frame",
                total_delay_ms=(time.perf_counter() - cached.received_monotonic) * 1000.0,
                reason="frame_fifo_full_race",
            )
            self.frame_cache.remove(frame_id)
            self.publish_status("dropped", frame_id, "frame_fifo_full")
            return

        if self.received_count % self.config.status_every_n_frames == 0:
            self.publish_status("queued", frame_id, "ok")


    def _safe_publish(self, publisher: Any, msg: Any) -> bool:
        """Publish without crashing during Ctrl+C / ROS context shutdown.

        Background worker threads may finish a frame after launch has already
        started shutting down the ROS context. Direct publisher.publish() then
        raises RCLError("publisher's context is invalid"). For normal runtime
        this returns True; during shutdown it returns False silently so the
        node can close debug files cleanly.
        """
        if publisher is None:
            return False
        try:
            if self._stop_event.is_set() or not rclpy.ok():
                return False
        except Exception:
            return False
        try:
            publisher.publish(msg)
            return True
        except Exception:
            return False

    def _segmentation_loop(self) -> None:
        """Consume the pre-SAM FIFO and run image conversion + SAM only.

        This runs on its own thread from ``_tracking_publish_loop``. SAM
        backends spend most of their wall time blocked on the GPU, which
        releases the interpreter's GIL, so this thread's next-frame SAM call
        can genuinely overlap with the other thread's CPU-bound geometry,
        tracking, and Hydra-publish work for the frame ahead of it -- the
        same overlap the RAP/VLM worker threads below already exploit.
        """
        while not self._stop_event.is_set():
            try:
                cached = self.frame_fifo.get(timeout=0.1)
            except queue.Empty:
                continue

            dequeue_time = time.perf_counter()
            frame = cached.msg
            cached.sent_to_classifier_monotonic = dequeue_time
            cached.sent_to_classifier_delay_ms = (dequeue_time - cached.received_monotonic) * 1000.0
            cached.frame_queue_wait_ms = (
                dequeue_time - (cached.enqueued_monotonic or cached.received_monotonic)
            ) * 1000.0
            cached.status = "dequeued_to_sam"
            fifo_wait_ms = cached.sent_to_classifier_delay_ms
            # Refresh lookup cache as a convenience for external/debug code, but
            # the queued CachedFrame is now the authoritative timing source.
            self.frame_cache.put(cached)
            self.record_frame_lifecycle_event(
                "dequeued_to_sam",
                cached,
                status="processing",
                timing_valid=True,
                timing_source="fifo_cached_frame",
                queue_wait_ms=fifo_wait_ms,
                total_delay_ms=(time.perf_counter() - cached.received_monotonic) * 1000.0,
                reason="worker_ready",
            )

            try:
                stage = self.run_segmentation_stage(frame, input_age_ms=fifo_wait_ms)
            except Exception as exc:
                self.failed_count += 1
                if rclpy.ok() and not self._stop_event.is_set():
                    self.get_logger().error(f"SAM stage failed for frame {frame.rsg_frame_id}: {exc}")
                self.record_frame_lifecycle_event(
                    "failed",
                    cached,
                    status="failed",
                    timing_valid=True,
                    timing_source="fifo_cached_frame",
                    total_delay_ms=(time.perf_counter() - cached.received_monotonic) * 1000.0,
                    reason=str(exc),
                )
                self.publish_status("failed", frame.rsg_frame_id, str(exc))
                continue

            self._enqueue_sam_output(stage, cached)

    def _enqueue_sam_output(self, stage: Dict[str, Any], cached: CachedFrame) -> None:
        """Hand a completed SAM stage to the tracking/publish thread.

        Mirrors ``frame_fifo``'s drop-oldest bias: a slow tracking/publish
        stage should not make Hydra fall further and further behind real
        time, so a not-yet-consumed handoff is replaced by a newer one
        rather than queued behind it.
        """
        if self.sam_output_fifo.full():
            if not self.config.drop_oldest_when_full:
                self.sam_output_dropped_count += 1
                return
            try:
                self.sam_output_fifo.get_nowait()
                self.sam_output_dropped_count += 1
            except queue.Empty:
                pass
        try:
            self.sam_output_fifo.put_nowait((stage, cached))
        except queue.Full:
            self.sam_output_dropped_count += 1

    def _tracking_publish_loop(self) -> None:
        """Consume completed SAM stages; run tracking, label maps, and Hydra publish.

        Runs on its own thread so a slow geometry/tracking pass or the ROS
        publish call never blocks ``_segmentation_loop`` from starting the
        next frame's SAM inference.
        """
        while not self._stop_event.is_set():
            try:
                stage, cached = self.sam_output_fifo.get(timeout=0.1)
            except queue.Empty:
                continue

            dequeue_time = time.perf_counter()
            sam_output_queue_wait_ms = max(
                0.0, (dequeue_time - stage["stage_a_complete_monotonic"]) * 1000.0
            )
            frame = stage["frame"]
            cached.status = "dequeued_to_tracking"
            self.frame_cache.put(cached)

            try:
                result = self.run_tracking_publish_stage(
                    stage, sam_output_queue_wait_ms=sam_output_queue_wait_ms
                )

                # Per-frame diagnostics are written once, after Hydra publish,
                # so measurement does not take the recorder lock repeatedly on
                # the hot path. The optional live topic remains available.
                timing_start = time.perf_counter()
                if self.timing_pub is not None:
                    self.publish_timing_event(result)
                result.classifier_debug_record_delay_ms = (time.perf_counter() - timing_start) * 1000.0

                self._publish_hydra_from_result(frame, result, cached)
                self.processed_count += 1
                self.publish_status("processed", frame.rsg_frame_id, "ok")
            except Exception as exc:
                self.failed_count += 1
                if rclpy.ok() and not self._stop_event.is_set():
                    self.get_logger().error(f"Failed to process frame {frame.rsg_frame_id}: {exc}")
                self.record_frame_lifecycle_event(
                    "failed",
                    cached,
                    status="failed",
                    timing_valid=True,
                    timing_source="fifo_cached_frame",
                    total_delay_ms=(time.perf_counter() - cached.received_monotonic) * 1000.0,
                    reason=str(exc),
                )
                self.publish_status("failed", frame.rsg_frame_id, str(exc))

    def _publish_hydra_from_result(self, frame: RsgFrame, result: Phase1ClassificationResult, cached: Optional[CachedFrame]) -> None:
        """Build and publish Hydra-ready output in the same process.

        The method now measures sub-phases explicitly so that a future
        ``pipeline_wait_ms`` spike can be traced to a named phase instead of
        remaining unexplained.
        """
        callback_start = time.perf_counter()

        build_start = time.perf_counter()
        hydra_msg, hydra_stage_ms = self.build_hydra_frame(frame, result, build_start, cached)
        hydra_build_delay_ms = (time.perf_counter() - build_start) * 1000.0

        publish_start = time.perf_counter()
        publish_success = True
        if self.config.publish_hydra_combined:
            publish_success = self._safe_publish(self.hydra_frame_pub, hydra_msg) and publish_success
        if self.config.publish_hydra_separate_topics:
            publish_success = self.publish_separate_hydra_topics(hydra_msg) and publish_success
        hydra_publish_delay_ms = (time.perf_counter() - publish_start) * 1000.0

        unknown_publish_start = time.perf_counter()
        unknowns = safe_json_loads(result.unknown_candidates_json, default=[])
        if unknowns and self.config.include_unknown_objects:
            self._safe_publish(self.unknown_pub, String(data=result.unknown_candidates_json))
        unknown_publish_delay_ms = (time.perf_counter() - unknown_publish_start) * 1000.0

        # Hydra latency ends here: the Hydra-ready output has been published.
        hydra_publish_complete = time.perf_counter()
        if publish_success:
            self.hydra_published_count += 1
        coordinator_delay_ms = (hydra_publish_complete - callback_start) * 1000.0
        hydra_status = "sent_to_hydra" if publish_success else "hydra_publish_skipped"
        timing_valid = cached is not None and float(getattr(cached, "received_monotonic", 0.0) or 0.0) > 0.0
        timing_source = "fifo_cached_frame" if timing_valid else "missing_timing_context"
        total_delay_ms = (hydra_publish_complete - cached.received_monotonic) * 1000.0 if timing_valid else 0.0
        sent_to_classifier_delay_ms = float(cached.sent_to_classifier_delay_ms) if timing_valid else 0.0
        classifier_debug_ms = float(getattr(result, "classifier_debug_record_delay_ms", 0.0))

        metadata = safe_json_loads(result.metadata_json, default={})
        stage_ms = metadata.get("diagnostic_stage_ms", {}) or {}
        sam_output_queue_wait_ms = float(stage_ms.get("sam_output_queue_wait_ms", 0.0))

        pipeline_wait_ms = max(
            0.0,
            total_delay_ms
            - sent_to_classifier_delay_ms
            - sam_output_queue_wait_ms
            - float(result.classifier_delay_ms)
            - classifier_debug_ms
            - coordinator_delay_ms,
        )
        classifier_known_ms = (
            float(result.image_conversion_delay_ms)
            + float(result.sam_delay_ms)
            + float(result.rap_delay_ms)
            + float(result.label_map_delay_ms)
            + float(result.metadata_delay_ms)
            + float(result.result_message_build_delay_ms)
        )
        classifier_other_ms = max(0.0, float(result.classifier_delay_ms) - classifier_known_ms)
        hydra_build_other_ms = max(
            0.0,
            hydra_build_delay_ms
            - float(hydra_stage_ms.get("hydra_depth_filter_ms", 0.0))
            - float(hydra_stage_ms.get("hydra_metadata_build_ms", 0.0)),
        )
        coordinator_other_ms = max(
            0.0,
            coordinator_delay_ms - hydra_build_delay_ms - hydra_publish_delay_ms - unknown_publish_delay_ms,
        )

        if self.config.timing_enabled:
            self.timing_recorder.add_sample(
                node="rsg_object_detection",
                event="frame_trace",
                sequence=int(result.sequence),
                frame_id=result.rsg_frame_id,
                status=hydra_status,
                reason="ok" if publish_success else "ros_context_shutdown_or_publish_failed",
                coordinator_delay_ms=coordinator_delay_ms,
                classifier_delay_ms=float(result.classifier_delay_ms),
                total_delay_ms=total_delay_ms,
                sent_to_classifier_delay_ms=sent_to_classifier_delay_ms,
                callback_enqueue_delay_ms=float(cached.callback_enqueue_delay_ms) if timing_valid else 0.0,
                frame_queue_wait_ms=float(cached.frame_queue_wait_ms) if timing_valid else 0.0,
                sam_output_queue_wait_ms=sam_output_queue_wait_ms,
                classifier_debug_record_delay_ms=classifier_debug_ms,
                image_conversion_delay_ms=float(result.image_conversion_delay_ms),
                sam_prepare_ms=float(stage_ms.get("sam_prepare_ms", 0.0)),
                sam_inference_ms=float(stage_ms.get("sam_inference_ms", 0.0)),
                sam_restore_ms=float(stage_ms.get("sam_restore_ms", 0.0)),
                sam_other_ms=float(stage_ms.get("sam_other_ms", 0.0)),
                sam_delay_ms=float(result.sam_delay_ms),
                geometry_metadata_ms=float(stage_ms.get("geometry_metadata_ms", 0.0)),
                geometry_mask_extract_ms=float(stage_ms.get("geometry_mask_extract_ms", 0.0)),
                geometry_depth_gather_ms=float(stage_ms.get("geometry_depth_gather_ms", 0.0)),
                geometry_projection_ms=float(stage_ms.get("geometry_projection_ms", 0.0)),
                geometry_stats_ms=float(stage_ms.get("geometry_stats_ms", 0.0)),
                frame_assignment_ms=float(stage_ms.get("frame_assignment_ms", 0.0)),
                assignment_candidate_search_ms=float(stage_ms.get("assignment_candidate_search_ms", 0.0)),
                assignment_row_init_ms=float(stage_ms.get("assignment_row_init_ms", 0.0)),
                assignment_3d_geometry_ms=float(stage_ms.get("assignment_3d_geometry_ms", 0.0)),
                assignment_centroid_iou_ms=float(stage_ms.get("assignment_centroid_iou_ms", 0.0)),
                assignment_scoring_ms=float(stage_ms.get("assignment_scoring_ms", 0.0)),
                assignment_a2_redundancy_ms=float(stage_ms.get("assignment_a2_redundancy_ms", 0.0)),
                assignment_a3_nested_ms=float(stage_ms.get("assignment_a3_nested_ms", 0.0)),
                assignment_hungarian_ms=float(stage_ms.get("assignment_hungarian_ms", 0.0)),
                assignment_candidate_count_total=float(stage_ms.get("assignment_candidate_count_total", 0.0)),
                assignment_candidate_count_max=float(stage_ms.get("assignment_candidate_count_max", 0.0)),
                assignment_lock_wait_ms=float(stage_ms.get("assignment_lock_wait_ms", 0.0)),
                association_lock_wait_ms=float(stage_ms.get("association_lock_wait_ms", 0.0)),
                track_association_ms=float(stage_ms.get("track_association_ms", 0.0)),
                crop_update_ms=float(stage_ms.get("crop_update_ms", 0.0)),
                run_rap_other_ms=float(stage_ms.get("run_rap_other_ms", 0.0)),
                active_segments_publish_ms=float(stage_ms.get("active_segments_publish_ms", 0.0)),
                semantic_dispatch_ms=float(stage_ms.get("semantic_dispatch_ms", 0.0)),
                quality_deferred_release_ms=float(stage_ms.get("quality_deferred_release_ms", 0.0)),
                rap_delay_ms=float(result.rap_delay_ms),
                label_map_delay_ms=float(result.label_map_delay_ms),
                metadata_delay_ms=float(result.metadata_delay_ms),
                result_message_build_delay_ms=float(result.result_message_build_delay_ms),
                classifier_other_ms=classifier_other_ms,
                hydra_build_delay_ms=hydra_build_delay_ms,
                hydra_depth_filter_ms=float(hydra_stage_ms.get("hydra_depth_filter_ms", 0.0)),
                hydra_metadata_build_ms=float(hydra_stage_ms.get("hydra_metadata_build_ms", 0.0)),
                hydra_build_other_ms=hydra_build_other_ms,
                hydra_publish_delay_ms=hydra_publish_delay_ms,
                unknown_publish_delay_ms=unknown_publish_delay_ms,
                coordinator_other_ms=coordinator_other_ms,
                pipeline_wait_ms=pipeline_wait_ms,
                num_masks=int(result.num_masks),
                num_known=int(result.num_known),
                num_unknown=int(result.num_unknown),
                num_unknown_tracks=int(metadata.get("num_unknown_tracks", 0) or 0),
                num_vlm_queued=int(metadata.get("num_vlm_queued", 0) or 0),
            )

        evidence_start = time.perf_counter()
        self.add_evidence_record(hydra_msg, result)
        evidence_record_delay_ms = (time.perf_counter() - evidence_start) * 1000.0
        # Evidence is post-Hydra-publish work. It is not added to
        # total_delay_ms, but measuring it prevents confusion if it later causes
        # FIFO wait on following frames.
        if self.config.timing_enabled and evidence_record_delay_ms > 1.0:
            self.get_logger().debug(
                f"Post-Hydra evidence recording took {evidence_record_delay_ms:.3f} ms for {frame.rsg_frame_id}"
            )

        # Keep the cache small: after Hydra output is built, this frame is no
        # longer needed for result matching in the combined mode.
        self.frame_cache.remove(frame.rsg_frame_id)

    def build_hydra_frame(self, frame: RsgFrame, result: Phase1ClassificationResult, callback_start: float, cached: Optional[CachedFrame]) -> Tuple[RsgHydraFrame, Dict[str, float]]:
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
        self, depth_msg: Image, semantic_msg: Image, instance_msg: Image
    ) -> Tuple[Image, Image, Image]:
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
            self.get_logger().warn(f"Hydra depth range filter failed; publishing original frame: {exc}")
            return depth_msg, semantic_msg, instance_msg


    def publish_hydra_camera_tf(self, hydra_msg: RsgHydraFrame) -> None:
        """Publish a synthetic camera TF only for deployments without an authoritative TF tree.

        The calibrated Tesse configuration disables this path. Hydra instead
        resolves world -> base_link_gt and base_link_gt -> left_cam directly
        from the rosbag TF stream at each image timestamp.
        """
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
            self.get_logger().warn(f"Failed to publish Hydra camera TF: {exc}")

    def publish_separate_hydra_topics(self, hydra_msg: RsgHydraFrame) -> bool:
        """Republish the combined Hydra frame fields as separate topics."""
        self.publish_hydra_camera_tf(hydra_msg)
        ok = True
        if self.hydra_rgb_pub is not None:
            ok = self._safe_publish(self.hydra_rgb_pub, hydra_msg.rgb) and ok
        if self.hydra_depth_pub is not None:
            ok = self._safe_publish(self.hydra_depth_pub, hydra_msg.depth_m) and ok
        if self.hydra_camera_info_pub is not None:
            ok = self._safe_publish(self.hydra_camera_info_pub, hydra_msg.camera_info) and ok
        if self.hydra_pose_pub is not None:
            ok = self._safe_publish(self.hydra_pose_pub, hydra_msg.camera_pose) and ok
        if self.hydra_semantic_pub is not None:
            ok = self._safe_publish(self.hydra_semantic_pub, hydra_msg.semantic_labels) and ok
        if self.hydra_instance_pub is not None:
            ok = self._safe_publish(self.hydra_instance_pub, hydra_msg.instance_labels) and ok
        if self.hydra_metadata_pub is not None:
            ok = self._safe_publish(self.hydra_metadata_pub, String(data=hydra_msg.metadata_json)) and ok
        return ok

    def add_evidence_record(self, hydra_msg: RsgHydraFrame, result: Phase1ClassificationResult) -> None:
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
        self.evidence_buffer.add(record)

    def record_frame_lifecycle_event(
        self,
        event: str,
        cached: Optional[CachedFrame],
        *,
        frame: Optional[RsgFrame] = None,
        status: str = "",
        timing_valid: bool = False,
        timing_source: str = "",
        queue_wait_ms: float = 0.0,
        total_delay_ms: float = 0.0,
        reason: str = "",
    ) -> None:
        """Record one frame lifecycle latency event."""
        if not self.config.timing_enabled or event not in {"failed", "dropped_oldest", "dropped_newest"}:
            return
        msg = frame if frame is not None else cached.msg if cached is not None else None
        if msg is None:
            return
        self.timing_recorder.add_sample(
            node="rsg_object_detection",
            event=event,
            sequence=int(msg.sequence),
            frame_id=msg.rsg_frame_id,
            status=status,
            timing_valid=bool(timing_valid),
            timing_source=timing_source,
            received_count=int(self.received_count),
            processed_count=int(self.processed_count),
            failed_count=int(self.failed_count),
            dropped_count=int(self.dropped_count),
            hydra_published_count=int(self.hydra_published_count),
            frame_fifo_size=int(self.frame_fifo.qsize()),
            frame_fifo_max_size=int(self.config.request_queue_size),
            queue_wait_ms=float(queue_wait_ms),
            total_delay_ms=float(total_delay_ms),
            reason=reason,
        )

    def run_segmentation_stage(self, frame: RsgFrame, input_age_ms: float = 0.0) -> Dict[str, Any]:
        """Run image conversion and SAM only. Executes on the segmentation thread.

        Everything downstream (geometry, tracking, label maps, Hydra publish)
        runs later on a separate thread via ``run_tracking_publish_stage``, so
        this stage's own timing is recorded here and carried through
        unchanged rather than folded into one combined measurement.
        """
        start = time.perf_counter()
        input_age_ms = float(input_age_ms)

        conversion_start = time.perf_counter()
        rgb = self.bridge.imgmsg_to_cv2(frame.rgb, desired_encoding="rgb8")
        depth = self.bridge.imgmsg_to_cv2(frame.depth_m, desired_encoding="32FC1")
        tx = np.array(frame.tx, dtype=np.float64)
        rot_m = np.array(frame.rot_m, dtype=np.float64).reshape(3, 3)
        image_conversion_delay_ms = (time.perf_counter() - conversion_start) * 1000.0

        sam_start = time.perf_counter()
        sam_prepare_start = sam_start
        sam_rgb, sam_prep = self.prepare_sam_input(rgb, depth)
        sam_prepare_delay_ms = (time.perf_counter() - sam_prepare_start) * 1000.0
        sam_inference_start = time.perf_counter()
        sam_masks_small = self.run_sam(sam_rgb, min_area_px=self.scaled_min_mask_pixels(sam_prep))
        sam_inference_delay_ms = (time.perf_counter() - sam_inference_start) * 1000.0
        sam_restore_start = time.perf_counter()
        sam_masks = self.restore_sam_masks_to_original(sam_masks_small, sam_prep, rgb.shape[:2])
        sam_restore_delay_ms = (time.perf_counter() - sam_restore_start) * 1000.0
        sam_delay_ms = (time.perf_counter() - sam_start) * 1000.0
        sam_other_ms = max(
            0.0,
            sam_delay_ms - sam_prepare_delay_ms - sam_inference_delay_ms - sam_restore_delay_ms,
        )
        sam_prep_summary = {k: v for k, v in sam_prep.items() if k != "valid_depth_mask_sam"}

        return {
            "frame": frame,
            "rgb": rgb,
            "depth": depth,
            "tx": tx,
            "rot_m": rot_m,
            "sam_masks": sam_masks,
            "input_age_ms": input_age_ms,
            "sam_prep_summary": sam_prep_summary,
            "timing": {
                "image_conversion_delay_ms": image_conversion_delay_ms,
                "sam_prepare_ms": sam_prepare_delay_ms,
                "sam_inference_ms": sam_inference_delay_ms,
                "sam_restore_ms": sam_restore_delay_ms,
                "sam_other_ms": sam_other_ms,
                "sam_delay_ms": sam_delay_ms,
                "segmentation_stage_elapsed_ms": (time.perf_counter() - start) * 1000.0,
            },
            # Marks the handoff point to the tracking/publish thread so that
            # thread can measure how long its own FIFO wait was.
            "stage_a_complete_monotonic": time.perf_counter(),
        }

    def run_tracking_publish_stage(
        self, stage: Dict[str, Any], sam_output_queue_wait_ms: float = 0.0
    ) -> Phase1ClassificationResult:
        """Run slot assignment, label-map construction, and result assembly.

        Consumes the output of ``run_segmentation_stage``, executing on the
        tracking/publish thread while the segmentation thread is free to
        already be running SAM on the next frame.
        """
        start = time.perf_counter()
        frame = stage["frame"]
        rgb = stage["rgb"]
        depth = stage["depth"]
        tx = stage["tx"]
        rot_m = stage["rot_m"]
        sam_masks = stage["sam_masks"]
        input_age_ms = float(stage["input_age_ms"])
        timing = stage["timing"]

        rap_start = time.perf_counter()
        rap_frame_start = time.perf_counter()
        classified, track_records, frame_stage_ms = self.run_rap_and_metadata(frame, rgb, depth, tx, rot_m, sam_masks)
        run_rap_and_metadata_ms = (time.perf_counter() - rap_frame_start) * 1000.0
        measured_rap_frame_ms = sum(
            float(frame_stage_ms.get(key, 0.0))
            for key in (
                "geometry_metadata_ms", "frame_assignment_ms",
                "track_association_ms", "crop_update_ms",
            )
        )
        frame_stage_ms["run_rap_other_ms"] = max(0.0, run_rap_and_metadata_ms - measured_rap_frame_ms)
        semantic_label_dispatches: List[Dict[str, Any]] = []
        current_timestamp_sec = float(stamp_to_float(frame.header.stamp))
        self._latest_processed_timestamp_sec = current_timestamp_sec
        stage_start = time.perf_counter()
        if self.config.persistent_tracking_enabled:
            self._publish_active_local_segments(frame, track_records, current_timestamp_sec)
        frame_stage_ms["active_segments_publish_ms"] = (time.perf_counter() - stage_start) * 1000.0
        stage_start = time.perf_counter()
        if self.config.persistent_tracking_enabled and self.config.semantic_labeling_enabled:
            semantic_label_dispatches = self._dispatch_tracks_after_settling(current_timestamp_sec)
        frame_stage_ms["semantic_dispatch_ms"] = (time.perf_counter() - stage_start) * 1000.0
        stage_start = time.perf_counter()
        if self.config.persistent_tracking_enabled and self.config.semantic_labeling_enabled:
            self._release_quality_deferred_vlm_if_expired(current_timestamp_sec)
        frame_stage_ms["quality_deferred_release_ms"] = (time.perf_counter() - stage_start) * 1000.0
        rap_delay_ms = (time.perf_counter() - rap_start) * 1000.0

        label_start = time.perf_counter()
        semantic, instance, label_table, objects, unknowns = self.label_map_builder.build(rgb.shape[:2], classified)
        label_map_delay_ms = (time.perf_counter() - label_start) * 1000.0
        metadata_start = time.perf_counter()
        # With asynchronous RAP, unknown-vs-known is decided by the RAP worker.
        # The worker queues VLM only for unresolved slots, so no RAP/VLM work
        # blocks this frame before it is published to Hydra.
        vlm_dispatch = [] if self.config.rap_enabled else self.dispatch_unknowns_to_vlm(frame, rgb, depth, unknowns, classified)
        metadata = self.build_result_metadata(
            frame, sam_masks, objects, unknowns, vlm_dispatch, track_records, semantic_label_dispatches
        )
        metadata["diagnostic_stage_ms"] = {
            "sam_prepare_ms": timing["sam_prepare_ms"],
            "sam_inference_ms": timing["sam_inference_ms"],
            "sam_restore_ms": timing["sam_restore_ms"],
            "sam_other_ms": timing["sam_other_ms"],
            "sam_output_queue_wait_ms": float(sam_output_queue_wait_ms),
            **frame_stage_ms,
        }
        metadata["sam_input_processing"] = stage["sam_prep_summary"]
        metadata_delay_ms = (time.perf_counter() - metadata_start) * 1000.0

        # Building ROS Image messages and JSON strings is a real cost and can be
        # significant with large label maps/metadata. Measure it separately.
        result_msg_start = time.perf_counter()
        result = Phase1ClassificationResult()
        result.header = frame.header
        result.rsg_frame_id = frame.rsg_frame_id
        result.sequence = frame.sequence
        result.success = True
        result.status = "ok"
        result.reason = "ok"
        result.semantic_labels = self.bridge.cv2_to_imgmsg(semantic, encoding=self.config.semantic_label_encoding)
        # Semantic/instance images are pixel-aligned with the RGB image, so their
        # headers must use the camera optical frame and timestamp. Hydra treats
        # these as image streams, not world-frame messages.
        result.semantic_labels.header = frame.rgb.header
        result.instance_labels = self.bridge.cv2_to_imgmsg(instance, encoding=self.config.instance_label_encoding)
        result.instance_labels.header = frame.rgb.header
        result.label_table_json = safe_json_dumps(label_table)
        result.object_metadata_json = safe_json_dumps(objects if self.config.include_object_metadata else [])
        result.unknown_candidates_json = safe_json_dumps(unknowns if self.config.include_unknown_objects else [])
        result.vlm_dispatch_json = safe_json_dumps(vlm_dispatch)
        result.metadata_json = safe_json_dumps(metadata)
        result.input_age_ms = float(input_age_ms)
        result.sam_delay_ms = float(timing["sam_delay_ms"])
        result.rap_delay_ms = float(rap_delay_ms)
        result.label_map_delay_ms = float(label_map_delay_ms)
        result.metadata_delay_ms = float(metadata_delay_ms)
        result.image_conversion_delay_ms = float(timing["image_conversion_delay_ms"])
        result.result_message_build_delay_ms = (time.perf_counter() - result_msg_start) * 1000.0
        result.classifier_debug_record_delay_ms = 0.0
        result.num_masks = int(len(sam_masks))
        result.num_known = int(len([obj for obj in objects if not str(obj.get("status", "")).startswith("unknown")]))
        result.num_unknown = int(len(unknowns))
        # Total processing time across both stages, excluding the inter-stage
        # queue wait -- that wait is measured separately as
        # sam_output_queue_wait_ms so it is never silently absorbed here.
        result.classifier_delay_ms = (
            float(timing["segmentation_stage_elapsed_ms"]) + (time.perf_counter() - start) * 1000.0
        )
        return result

    def run_sam(self, rgb: np.ndarray, min_area_px: Optional[int] = None) -> List[SamMask]:
        """Run configured SAM backend and filter tiny masks at the SAM input scale."""
        if not self.config.sam_enabled:
            return []
        min_area = max(1, int(min_area_px if min_area_px is not None else self.config.sam_min_mask_pixels))
        masks = self.sam_backend.segment(rgb)
        filtered: List[SamMask] = []
        for mask in masks[: self.config.sam_max_masks]:
            if int(mask.area_px) >= min_area:
                filtered.append(mask)
        return filtered

    def prepare_sam_input(self, rgb: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Return the image sent to SAM and metadata needed to restore masks.

        The processing order is intentionally simple and deterministic:

        1. resize RGB/depth by ``phase1.sam.input_scale_ratio``;
        2. optionally compute a valid-depth mask using the configured distance
           interval;
        3. optionally crop SAM input to the valid-depth bounding ROI;
        4. replace invalid-depth pixels with a constant background colour so
           SAM is not asked to segment pixels outside the distance gate.
        """
        orig_h, orig_w = rgb.shape[:2]
        scale = float(self.config.sam_input_scale_ratio)
        scale = max(0.05, min(1.0, scale))
        proc_h = max(1, int(round(orig_h * scale)))
        proc_w = max(1, int(round(orig_w * scale)))

        if proc_h != orig_h or proc_w != orig_w:
            try:
                import cv2  # type: ignore
                interp = self.cv_resize_interpolation(self.config.sam_resize_interpolation)
                rgb_proc = cv2.resize(rgb, (proc_w, proc_h), interpolation=interp)
                depth_proc = cv2.resize(depth, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
            except Exception:
                # Fallback for environments without OpenCV. This is less exact
                # but keeps dummy/unit tests functional.
                y_idx = np.linspace(0, orig_h - 1, proc_h).astype(np.int64)
                x_idx = np.linspace(0, orig_w - 1, proc_w).astype(np.int64)
                rgb_proc = rgb[np.ix_(y_idx, x_idx)]
                depth_proc = depth[np.ix_(y_idx, x_idx)]
        else:
            rgb_proc = rgb.copy()
            depth_proc = depth

        meta: Dict[str, Any] = {
            "scale_ratio": float(scale),
            "original_shape_hw": [int(orig_h), int(orig_w)],
            "processed_shape_hw": [int(proc_h), int(proc_w)],
            "sam_shape_hw": [int(proc_h), int(proc_w)],
            "roi_xyxy": None,
            "depth_filter_enabled": bool(self.config.sam_depth_filter_enabled),
            "depth_filter_valid_ratio": None,
            "depth_filter_effective_range_m": [
                float(self.config.sam_depth_filter_min_m),
                float(self.config.sam_depth_filter_max_m),
            ],
        }

        if not self.config.sam_depth_filter_enabled:
            meta["valid_depth_mask_sam"] = None
            return rgb_proc, meta

        valid = (
            np.isfinite(depth_proc)
            & (depth_proc >= float(self.config.sam_depth_filter_min_m))
            & (depth_proc <= float(self.config.sam_depth_filter_max_m))
        )
        valid_ratio = float(np.count_nonzero(valid)) / float(valid.size) if valid.size else 0.0
        meta["depth_filter_valid_ratio"] = valid_ratio

        background = np.array(self.config.sam_depth_filter_background_value, dtype=rgb_proc.dtype).reshape(1, 1, 3)

        if valid_ratio < float(self.config.sam_depth_filter_min_valid_ratio):
            # Too little valid depth: give SAM a blank image and restore no masks.
            rgb_blank = np.zeros_like(rgb_proc)
            if rgb_blank.ndim == 3 and rgb_blank.shape[2] == 3:
                rgb_blank[:] = background
            meta["valid_depth_mask_sam"] = np.zeros(rgb_blank.shape[:2], dtype=bool)
            meta["depth_filter_rejected_by_valid_ratio"] = True
            return rgb_blank, meta

        if bool(self.config.sam_depth_filter_crop_to_roi) and np.any(valid):
            ys, xs = np.where(valid)
            margin = int(self.config.sam_depth_filter_roi_margin_px)
            x0 = max(0, int(xs.min()) - margin)
            y0 = max(0, int(ys.min()) - margin)
            x1 = min(proc_w, int(xs.max()) + margin + 1)
            y1 = min(proc_h, int(ys.max()) + margin + 1)
            rgb_sam = rgb_proc[y0:y1, x0:x1].copy()
            valid_sam = valid[y0:y1, x0:x1]
            meta["roi_xyxy"] = [int(x0), int(y0), int(x1), int(y1)]
            meta["sam_shape_hw"] = [int(rgb_sam.shape[0]), int(rgb_sam.shape[1])]
        else:
            rgb_sam = rgb_proc.copy()
            valid_sam = valid

        if rgb_sam.ndim == 3 and rgb_sam.shape[2] == 3:
            rgb_sam[~valid_sam] = background
        meta["valid_depth_mask_sam"] = valid_sam
        return rgb_sam, meta

    @staticmethod
    def cv_resize_interpolation(name: str) -> int:
        """Map a YAML interpolation name to an OpenCV constant."""
        import cv2  # type: ignore
        key = str(name).strip().lower()
        if key in {"nearest", "nearest_neighbor"}:
            return cv2.INTER_NEAREST
        if key in {"linear", "bilinear"}:
            return cv2.INTER_LINEAR
        if key in {"cubic", "bicubic"}:
            return cv2.INTER_CUBIC
        return cv2.INTER_AREA

    def scaled_min_mask_pixels(self, sam_prep: Dict[str, Any]) -> int:
        """Return the mask-area threshold at the SAM input scale."""
        orig_h, orig_w = [int(v) for v in sam_prep.get("original_shape_hw", [1, 1])]
        proc_h, proc_w = [int(v) for v in sam_prep.get("processed_shape_hw", [orig_h, orig_w])]
        area_scale = (float(proc_h * proc_w) / float(max(1, orig_h * orig_w)))
        return max(1, int(round(float(self.config.sam_min_mask_pixels) * area_scale)))

    def restore_sam_masks_to_original(
        self,
        masks: List[SamMask],
        sam_prep: Dict[str, Any],
        original_shape: Tuple[int, int],
    ) -> List[SamMask]:
        """Map masks from SAM input coordinates back to original RGB coordinates."""
        if not masks:
            return []
        try:
            import cv2  # type: ignore
        except Exception as exc:
            raise RuntimeError("OpenCV is required to restore scaled SAM masks to original image size") from exc

        orig_h, orig_w = int(original_shape[0]), int(original_shape[1])
        proc_h, proc_w = [int(v) for v in sam_prep.get("processed_shape_hw", [orig_h, orig_w])]
        roi = sam_prep.get("roi_xyxy")
        valid_sam = sam_prep.get("valid_depth_mask_sam")

        restored: List[SamMask] = []
        for idx, mask in enumerate(masks):
            sam_mask = np.asarray(mask.mask, dtype=bool)
            if valid_sam is not None and sam_mask.shape == np.asarray(valid_sam).shape:
                sam_mask = sam_mask & np.asarray(valid_sam, dtype=bool)

            proc_mask = np.zeros((proc_h, proc_w), dtype=np.uint8)
            if roi is not None:
                x0, y0, x1, y1 = [int(v) for v in roi]
                roi_h = max(0, y1 - y0)
                roi_w = max(0, x1 - x0)
                if sam_mask.shape[:2] != (roi_h, roi_w):
                    sam_mask_u8 = cv2.resize(sam_mask.astype(np.uint8), (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                else:
                    sam_mask_u8 = sam_mask.astype(np.uint8)
                proc_mask[y0:y1, x0:x1] = sam_mask_u8[:roi_h, :roi_w]
            else:
                if sam_mask.shape[:2] != (proc_h, proc_w):
                    proc_mask = cv2.resize(sam_mask.astype(np.uint8), (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
                else:
                    proc_mask = sam_mask.astype(np.uint8)

            full_mask = cv2.resize(proc_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            area = int(np.count_nonzero(full_mask))
            if area < int(self.config.sam_min_mask_pixels):
                continue
            ys, xs = np.where(full_mask)
            if xs.size == 0 or ys.size == 0:
                continue
            x_min = int(xs.min())
            y_min = int(ys.min())
            x_max = int(xs.max()) + 1
            y_max = int(ys.max()) + 1
            metadata = dict(mask.metadata or {})
            metadata.update({
                "restored_from_scaled_sam": True,
                "sam_input_scale_ratio": float(sam_prep.get("scale_ratio", 1.0)),
                "sam_depth_filter_enabled": bool(sam_prep.get("depth_filter_enabled", False)),
                "sam_depth_filter_valid_ratio": sam_prep.get("depth_filter_valid_ratio"),
                "sam_roi_xyxy": sam_prep.get("roi_xyxy"),
            })
            restored.append(
                SamMask(
                    mask_id=str(mask.mask_id or f"mask_{idx:03d}"),
                    mask=full_mask,
                    bbox_2d=[x_min, y_min, max(0, x_max - x_min), max(0, y_max - y_min)],
                    area_px=area,
                    crop=mask.crop,
                    score=float(mask.score),
                    metadata=metadata,
                )
            )
            if len(restored) >= int(self.config.sam_max_masks):
                break
        return restored

    def run_rap_and_metadata(
        self,
        frame: RsgFrame,
        rgb: np.ndarray,
        depth: np.ndarray,
        tx: np.ndarray,
        rot_m: np.ndarray,
        sam_masks: List[SamMask],
    ) -> Tuple[List[ClassifiedMask], List[Dict[str, Any]], Dict[str, float]]:
        """Build one stable Hydra object slot per physical object.

        Each physical object receives a persistent slot immediately. The
        per-track RAP worker runs asynchronously after the fixed crop-settling
        window and publishes only a later slot-to-label semantic update. Spatial
        geometry and Hydra slot assignment never wait for RAP or VLM.
        """
        classified: List[ClassifiedMask] = []
        track_records: List[Dict[str, Any]] = []
        next_instance_id = 1
        timestamp_sec = stamp_to_float(frame.header.stamp)
        timing_enabled = self.config.timing_enabled
        geometry_ms = 0.0
        assignment_ms = 0.0
        association_ms = 0.0
        crop_update_ms = 0.0
        # Part 2 profiling only: accumulates ObjectGeometryEstimator sub-step
        # timing across every mask in this frame. Never read by any non-timing
        # code path; does not affect any published value. See
        # docs/PHASE1_LATENCY_OPTIMIZATION_PROPOSAL.md / optimisation_part2.
        geometry_stage_ms: Optional[Dict[str, float]] = {} if timing_enabled else None
        # Part 3 Path B profiling only: accumulates associate()'s lock-wait
        # time across every mask in this frame. Never read by any
        # non-timing code path; does not affect any published value. See
        # debug/optimisation/optimisation_part3/PART3_REPORT.md.
        association_stage_ms: Optional[Dict[str, float]] = {} if timing_enabled else None
        if self.config.persistent_tracking_enabled:
            self.persistent_tracker.begin_frame()

        # Build geometry for every SAM observation before mutating any track.
        # This enables track-aware mask redundancy analysis (A2) and one global
        # frame-level assignment (E), eliminating SAM-output-order bias.
        prepared: List[Dict[str, Any]] = []
        for idx, mask in enumerate(sam_masks):
            stage_start = time.perf_counter() if timing_enabled else 0.0
            candidate_id = self.make_candidate_id(frame, mask.mask_id, idx, False)
            metadata = self.build_object_metadata(
                frame=frame, mask=mask, depth=depth, tx=tx, rot_m=rot_m,
                label="unknown_object", label_id=0, instance_id=idx + 1,
                confidence=0.0, status="collecting_best_crop",
                candidate_id=candidate_id, rap_metadata={},
                geometry_stage_ms=geometry_stage_ms,
            )
            prepared.append({
                "metadata": metadata, "mask": mask.mask,
                "timestamp_sec": timestamp_sec, "desired_hydra_label_id": 0,
            })
            if timing_enabled:
                geometry_ms += (time.perf_counter() - stage_start) * 1000.0
        # Part 3 profiling only: accumulates prepare_frame_assignments'
        # internal sub-step timing for this frame. Never read by any
        # non-timing code path; does not affect any published value. See
        # debug/optimisation/optimisation_part3/PART3_REPORT.md.
        assignment_stage_ms: Optional[Dict[str, float]] = {} if timing_enabled else None
        stage_start = time.perf_counter() if timing_enabled else 0.0
        keep_mask = (self.persistent_tracker.prepare_frame_assignments(prepared, stage_ms=assignment_stage_ms)
                     if self.config.persistent_tracking_enabled else [True] * len(sam_masks))
        if timing_enabled:
            assignment_ms = (time.perf_counter() - stage_start) * 1000.0

        for idx, mask in enumerate(sam_masks):
            if not keep_mask[idx]:
                continue
            # Do not query RAP inline. Every new physical object receives its
            # slot immediately; one background RAP job is queued only after the
            # fixed settling window has collected a representative crop.
            rap_info: Dict[str, Any] = {
                "label": "unknown_object", "confidence": 0.0,
                "is_known": False, "metadata": {}, "status": "queued_after_first_valid_crop",
            }
            rap_label = "unknown_object"
            rap_known = False
            raw_label = "unknown_object"
            use_known_class = False
            forced_slot_id = 0
            desired_label_id, desired_label_name = 0, "unknown"
            candidate_id = self.make_candidate_id(frame, mask.mask_id, idx, False)
            status = "collecting_best_crop"
            metadata = dict(prepared[idx]["metadata"])

            semantic_label_id = desired_label_id
            semantic_label_name = desired_label_name
            instance_id = next_instance_id
            track_record: Dict[str, Any] = {}

            if self.config.persistent_tracking_enabled:
                stage_start = time.perf_counter() if timing_enabled else 0.0
                # ``new_track_use_hydra_slot`` applies only when there is no
                # geometry match.  Matching slot 24 always remains slot 24,
                # even if RAP now returns "chair".
                metadata, track_record = self.persistent_tracker.associate(
                    metadata=metadata,
                    frame_id=frame.rsg_frame_id,
                    sequence=int(frame.sequence),
                    timestamp_sec=timestamp_sec,
                    desired_hydra_label_id=desired_label_id,
                    desired_hydra_label_name=desired_label_name,
                    raw_label=rap_label if rap_known else "",
                    label_source="rap" if rap_known else "pending",
                    label_confidence=float(rap_info.get("confidence", 0.0) or 0.0),
                    # Known labels loaded from the frozen registry share
                    # their canonical class slot. Unresolved/new labels obtain
                    # a unique temporary slot for this session.
                    new_track_use_hydra_slot=not use_known_class,
                    forced_hydra_slot_id=int(forced_slot_id),
                    stage_ms=association_stage_ms,
                )
                track_record.update({
                    "frame_id": frame.rsg_frame_id,
                    "sequence": int(frame.sequence),
                    "candidate_id": candidate_id,
                })
                track_records.append(track_record)
                semantic_label_id = int(metadata.get("hydra_label_id", semantic_label_id) or 0)
                semantic_label_name = str(metadata.get("hydra_label_name", semantic_label_name))
                instance_id = int(metadata.get("persistent_instance_id", instance_id) or 0)
                if timing_enabled:
                    association_ms += (time.perf_counter() - stage_start) * 1000.0

            external_track_id = str(metadata.get("persistent_track_id", "")) or None
            # Keep updating the shared best crop. RAP/VLM receive only this
            # track ID and retrieve the latest crop when each worker dequeues it.
            stage_start = time.perf_counter() if timing_enabled else 0.0
            self._remember_track_crop(external_track_id, rgb, metadata, frame, mask.mask)
            if timing_enabled:
                crop_update_ms += (time.perf_counter() - stage_start) * 1000.0

            # VLM remains a one-shot fallback only when the asynchronous RAP
            # lookup cannot identify this slot.
            unresolved_for_vlm = False

            semantic_kind = str(metadata.get("semantic_kind", "slot" if semantic_label_id >= self.config.persistent_slot_first_label_id else "class"))
            active_slot = semantic_kind == "slot"
            metadata.update({
                "label_id": int(semantic_label_id),
                "instance_id": int(instance_id),
                "hydra_label_id": int(semantic_label_id),
                "hydra_label_name": str(semantic_label_name),
                "label": str(semantic_label_name),
                "status": "unknown_slot" if active_slot else status,
                "rap_status": str(rap_info.get("status", "pending" if self.rap_runs_async else "unknown")),
                "semantic_label_source": "hydra_slot" if active_slot else "rap_registry_class",
                "semantic_label_confidence": float(rap_info.get("confidence", 0.0) or 0.0),
                "canonical_label": str(metadata.get("canonical_label", raw_label)),
            })

            classified_mask = ClassifiedMask(
                mask_id=mask.mask_id,
                mask=mask.mask,
                label=str(semantic_label_name),
                label_id=int(semantic_label_id),
                instance_id=int(instance_id),
                confidence=float(rap_info.get("confidence", 0.0) or 0.0),
                status=str(metadata["status"]),
                candidate_id=candidate_id,
                metadata=metadata,
            )
            classified.append(classified_mask)

            metadata["rap_dispatch_status"] = "track_id_pending_rap" if self.config.rap_enabled else "rap_disabled"
            next_instance_id += 1
        result_stage_ms = {
            "geometry_metadata_ms": geometry_ms,
            "frame_assignment_ms": assignment_ms,
            "track_association_ms": association_ms,
            "crop_update_ms": crop_update_ms,
        }
        if geometry_stage_ms is not None:
            result_stage_ms.update(geometry_stage_ms)
        if assignment_stage_ms is not None:
            result_stage_ms.update(assignment_stage_ms)
        if association_stage_ms is not None:
            result_stage_ms.update(association_stage_ms)
        return classified, track_records, result_stage_ms

    def _classify_rap_synchronously(self, rgb: np.ndarray, mask: SamMask, mask_index: int) -> Dict[str, Any]:
        """Return one RAP decision before Hydra publication in reuse mode."""
        if not self.config.rap_enabled:
            return {"label": "unknown_object", "confidence": 0.0, "is_known": False, "metadata": {}, "status": "disabled", "delay_ms": 0.0}
        start = time.perf_counter()
        try:
            rap_crop = self._build_rap_semantic_crop(rgb, mask.mask, mask.bbox_2d)
            if rap_crop is None or rap_crop.size == 0:
                raise RuntimeError("Synchronous RAP target-only crop is empty")
            height, width = rap_crop.shape[:2]
            synthetic_mask = SamMask(
                mask_id=str(mask.mask_id),
                mask=np.ones((height, width), dtype=bool),
                bbox_2d=[0, 0, int(width), int(height)],
                area_px=int(height * width),
                crop=rap_crop,
                score=float(mask.score),
                metadata={**dict(mask.metadata or {}), "semantic_crop_representation": "target_only"},
            )
            rap = self.rap_backend.classify(rap_crop, synthetic_mask, int(mask_index))
            confidence = float(rap.confidence)
            is_known = bool(rap.is_known and confidence >= self.config.rap_confidence_threshold)
            return {
                "label": str(rap.label or "unknown_object"),
                "confidence": confidence,
                "is_known": is_known,
                "metadata": dict(rap.metadata or {}),
                "status": "known" if is_known else "unknown",
                "delay_ms": (time.perf_counter() - start) * 1000.0,
            }
        except Exception as exc:
            self.get_logger().warn(f"Synchronous RAP lookup failed: {exc}")
            return {"label": "unknown_object", "confidence": 0.0, "is_known": False, "metadata": {"error": str(exc)}, "status": "error", "delay_ms": (time.perf_counter() - start) * 1000.0}

    def _score_track_crop(
        self,
        metadata: Dict[str, Any],
        image_shape: Tuple[int, int],
        bbox_2d: Any,
    ) -> Dict[str, Any]:
        """Score one candidate crop for semantic usefulness rather than depth alone.

        A thin NanoSAM fragment can have a high valid-depth ratio while still
        being unusable for visual recognition. The score therefore combines the
        raw object-box size, mask support, depth support, mask fill, and image
        border truncation. Long objects are not penalised for aspect ratio.
        """
        image_height, image_width = [max(1, int(value)) for value in image_shape]
        if not bbox_2d or len(bbox_2d) != 4:
            return {
                "vlm_crop_quality_score": 0.0,
                "vlm_crop_quality_eligible": False,
                "vlm_crop_quality_reasons": ["invalid_bbox"],
                "vlm_crop_border_edges": 0,
            }

        x, y, width, height = [int(value) for value in bbox_2d]
        width = max(0, min(width, image_width))
        height = max(0, min(height, image_height))
        bbox_area = int(width * height)
        short_side = int(min(width, height))
        mask_area = max(0.0, float(metadata.get("mask_area_px", 0) or 0.0))
        depth_valid_ratio = max(0.0, min(1.0, float(metadata.get("depth_valid_ratio", 0.0) or 0.0)))
        fill_ratio = max(0.0, min(1.0, mask_area / float(max(1, bbox_area))))

        border_edges = int(x <= 0) + int(y <= 0) + int(x + width >= image_width) + int(y + height >= image_height)
        area_score = min(1.0, bbox_area / float(max(1, self.config.vlm_crop_target_area_px)))
        mask_score = min(1.0, mask_area / float(max(1, self.config.vlm_crop_target_area_px)))
        short_side_score = min(1.0, short_side / float(max(1, self.config.vlm_crop_target_short_side_px)))
        border_factor = max(0.35, 1.0 - float(self.config.vlm_crop_border_penalty) * border_edges)

        score = (
            0.28 * area_score
            + 0.24 * mask_score
            + 0.22 * short_side_score
            + 0.16 * depth_valid_ratio
            + 0.10 * fill_ratio
        ) * border_factor
        score = max(0.0, min(1.0, float(score)))

        reasons: List[str] = []
        if bbox_area < int(self.config.vlm_crop_min_area_px):
            reasons.append("crop_area_below_minimum")
        if short_side < int(self.config.vlm_crop_min_short_side_px):
            reasons.append("crop_short_side_below_minimum")
        if score < float(self.config.vlm_crop_min_quality_score):
            reasons.append("crop_quality_below_minimum")
        if border_edges >= 2:
            reasons.append("object_box_heavily_border_clipped")

        return {
            "vlm_crop_quality_score": score,
            "vlm_crop_quality_eligible": not reasons,
            "vlm_crop_quality_reasons": reasons,
            "vlm_crop_border_edges": border_edges,
            "vlm_crop_object_bbox_area_px": bbox_area,
            "vlm_crop_object_short_side_px": short_side,
            "vlm_crop_mask_fill_ratio": fill_ratio,
            "vlm_crop_depth_valid_ratio": depth_valid_ratio,
        }

    def _build_rap_semantic_crop(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray],
        object_bbox_2d: Any,
        prepared_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Return the exact target-only image supplied to RAP."""
        if not bool(self.config.semantic_crop_rap_target_only_enabled):
            return self.extract_crop(rgb, object_bbox_2d)
        return build_rap_target_only_crop(
            rgb,
            mask,
            object_bbox_2d,
            background_rgb=self.config.semantic_crop_rap_background_rgb,
            cleanup_mask=self.config.semantic_crop_mask_cleanup_enabled,
            cleanup_min_component_area_ratio=self.config.semantic_crop_mask_cleanup_min_component_area_ratio,
            cleanup_component_max_gap_px=self.config.semantic_crop_mask_cleanup_component_max_gap_px,
            prepared_mask=prepared_mask,
        )

    def _build_vlm_semantic_crop(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray],
        context_bbox_2d: Any,
        prepared_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Return the exact target-focused image supplied to the VLM."""
        if not bool(self.config.semantic_crop_vlm_target_focus_enabled):
            return self.extract_crop(rgb, context_bbox_2d)
        return build_vlm_target_focus_crop(
            rgb,
            mask,
            context_bbox_2d,
            context_alpha=self.config.semantic_crop_vlm_context_alpha,
            grayscale_context=self.config.semantic_crop_vlm_context_grayscale,
            near_context_enabled=self.config.semantic_crop_vlm_near_context_enabled,
            near_context_alpha=self.config.semantic_crop_vlm_near_context_alpha,
            near_context_dilation_px=self.config.semantic_crop_vlm_near_context_dilation_px,
            near_context_grayscale=self.config.semantic_crop_vlm_near_context_grayscale,
            cleanup_mask=self.config.semantic_crop_mask_cleanup_enabled,
            cleanup_min_component_area_ratio=self.config.semantic_crop_mask_cleanup_min_component_area_ratio,
            cleanup_component_max_gap_px=self.config.semantic_crop_mask_cleanup_component_max_gap_px,
            draw_target_contour=self.config.semantic_crop_draw_target_contour,
            contour_rgb=self.config.semantic_crop_target_contour_rgb,
            contour_thickness_px=self.config.semantic_crop_target_contour_thickness_px,
            prepared_mask=prepared_mask,
        )

    def _remember_track_crop(
        self,
        track_id: Optional[str],
        rgb: np.ndarray,
        metadata: Dict[str, Any],
        frame: RsgFrame,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        """Keep one immutable source ROI for the best observation of a track.

        Semantic rendering is intentionally deferred to the RAP/VLM worker
        that dequeues the track.  The frame-critical path therefore performs
        only one bounded RGB/mask ROI copy and never renders two crops that may
        be replaced before either worker consumes them.
        """
        if not track_id:
            return
        key = str(track_id)
        if not self.persistent_tracker.is_semantic_labeling_open(key):
            return

        bbox_2d = metadata.get("bbox_2d")
        context_bbox_2d = context_bbox_xywh(
            rgb.shape[:2],
            bbox_2d,
            context_ratio=float(self.config.vlm_crop_context_ratio),
        )
        if not context_bbox_2d:
            return
        timestamp_sec = float(stamp_to_float(frame.header.stamp))
        crop_quality = self._score_track_crop(metadata, rgb.shape[:2], bbox_2d)
        score = float(crop_quality["vlm_crop_quality_score"])

        # Score before mask cleanup or rendering. A lower-scoring observation
        # cannot replace the current crop, so only refresh track recency.
        with self._track_crop_lock:
            current = self._track_best_crops.get(key)
            if current is not None and score < float(current.get("score", -1.0)):
                current["last_observed_timestamp_sec"] = timestamp_sec
                return

        context_x, context_y, context_width, context_height = context_bbox_2d
        image_height, image_width = rgb.shape[:2]
        bbox_x, bbox_y, bbox_width, bbox_height = [int(value) for value in bbox_2d]
        target_x0 = max(0, min(image_width, bbox_x))
        target_y0 = max(0, min(image_height, bbox_y))
        target_x1 = max(0, min(image_width, bbox_x + bbox_width))
        target_y1 = max(0, min(image_height, bbox_y + bbox_height))
        if target_x1 <= target_x0 or target_y1 <= target_y0:
            return
        target_bbox_in_roi = [
            int(target_x0 - context_x),
            int(target_y0 - context_y),
            int(target_x1 - target_x0),
            int(target_y1 - target_y0),
        ]

        source_rgb = np.array(
            rgb[
                context_y:context_y + context_height,
                context_x:context_x + context_width,
            ],
            copy=True,
            order="C",
        )
        if source_rgb.size == 0:
            return

        source_mask = None
        semantic_rendering_enabled = bool(
            self.config.semantic_crop_rap_target_only_enabled
            or self.config.semantic_crop_vlm_target_focus_enabled
        )
        if semantic_rendering_enabled:
            if mask is None:
                return
            mask_array = np.asarray(mask)
            if mask_array.shape != rgb.shape[:2]:
                return
            source_mask = np.array(
                mask_array[
                    context_y:context_y + context_height,
                    context_x:context_x + context_width,
                ],
                dtype=bool,
                copy=True,
                order="C",
            )
            if source_mask.shape != source_rgb.shape[:2]:
                return

        # Revisions are immutable after publication to the registry. Workers
        # can safely retain these references after releasing the registry lock.
        source_rgb.setflags(write=False)
        if source_mask is not None:
            source_mask.setflags(write=False)

        crop_metadata = dict(metadata)
        crop_metadata.update(crop_quality)
        crop_metadata.update({
            "vlm_crop_context_bbox_2d": context_bbox_2d,
            "vlm_crop_context_ratio": float(self.config.vlm_crop_context_ratio),
            "vlm_crop_width_px": int(context_width),
            "vlm_crop_height_px": int(context_height),
            "rap_crop_representation": "target_only" if self.config.semantic_crop_rap_target_only_enabled else "raw_bbox",
            "vlm_crop_representation": "target_full_colour_local_halo_dimmed_context" if self.config.semantic_crop_vlm_target_focus_enabled else "raw_context",
            "vlm_context_alpha": float(self.config.semantic_crop_vlm_context_alpha),
            "vlm_context_grayscale": bool(self.config.semantic_crop_vlm_context_grayscale),
            "vlm_near_context_enabled": bool(self.config.semantic_crop_vlm_near_context_enabled),
            "vlm_near_context_alpha": float(self.config.semantic_crop_vlm_near_context_alpha),
            "vlm_near_context_dilation_px": int(self.config.semantic_crop_vlm_near_context_dilation_px),
            "vlm_near_context_grayscale": bool(self.config.semantic_crop_vlm_near_context_grayscale),
            "semantic_crop_mask_cleanup_enabled": bool(self.config.semantic_crop_mask_cleanup_enabled),
            "semantic_crop_mask_cleanup_min_component_area_ratio": float(self.config.semantic_crop_mask_cleanup_min_component_area_ratio),
            "semantic_crop_mask_cleanup_component_max_gap_px": int(self.config.semantic_crop_mask_cleanup_component_max_gap_px),
        })
        updated = False
        previous_score: Optional[float] = None
        revision = 0
        selection_reason = "lower_score_than_current_best"

        with self._track_crop_lock:
            current = self._track_best_crops.get(key)
            if current is None:
                revision = 1
                selection_reason = "first_valid_crop"
                self._track_best_crops[key] = {
                    "score": score,
                    "source_rgb": source_rgb,
                    "source_mask": source_mask,
                    "target_bbox_in_roi": target_bbox_in_roi,
                    "crop_revision": revision,
                    "first_crop_timestamp_sec": timestamp_sec,
                    "last_crop_update_timestamp_sec": timestamp_sec,
                    "last_observed_timestamp_sec": timestamp_sec,
                    "frame_header": frame.header,
                    "frame_id": frame.rsg_frame_id,
                    "sequence": int(frame.sequence),
                    "timestamp_sec": timestamp_sec,
                    "candidate_id": str(metadata.get("candidate_id", "")),
                    "centroid_frame_id": str(frame.camera_pose.header.frame_id or frame.header.frame_id or ""),
                    "object_metadata": crop_metadata,
                }
                updated = True
            else:
                previous_score = float(current.get("score", -1.0))
                current["last_observed_timestamp_sec"] = timestamp_sec
                revision = int(current.get("crop_revision", 0) or 0)
                if score >= previous_score:
                    revision += 1
                    selection_reason = "score_not_lower_than_current_best"
                    self._track_best_crops[key] = {
                        "score": score,
                        "source_rgb": source_rgb,
                        "source_mask": source_mask,
                        "target_bbox_in_roi": target_bbox_in_roi,
                        "crop_revision": revision,
                        "first_crop_timestamp_sec": float(current.get("first_crop_timestamp_sec", timestamp_sec) or timestamp_sec),
                        "last_crop_update_timestamp_sec": timestamp_sec,
                        "last_observed_timestamp_sec": timestamp_sec,
                        "frame_header": frame.header,
                        "frame_id": frame.rsg_frame_id,
                        "sequence": int(frame.sequence),
                        "timestamp_sec": timestamp_sec,
                        "candidate_id": str(metadata.get("candidate_id", "")),
                        "centroid_frame_id": str(frame.camera_pose.header.frame_id or frame.header.frame_id or ""),
                        "object_metadata": crop_metadata,
                    }
                    updated = True

        if updated:
            self._resume_quality_deferred_vlm_if_ready(key)

    def _describe_track_crop(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Return queue-safe crop metadata without copying image payloads."""
        key = str(track_id)
        with self._track_crop_lock:
            best = self._track_best_crops.get(key)
            if best is None:
                return None
            metadata = dict(best.get("object_metadata") or {})
            return {
                "best_frame_score": float(best.get("score", 0.0) or 0.0),
                "crop_revision": int(best.get("crop_revision", 0) or 0),
                "crop_timestamp_sec": float(best.get("timestamp_sec", 0.0) or 0.0),
                "last_crop_update_timestamp_sec": float(best.get("last_crop_update_timestamp_sec", 0.0) or 0.0),
                "last_observed_timestamp_sec": float(best.get("last_observed_timestamp_sec", 0.0) or 0.0),
                "vlm_crop_quality_score": float(metadata.get("vlm_crop_quality_score", 0.0) or 0.0),
                "vlm_crop_quality_eligible": bool(metadata.get("vlm_crop_quality_eligible", False)),
                "vlm_crop_quality_reasons": list(metadata.get("vlm_crop_quality_reasons", []) or []),
                "vlm_crop_width_px": int(metadata.get("vlm_crop_width_px", 0) or 0),
                "vlm_crop_height_px": int(metadata.get("vlm_crop_height_px", 0) or 0),
            }

    def _retire_track_crop(self, track_id: str) -> None:
        """Release a completed track's crop after its final semantic result."""
        key = str(track_id)
        if not key:
            return
        with self._track_crop_lock:
            self._track_best_crops.pop(key, None)

    def _snapshot_track_task(self, track_id: str, stage: str) -> Optional[Dict[str, Any]]:
        """Render one immutable ROI revision for the dequeuing worker."""
        key = str(track_id)
        with self._track_crop_lock:
            best = self._track_best_crops.get(key)
            if best is None:
                return None
            source_rgb = best.get("source_rgb")
            source_mask = best.get("source_mask")
            target_bbox_in_roi = list(best.get("target_bbox_in_roi") or [])
            if source_rgb is None or getattr(source_rgb, "size", 0) == 0:
                return None
            if len(target_bbox_in_roi) != 4:
                return None
            metadata = dict(best.get("object_metadata") or {})
            score = float(best.get("score", 0.0) or 0.0)
            revision = int(best.get("crop_revision", 0) or 0)
            frame_header = best.get("frame_header")
            frame_id = str(best.get("frame_id", ""))
            sequence = int(best.get("sequence", 0) or 0)
            timestamp_sec = float(best.get("timestamp_sec", 0.0) or 0.0)
            candidate_id = str(best.get("candidate_id", ""))
            centroid_frame_id = str(best.get("centroid_frame_id", ""))
            last_crop_update_timestamp_sec = float(best.get("last_crop_update_timestamp_sec", timestamp_sec) or timestamp_sec)
            last_observed_timestamp_sec = float(best.get("last_observed_timestamp_sec", timestamp_sec) or timestamp_sec)

        # Render outside the registry lock. A VLM task also retains the RAP
        # representation from this exact revision so any later memory update
        # cannot pair the VLM label with a different observation.
        prepared_mask = None
        if bool(self.config.semantic_crop_rap_target_only_enabled) or bool(
            self.config.semantic_crop_vlm_target_focus_enabled
        ):
            prepared_mask = prepare_target_mask(
                source_rgb,
                source_mask,
                cleanup_enabled=self.config.semantic_crop_mask_cleanup_enabled,
                cleanup_min_component_area_ratio=self.config.semantic_crop_mask_cleanup_min_component_area_ratio,
                cleanup_component_max_gap_px=self.config.semantic_crop_mask_cleanup_component_max_gap_px,
            )
            if prepared_mask is None:
                return None

        object_crop = self._build_rap_semantic_crop(
            source_rgb,
            source_mask,
            target_bbox_in_roi,
            prepared_mask=prepared_mask,
        )
        if object_crop is None or object_crop.size == 0:
            return None

        vlm_crop = None
        if str(stage).startswith("vlm"):
            roi_height, roi_width = source_rgb.shape[:2]
            vlm_crop = self._build_vlm_semantic_crop(
                source_rgb,
                source_mask,
                [0, 0, int(roi_width), int(roi_height)],
                prepared_mask=prepared_mask,
            )
            if vlm_crop is None or vlm_crop.size == 0:
                return None

        with self._vlm_quality_deferred_lock:
            quality_timeout_forced = bool(key in self._vlm_quality_force_track_ids)
        slot_id = int(metadata.get("hydra_label_id", metadata.get("hydra_slot_id", 0)) or 0)
        slot_name = str(metadata.get("hydra_label_name", metadata.get("hydra_slot_name", "")))
        metadata.update({
            "persistent_track_id": key,
            "internal_object_id": str(metadata.get("internal_object_id", key)),
            "hydra_label_id": slot_id,
            "hydra_slot_id": slot_id,
            "hydra_label_name": slot_name,
            "hydra_slot_name": slot_name,
            "crop_score": score,
            "crop_revision": revision,
            "crop_stage": str(stage),
            "last_crop_update_timestamp_sec": last_crop_update_timestamp_sec,
            "last_observed_timestamp_sec": last_observed_timestamp_sec,
            "vlm_crop_quality_timeout_forced": quality_timeout_forced,
        })

        queued_time = self._rap_enqueued_monotonic.get(key) if stage.startswith("rap") else self._vlm_enqueued_monotonic.get(key)
        return {
            "persistent_track_id": key,
            "hydra_slot_id": slot_id,
            "hydra_slot_name": slot_name,
            "frame_header": frame_header,
            "frame_id": frame_id,
            "rsg_frame_id": frame_id,
            "sequence": sequence,
            "timestamp_sec": timestamp_sec,
            "candidate_id": candidate_id,
            "mask_id": str(metadata.get("mask_id", candidate_id)),
            "rgb_crop": object_crop,
            "vlm_rgb_crop": vlm_crop,
            "object_metadata": metadata,
            "centroid_frame_id": centroid_frame_id,
            "created_monotonic": float(queued_time if queued_time is not None else time.perf_counter()),
            "track_seen_count": int(metadata.get("persistent_track_seen_count", 0) or 0),
            "best_frame_score": score,
            "crop_revision": revision,
            "vlm_crop_quality_score": float(metadata.get("vlm_crop_quality_score", 0.0) or 0.0),
            "vlm_crop_quality_eligible": bool(metadata.get("vlm_crop_quality_eligible", False)),
            "vlm_crop_quality_reasons": list(metadata.get("vlm_crop_quality_reasons", []) or []),
            "last_crop_update_timestamp_sec": last_crop_update_timestamp_sec,
            "last_observed_timestamp_sec": last_observed_timestamp_sec,
            "vlm_crop_quality_timeout_forced": quality_timeout_forced,
            "queue_stage": str(stage),
        }

    @staticmethod
    def _normalise_label_key(label: Any) -> str:
        return " ".join(str(label or "").strip().lower().replace("_", " ").split())

    def _resolve_hydra_semantic_label(self, label_key: str, is_known: bool) -> Tuple[int, str]:
        """Compatibility shim: semantic classes never replace physical slots."""
        del label_key, is_known
        return 0, "unknown"

    def build_object_metadata(
        self,
        frame: RsgFrame,
        mask: SamMask,
        depth: np.ndarray,
        tx: np.ndarray,
        rot_m: np.ndarray,
        label: str,
        label_id: int,
        instance_id: int,
        confidence: float,
        status: str,
        candidate_id: str,
        rap_metadata: Dict[str, Any],
        geometry_stage_ms: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Create configurable object metadata used by Hydra/fusion/risk nodes."""
        geometry = self.geometry_estimator.estimate(mask.mask, depth, frame.camera_info, tx, rot_m, stage_ms=geometry_stage_ms)
        metadata = {
            "source_frame_id": frame.rsg_frame_id,
            "timestamp_sec": stamp_to_float(frame.header.stamp),
            "candidate_id": candidate_id,
            "mask_id": mask.mask_id,
            "label": label,
            "label_id": int(label_id),
            "instance_id": int(instance_id),
            "confidence": float(confidence),
            "status": status,
            "rap": rap_metadata,
            **geometry,
        }
        # Persistent association needs the raw 3D geometry even when JSON
        # metadata publication is disabled for performance. The LabelMapBuilder
        # still honours metadata.include_* flags when serialising outputs.
        if self.config.persistent_tracking_enabled:
            return metadata
        return filter_metadata(metadata, self.config)

    def _dispatch_tracks_after_settling(
        self,
        current_timestamp_sec: float,
        *,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """Schedule settled tracks for RAP using their current best crop.

        The raw crop is not placed into the queue. It remains in the shared
        registry throughout the fixed collection window and may still improve
        while the worker FIFO is waiting. Queue pressure defers an ID instead of
        losing the semantic job.
        """
        records: List[Dict[str, Any]] = []
        ready = self.persistent_tracker.prepare_active_for_labeling(
            current_timestamp_sec,
            force=force,
        )
        for event in ready:
            track_id = str(event.get("persistent_track_id", ""))
            crop_state = self._describe_track_crop(track_id) if track_id else None
            if not track_id or crop_state is None:
                if track_id:
                    self.persistent_tracker.release_labeling_request(track_id, "missing_representative_crop")
                continue

            # The RAP FIFO contains only the persistent track ID.  Do not copy
            # or freeze the crop here: later observations remain eligible until
            # the RAP worker actually dequeues this ID.
            status = self.enqueue_rap_task(track_id)
            if status in {"queued_for_rap", "deferred_for_rap"}:
                self.persistent_tracker.set_labeling_status(
                    track_id,
                    "rap_queued" if status == "queued_for_rap" else "rap_deferred",
                )
                with self._semantic_label_lock:
                    self._semantic_label_pending_track_ids.add(track_id)
            else:
                self.persistent_tracker.release_labeling_request(track_id, status)
            records.append({
                "persistent_track_id": track_id,
                "hydra_slot_id": int(event.get("hydra_slot_id", 0) or 0),
                "status": status,
                "best_crop_score": float(crop_state.get("best_frame_score", 0.0) or 0.0),
                "crop_revision": int(crop_state.get("crop_revision", 0) or 0),
                "crop_timestamp_sec": float(crop_state.get("crop_timestamp_sec", 0.0) or 0.0),
                "last_crop_update_timestamp_sec": float(crop_state.get("last_crop_update_timestamp_sec", 0.0) or 0.0),
                "settling_age_sec": float(event.get("settling_age_sec", 0.0) or 0.0),
                "settle_time_sec": float(event.get("settle_time_sec", 0.0) or 0.0),
                "forced_dispatch": bool(event.get("forced_dispatch", False)),
            })
        return records

    def _emit_semantic_label_result(self, event: Dict[str, Any], task: Dict[str, Any], *, source: str, finalize_track: bool = True) -> None:
        """Publish final class labels for every local semantic slot of one object.

        RAP/VLM still classifies the internal object track once.  If that object
        owns several local Hydra slots, the same class label is emitted once per
        slot so the fuser can label every Hydra node while keeping confidence
        timestamps slot-local.
        """
        # Do not start or continue semantic fan-out during shutdown.
        if self._stop_event.is_set() or not rclpy.ok():
            return

        payload = dict(event)
        source_name = str(source)
        label = str(payload.get("semantic_label") or payload.get("canonical_label") or "unclassified_object")
        if source_name in {"vlm_failed", "rap_unknown", "rap_error"}:
            label = "unknown_object"

        segments = list(payload.get("semantic_segments") or [])
        if not segments:
            segments = [{
                "hydra_slot_id": int(payload.get("hydra_slot_id", task.get("hydra_slot_id", 0)) or 0),
                "hydra_slot_name": str(payload.get("hydra_slot_name", task.get("hydra_slot_name", ""))),
                "local_segment_id": str(payload.get("local_segment_id", "")),
                "centroid_3d": payload.get("centroid_3d", (task.get("object_metadata") or {}).get("centroid_3d")),
            }]

        for segment in segments:
            # Abort a partially-started fan-out when ROS shutdown begins.
            if self._stop_event.is_set() or not rclpy.ok():
                break
            slot_id = int(segment.get("hydra_slot_id", segment.get("hydra_label_id", 0)) or 0)
            if slot_id <= 0:
                continue
            segment_payload = dict(payload)
            segment_payload.update({
                "event": "semantic_label_result",
                "source": source_name,
                "persistent_track_id": str(payload.get("persistent_track_id", task.get("persistent_track_id", ""))),
                "internal_object_id": str(payload.get("internal_object_id", payload.get("persistent_track_id", task.get("persistent_track_id", "")))),
                "local_segment_id": str(segment.get("local_segment_id", segment.get("semantic_segment_id", f"slot_{slot_id}"))),
                "semantic_segment_id": str(segment.get("semantic_segment_id", segment.get("local_segment_id", f"slot_{slot_id}"))),
                "hydra_slot_id": slot_id,
                "hydra_slot_name": str(segment.get("hydra_slot_name", segment.get("hydra_label_name", task.get("hydra_slot_name", payload.get("hydra_slot_name", ""))))),
                "frame_id": str(task.get("frame_id", "")),
                "sequence": int(task.get("sequence", 0) or 0),
                "timestamp_sec": float(task.get("timestamp_sec", payload.get("semantic_timestamp_sec", 0.0)) or 0.0),
                "label": label,
                "confidence": float(payload.get("semantic_label_confidence", 0.0) or 0.0),
                "label_confidence": float(payload.get("semantic_label_confidence", 0.0) or 0.0),
                "mobility_class": str(payload.get("mobility_class", "unknown") or "unknown"),
                "mobility_confidence": float(payload.get("mobility_confidence", 0.0) or 0.0),
                "mobility_source": str(payload.get("mobility_source", "none") or "none"),
                "crop_revision": int(task.get("crop_revision", 0) or 0),
                "vlm_crop_quality_score": float(task.get("vlm_crop_quality_score", 0.0) or 0.0),
                "vlm_crop_quality_eligible": bool(task.get("vlm_crop_quality_eligible", False)),
                "vlm_crop_quality_reasons": list(task.get("vlm_crop_quality_reasons", []) or []),
                "vlm_crop_quality_timeout_forced": bool(task.get("vlm_crop_quality_timeout_forced", False)),
                "vlm_failure_reason": str(task.get("vlm_failure_reason", "")),
                "vlm_raw_response": str(task.get("vlm_raw_response", ""))[:500],
                "centroid_3d": segment.get("centroid_3d", payload.get("centroid_3d", (task.get("object_metadata") or {}).get("centroid_3d"))),
                "centroid_frame_id": str(task.get("centroid_frame_id", "")),
            })
            self._safe_publish(self.semantic_label_result_pub, String(data=safe_json_dumps(segment_payload)))

        track_id = str(payload.get("persistent_track_id", ""))
        if finalize_track and track_id:
            with self._semantic_label_lock:
                self._semantic_label_pending_track_ids.discard(track_id)
            self._finalize_track_queue_state(track_id)
            self._retire_track_crop(track_id)

    def _publish_active_local_segments(self, frame: RsgFrame, track_records: List[Dict[str, Any]], timestamp_sec: float) -> None:
        """Publish the local Hydra slots observed in the current frame.

        The fuser uses these timestamps for presence confidence.  The message is
        keyed by semantic slot, not by internal object, so revisiting section B
        of a long object resets only section B.
        """
        if not track_records:
            return
        segments: List[Dict[str, Any]] = []
        seen_slots = set()
        for record in track_records:
            slot_id = int(record.get("hydra_slot_id", record.get("hydra_label_id", 0)) or 0)
            if slot_id <= 0 or slot_id in seen_slots:
                continue
            seen_slots.add(slot_id)
            all_segments = list(record.get("semantic_segments", []) or [])
            segment_slot_ids = []
            for segment in all_segments:
                try:
                    segment_slot = int(segment.get("hydra_slot_id", segment.get("hydra_label_id", 0)) or 0)
                except Exception:
                    segment_slot = 0
                if segment_slot > 0:
                    segment_slot_ids.append(segment_slot)
            segments.append({
                "persistent_track_id": str(record.get("persistent_track_id", "")),
                "internal_object_id": str(record.get("internal_object_id", record.get("persistent_track_id", ""))),
                "persistent_instance_id": int(record.get("persistent_instance_id", 0) or 0),
                "local_segment_id": str(record.get("local_segment_id", record.get("semantic_segment_id", f"slot_{slot_id}"))),
                "semantic_segment_id": str(record.get("semantic_segment_id", record.get("local_segment_id", f"slot_{slot_id}"))),
                "hydra_slot_id": slot_id,
                "hydra_slot_name": str(record.get("hydra_slot_name", record.get("hydra_label_name", ""))),
                "last_observed_timestamp_sec": float(record.get("last_seen_timestamp_sec", timestamp_sec) or timestamp_sec),
                "centroid_3d": record.get("centroid_3d"),
                "bbox_3d_min": record.get("local_segment_bbox_3d_min", record.get("last_bbox_3d_min", record.get("bbox_3d_min"))),
                "bbox_3d_max": record.get("local_segment_bbox_3d_max", record.get("last_bbox_3d_max", record.get("bbox_3d_max"))),
                "last_bbox_3d_min": record.get("local_segment_bbox_3d_min", record.get("last_bbox_3d_min")),
                "last_bbox_3d_max": record.get("local_segment_bbox_3d_max", record.get("last_bbox_3d_max")),
                "local_segment_xy_span_m": record.get("local_segment_xy_span_m"),
                "track_event": str(record.get("track_event", record.get("persistent_track_event", ""))),
                "local_segment_event": str(record.get("local_segment_event", "")),
                "local_segment_match_reason": str(record.get("local_segment_match_reason", "")),
                "local_segment_match_score": record.get("local_segment_match_score"),
                "canonical_label": str(record.get("canonical_label", "")),
                "semantic_label": str(record.get("semantic_label", "")),
                "semantic_label_source": str(record.get("semantic_label_source", "")),
                "semantic_label_confidence": float(record.get("semantic_label_confidence", 0.0) or 0.0),
                "mobility_class": str(record.get("mobility_class", "unknown") or "unknown"),
                "mobility_confidence": float(record.get("mobility_confidence", 0.0) or 0.0),
                "mobility_source": str(record.get("mobility_source", "none") or "none"),
                "semantic_timestamp_sec": record.get("semantic_timestamp_sec"),
                "first_seen_timestamp_sec": record.get("first_seen_timestamp_sec"),
                "last_seen_timestamp_sec": record.get("last_seen_timestamp_sec"),
                "persistent_track_seen_count": int(record.get("persistent_track_seen_count", 0) or 0),
                "labeling_status": str(record.get("labeling_status", "")),
                "labeling_completed": bool(record.get("labeling_completed", False)),
                "semantic_segments": all_segments,
                "all_semantic_segments": all_segments,
                "semantic_slot_ids": segment_slot_ids,
                "object_identity": {
                    "internal_object_id": str(record.get("internal_object_id", record.get("persistent_track_id", ""))),
                    "persistent_track_id": str(record.get("persistent_track_id", "")),
                    "persistent_instance_id": int(record.get("persistent_instance_id", 0) or 0),
                    "canonical_label": str(record.get("canonical_label", "")),
                    "semantic_label": str(record.get("semantic_label", "")),
                    "semantic_label_source": str(record.get("semantic_label_source", "")),
                    "semantic_label_confidence": float(record.get("semantic_label_confidence", 0.0) or 0.0),
                    "mobility_class": str(record.get("mobility_class", "unknown") or "unknown"),
                    "mobility_confidence": float(record.get("mobility_confidence", 0.0) or 0.0),
                    "mobility_source": str(record.get("mobility_source", "none") or "none"),
                    "semantic_slot_ids": segment_slot_ids,
                    "semantic_segment_count": len(segment_slot_ids),
                },
            })

            # If a resolved object later creates a new local slot, immediately
            # propagate the cached class label to that slot.  This avoids waiting
            # for another RAP/VLM job for a section of the same long object.
            if (
                str(record.get("local_segment_event", "")) == "new_segment"
                and bool(record.get("labeling_completed", False))
                and str(record.get("semantic_label", ""))
            ):
                task = {
                    "persistent_track_id": str(record.get("persistent_track_id", "")),
                    "hydra_slot_id": slot_id,
                    "hydra_slot_name": str(record.get("hydra_slot_name", record.get("hydra_label_name", ""))),
                    "frame_id": frame.rsg_frame_id,
                    "sequence": int(frame.sequence),
                    "timestamp_sec": float(timestamp_sec),
                    "object_metadata": record,
                }
                propagation_event = dict(record)
                propagation_event["semantic_segments"] = [{
                    "hydra_slot_id": slot_id,
                    "hydra_slot_name": str(record.get(
                        "hydra_slot_name",
                        record.get("hydra_label_name", ""),
                    )),
                    "local_segment_id": str(record.get(
                        "local_segment_id",
                        record.get("semantic_segment_id", f"slot_{slot_id}"),
                    )),
                    "semantic_segment_id": str(record.get(
                        "semantic_segment_id",
                        record.get("local_segment_id", f"slot_{slot_id}"),
                    )),
                    "centroid_3d": record.get("centroid_3d"),
                }]
                self._emit_semantic_label_result(
                    propagation_event,
                    task,
                    source="object_label_propagation",
                    finalize_track=False,
                )

        if not segments:
            return
        payload = {
            "event": "local_segment_observations",
            "frame_id": frame.rsg_frame_id,
            "sequence": int(frame.sequence),
            "timestamp_sec": float(timestamp_sec),
            "segments": segments,
        }
        self._safe_publish(self.active_segments_pub, String(data=safe_json_dumps(payload)))

    def _finalize_track_queue_state(self, track_id: str) -> None:
        """Release scheduler de-duplication after a final semantic outcome."""
        key = str(track_id)
        if not key:
            return
        with self._rap_task_lock:
            self._rap_task_keys.discard(key)
            self._rap_deferred_track_id_set.discard(key)
            self._rap_enqueued_monotonic.pop(key, None)
        with self._vlm_task_lock:
            self._vlm_task_keys.discard(key)
            self._vlm_deferred_track_id_set.discard(key)
            self._vlm_enqueued_monotonic.pop(key, None)
        with self._vlm_quality_deferred_lock:
            self._vlm_quality_deferred_track_ids.discard(key)
            self._vlm_quality_deferred_since_timestamp_sec.pop(key, None)
            self._vlm_quality_force_track_ids.discard(key)

    def _release_rap_task_key(self, task_or_track_id: Any) -> None:
        """Compatibility wrapper used by older error paths."""
        if isinstance(task_or_track_id, dict):
            key = str(task_or_track_id.get("persistent_track_id", ""))
        else:
            key = str(task_or_track_id or "")
        self._finalize_track_queue_state(key)

    def _pump_rap_deferred(self) -> None:
        """Move deferred RAP IDs into the bounded FIFO when capacity exists."""
        with self._rap_task_lock:
            while self._rap_deferred_track_ids:
                track_id = str(self._rap_deferred_track_ids[0])
                if track_id not in self._rap_task_keys or track_id not in self._rap_deferred_track_id_set:
                    self._rap_deferred_track_ids.popleft()
                    self._rap_deferred_track_id_set.discard(track_id)
                    continue
                try:
                    self.rap_queue.put_nowait(track_id)
                except queue.Full:
                    return
                self._rap_deferred_track_ids.popleft()
                self._rap_deferred_track_id_set.discard(track_id)

    def enqueue_rap_task(self, track_id: str) -> str:
        """Schedule one persistent track for RAP without queuing its crop.

        A bounded FIFO protects the worker, while the deferred registry preserves
        every unique unresolved track ID during temporary overload.
        """
        key = str(track_id)
        if not key:
            return "missing_track_id"
        with self._rap_task_lock:
            if key in self._rap_task_keys:
                return "rap_already_requested_for_track"
            self._rap_task_keys.add(key)
            self._rap_enqueued_monotonic[key] = time.perf_counter()
            try:
                self.rap_queue.put_nowait(key)
                return "queued_for_rap"
            except queue.Full:
                self._rap_deferred_track_ids.append(key)
                self._rap_deferred_track_id_set.add(key)
                self.rap_queue_deferred_count += 1
                return "deferred_for_rap"

    def _pump_vlm_deferred(self) -> None:
        """Move deferred VLM IDs into the bounded FIFO when capacity exists."""
        with self._vlm_task_lock:
            while self._vlm_deferred_track_ids:
                track_id = str(self._vlm_deferred_track_ids[0])
                if track_id not in self._vlm_task_keys or track_id not in self._vlm_deferred_track_id_set:
                    self._vlm_deferred_track_ids.popleft()
                    self._vlm_deferred_track_id_set.discard(track_id)
                    continue
                try:
                    self.vlm_queue.put_nowait(track_id)
                except queue.Full:
                    return
                self._vlm_deferred_track_ids.popleft()
                self._vlm_deferred_track_id_set.discard(track_id)

    def _is_vlm_crop_eligible(self, task: Dict[str, Any], track_id: str = "") -> bool:
        """Return whether a normal VLM request has a sufficiently useful crop."""
        key = str(track_id or task.get("persistent_track_id", ""))
        with self._vlm_quality_deferred_lock:
            if key and key in self._vlm_quality_force_track_ids:
                return True
        return bool(task.get("vlm_crop_quality_eligible", (task.get("object_metadata") or {}).get("vlm_crop_quality_eligible", True)))

    def _defer_vlm_for_better_crop(
        self,
        track_id: str,
        reason: str,
        *,
        current_timestamp_sec: Optional[float] = None,
    ) -> str:
        """Hold one RAP-unknown track for a better crop, with a bounded wait.

        The timer is measured in recorded bag time so its behaviour is stable
        across rosbag playback rates. The ID remains outside the VLM FIFO while
        the crop registry is still mutable; a later eligible crop resumes it
        immediately, otherwise the timeout releases the best available crop.
        """
        key = str(track_id)
        if not key:
            return "missing_track_id"
        defer_timestamp_sec = float(
            self._latest_processed_timestamp_sec
            if current_timestamp_sec is None
            else current_timestamp_sec
        )
        with self._vlm_task_lock:
            self._vlm_task_keys.discard(key)
            self._vlm_deferred_track_id_set.discard(key)
            self._vlm_enqueued_monotonic.pop(key, None)
        with self._vlm_quality_deferred_lock:
            self._vlm_quality_deferred_track_ids.add(key)
            self._vlm_quality_deferred_since_timestamp_sec.setdefault(key, defer_timestamp_sec)
        self.persistent_tracker.set_labeling_status(key, "vlm_waiting_for_better_crop")
        return str(reason)

    def _resume_quality_deferred_vlm_if_ready(self, track_id: str) -> None:
        """Queue a deferred track as soon as a stronger crop satisfies the VLM gate."""
        key = str(track_id)
        with self._vlm_quality_deferred_lock:
            if key not in self._vlm_quality_deferred_track_ids:
                return
        crop_state = self._describe_track_crop(key)
        if crop_state is None or not bool(crop_state.get("vlm_crop_quality_eligible", False)):
            return
        with self._vlm_quality_deferred_lock:
            self._vlm_quality_deferred_track_ids.discard(key)
            self._vlm_quality_deferred_since_timestamp_sec.pop(key, None)
        status = self.enqueue_vlm_track(key)
        if status in {"queued_for_vlm_fifo", "deferred_for_vlm"}:
            self.persistent_tracker.set_labeling_status(
                key,
                "vlm_queued_after_crop_quality" if status == "queued_for_vlm_fifo" else "vlm_deferred_after_crop_quality",
            )

    def _release_quality_deferred_vlm_if_expired(self, current_timestamp_sec: float) -> None:
        """Release weak crops after the bounded post-RAP collection interval.

        A track keeps accepting better crops until this method queues its ID.
        It is called once for each processed frame, so the deadline is driven by
        bag time even when no later observation of the deferred object arrives.
        """
        if not bool(self.config.vlm_crop_quality_force_on_timeout):
            return
        max_wait_sec = max(0.0, float(self.config.vlm_crop_quality_max_wait_sec))
        now_sec = float(current_timestamp_sec)
        due_track_ids: List[str] = []
        with self._vlm_quality_deferred_lock:
            for track_id in list(self._vlm_quality_deferred_track_ids):
                deferred_since_sec = float(
                    self._vlm_quality_deferred_since_timestamp_sec.get(track_id, now_sec)
                )
                if max(0.0, now_sec - deferred_since_sec) < max_wait_sec:
                    continue
                self._vlm_quality_deferred_track_ids.discard(track_id)
                self._vlm_quality_deferred_since_timestamp_sec.pop(track_id, None)
                self._vlm_quality_force_track_ids.add(track_id)
                due_track_ids.append(track_id)

        for track_id in due_track_ids:
            status = self.enqueue_vlm_track(track_id)
            if status in {"queued_for_vlm_fifo", "deferred_for_vlm"}:
                self.persistent_tracker.set_labeling_status(
                    track_id,
                    "vlm_queued_after_quality_timeout"
                    if status == "queued_for_vlm_fifo"
                    else "vlm_deferred_after_quality_timeout",
                )
                crop_state = self._describe_track_crop(track_id) or {}
                self.record_vlm_queue_event(
                    event="quality_timeout_force",
                    task={
                        "persistent_track_id": track_id,
                        "unknown_track_id": track_id,
                        "crop_revision": int(crop_state.get("crop_revision", 0) or 0),
                        "best_frame_score": float(crop_state.get("best_frame_score", 0.0) or 0.0),
                    },
                    queue_wait_ms=0.0,
                    reason="vlm_crop_quality_timeout",
                )

    def _force_release_quality_deferred_vlm_tracks(self) -> None:
        """Release deferred IDs at controlled shutdown with their best available crop."""
        with self._vlm_quality_deferred_lock:
            track_ids = list(self._vlm_quality_deferred_track_ids)
            self._vlm_quality_deferred_track_ids.clear()
            self._vlm_quality_deferred_since_timestamp_sec.clear()
            self._vlm_quality_force_track_ids.update(track_ids)
        for track_id in track_ids:
            status = self.enqueue_vlm_track(track_id)
            if status in {"queued_for_vlm_fifo", "deferred_for_vlm"}:
                self.persistent_tracker.set_labeling_status(track_id, "vlm_forced_low_quality_at_shutdown")

    def enqueue_vlm_track(self, track_id: str) -> str:
        """Schedule one unresolved persistent track for VLM by ID only."""
        key = str(track_id)
        if not key:
            return "missing_track_id"
        with self._vlm_task_lock:
            if key in self._vlm_task_keys:
                return "vlm_already_requested_for_track"
            self._vlm_task_keys.add(key)
            self._vlm_enqueued_monotonic[key] = time.perf_counter()
            try:
                self.vlm_queue.put_nowait(key)
                status = "queued_for_vlm_fifo"
            except queue.Full:
                self._vlm_deferred_track_ids.append(key)
                self._vlm_deferred_track_id_set.add(key)
                self.vlm_queue_deferred_count += 1
                status = "deferred_for_vlm"
        # The legacy unknown tracker may not own this persistent-track ID; this
        # call is harmless in that case and keeps fallback diagnostics coherent.
        self.unknown_tracker.mark_vlm_queued(key)
        self.unknown_vlm_count += 1
        return status

    @staticmethod
    def _vlm_schedule_accepted(status: str) -> bool:
        return str(status) in {
            "queued_for_vlm_fifo",
            "deferred_for_vlm",
            "deferred_for_better_crop",
            "vlm_already_requested_for_track",
        }

    def _finish_unknown_without_vlm(self, track_id: str, task: Dict[str, Any], reason: str) -> None:
        """Publish a terminal unknown state when no VLM work can run."""
        completed = self.persistent_tracker.complete_semantic_labeling(
            str(track_id), float(task.get("timestamp_sec", 0.0) or 0.0), str(reason)
        )
        if completed is not None:
            self._emit_semantic_label_result(completed, task, source="vlm_failed")

    def _rap_loop(self) -> None:
        """Run VisualRAP over the latest crop available at ID dequeue time."""
        while not self._stop_event.is_set():
            self._pump_rap_deferred()
            try:
                track_id = self.rap_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._pump_rap_deferred()
            task = self._snapshot_track_task(str(track_id), "rap_dequeue")
            if task is not None:
                self.persistent_tracker.set_labeling_status(str(track_id), "rap_dequeued")
            if task is None:
                fallback = {
                    "persistent_track_id": str(track_id),
                    "hydra_slot_id": 0,
                    "timestamp_sec": 0.0,
                    "object_metadata": {},
                }
                self.get_logger().warn(f"RAP track {track_id} has no active crop; finalizing as unknown.")
                self._publish_rap_result(fallback, label="unknown_object", confidence=0.0,
                                         is_known=False, status="rap_missing_crop", reason="no_active_crop")
                self._finish_unknown_without_vlm(str(track_id), fallback, "rap_missing_crop")
                continue
            try:
                self._process_rap_task(task)
            except Exception as exc:
                if rclpy.ok() and not self._stop_event.is_set():
                    self.get_logger().error(
                        f"Async RAP failed for slot={task.get('hydra_slot_id', 0)} "
                        f"track={task.get('persistent_track_id', '')}: {exc}"
                    )
                vlm_status = self._enqueue_vlm_after_rap(str(track_id))
                self._publish_rap_result(task, label="unknown_object", confidence=0.0,
                                         is_known=False, status="rap_error", reason=str(exc),
                                         vlm_dispatch_status=vlm_status)
                if not self._vlm_schedule_accepted(vlm_status):
                    self._finish_unknown_without_vlm(str(track_id), task, "rap_worker_error")

    def _process_rap_task(self, task: Dict[str, Any]) -> str:
        """Run RAP on a dequeue-time snapshot of one track's best crop."""
        start = time.perf_counter()
        crop = task.get("rgb_crop")
        if crop is None or getattr(crop, "size", 0) == 0:
            raise RuntimeError("Asynchronous RAP task has no representative crop")
        height, width = crop.shape[:2]
        synthetic_mask = SamMask(
            mask_id=str(task.get("candidate_id", "semantic_crop")),
            mask=np.ones((height, width), dtype=bool),
            bbox_2d=[0, 0, int(width), int(height)],
            area_px=int(height * width),
            crop=crop,
            score=1.0,
            metadata={"semantic_track_labeling": True, "crop_revision": task.get("crop_revision", 0)},
        )
        rap = self.rap_backend.classify(crop, synthetic_mask, 0)
        is_known = bool(rap.is_known and rap.confidence >= self.config.rap_confidence_threshold)
        label = str(rap.label or "unknown_object")
        rap_metadata = dict(rap.metadata or {})
        rap_has_mobility_metadata = "mobility_class" in rap_metadata
        mobility_class = str(rap_metadata.get("mobility_class", "unknown") or "unknown").strip().lower()
        if mobility_class not in {"static", "dynamic", "unknown"}:
            mobility_class = "unknown"
        try:
            stored_mobility_confidence = float(rap_metadata.get("mobility_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            stored_mobility_confidence = 0.0
        try:
            stored_label_confidence = float(rap_metadata.get("label_confidence", rap.confidence) or 0.0)
        except (TypeError, ValueError):
            stored_label_confidence = float(rap.confidence)
        label_confidence = min(
            max(0.0, min(1.0, stored_label_confidence)),
            max(0.0, min(1.0, float(rap.confidence))),
        )
        mobility_confidence = min(
            max(0.0, min(1.0, stored_mobility_confidence)),
            max(0.0, min(1.0, float(rap.confidence))),
        )
        mobility_source = "rap_memory" if mobility_class != "unknown" else "none"
        if is_known and mobility_class == "unknown" and not rap_has_mobility_metadata:
            mobility_class = infer_mobility_from_label(
                label,
                dynamic_label_hints=self.config.vlm_dynamic_label_hints,
                static_label_hints=self.config.vlm_static_label_hints,
            )
            if mobility_class != "unknown":
                mobility_confidence = max(0.0, min(1.0, float(rap.confidence)))
                mobility_source = "rap_label_hint"
        track_id = str(task.get("persistent_track_id", ""))
        persistent_update = None
        if self.config.persistent_tracking_enabled and track_id:
            persistent_update = self.persistent_tracker.apply_rap_result(
                track_id=track_id,
                label=label,
                confidence=float(label_confidence),
                is_known=is_known,
                mobility_class=mobility_class,
                mobility_confidence=mobility_confidence,
                mobility_source=mobility_source,
            )

        vlm_status = "not_requested"
        if is_known and track_id:
            completed = self.persistent_tracker.complete_semantic_labeling(
                track_id, float(task.get("timestamp_sec", 0.0) or 0.0), "rap_known"
            )
            if completed is not None:
                self._emit_semantic_label_result(completed, task, source="rap")
        elif not is_known:
            vlm_status = self._enqueue_vlm_after_rap(track_id)
            if not self._vlm_schedule_accepted(vlm_status):
                self._finish_unknown_without_vlm(track_id, task, "rap_unknown_vlm_unavailable")

        self.rap_completed_count += 1
        self._publish_rap_result(
            task,
            label=label,
            confidence=float(rap.confidence),
            label_confidence=float(label_confidence),
            is_known=is_known,
            status="known" if is_known else "unknown",
            reason="async_retrieval_complete",
            rap_metadata=rap_metadata,
            mobility_class=mobility_class,
            mobility_confidence=mobility_confidence,
            mobility_source=mobility_source,
            persistent_update=persistent_update,
            vlm_dispatch_status=vlm_status,
            rap_delay_ms=(time.perf_counter() - start) * 1000.0,
        )
        return "completed"

    def _enqueue_vlm_after_rap(self, track_id: str) -> str:
        """Queue an unresolved track only when its current VLM crop is useful.

        The track ID, rather than an image payload, remains the queued unit.
        Weak fragments are retained for later observations and resume as soon as
        the shared best crop reaches the quality threshold.
        """
        if not self.config.vlm_enabled:
            return "vlm_disabled"
        key = str(track_id)
        crop_state = self._describe_track_crop(key)
        if crop_state is None:
            return "vlm_missing_crop"
        if not bool(crop_state.get("vlm_crop_quality_eligible", False)):
            return self._defer_vlm_for_better_crop(
                key,
                "deferred_for_better_crop",
                current_timestamp_sec=float(self._latest_processed_timestamp_sec),
            )
        status = self.enqueue_vlm_track(key)
        if status in {"queued_for_vlm_fifo", "deferred_for_vlm"}:
            self.persistent_tracker.set_labeling_status(
                key,
                "vlm_queued" if status == "queued_for_vlm_fifo" else "vlm_deferred",
            )
        return status

    def _dispatch_vlm_from_rap_task(self, task: Dict[str, Any]) -> str:
        """Compatibility path used only when the old unknown tracker is active."""
        if not self.config.vlm_enabled:
            return "vlm_disabled"
        unknown = dict(task.get("unknown_metadata") or {})
        if not unknown:
            return "missing_unknown_metadata"
        vlm_task, dispatch_info = self.unknown_tracker.update_evidence_and_build_vlm_task(
            unknown=unknown,
            rgb_crop=task.get("rgb_crop"),
            frame_header=task.get("frame_header"),
            frame_id=str(task.get("frame_id", "")),
            sequence=int(task.get("sequence", 0)),
            image_area_px=int(task["rgb"].shape[0] * task["rgb"].shape[1]) if getattr(task.get("rgb"), "ndim", 0) >= 2 else None,
        )
        status = str(dispatch_info.get("vlm_dispatch_status", "not_queued"))
        if vlm_task is not None:
            status = self.enqueue_vlm_task(vlm_task, status)
        return status

    def _publish_rap_result(
        self,
        task: Dict[str, Any],
        *,
        label: str,
        confidence: float,
        label_confidence: Optional[float] = None,
        is_known: bool,
        status: str,
        reason: str,
        rap_metadata: Optional[Dict[str, Any]] = None,
        mobility_class: str = "unknown",
        mobility_confidence: float = 0.0,
        mobility_source: str = "none",
        persistent_update: Optional[Dict[str, Any]] = None,
        vlm_dispatch_status: str = "",
        rap_delay_ms: float = 0.0,
    ) -> None:
        """Publish the asynchronous RAP contract: stable slot ID and label."""
        payload = {
            "event": "rap_result",
            "status": str(status),
            "reason": str(reason),
            "persistent_track_id": str(task.get("persistent_track_id", "")),
            "hydra_slot_id": int(task.get("hydra_slot_id", 0) or 0),
            "hydra_slot_name": str(task.get("hydra_slot_name", "")),
            "candidate_id": str(task.get("candidate_id", "")),
            "frame_id": str(task.get("frame_id", "")),
            "sequence": int(task.get("sequence", 0) or 0),
            "timestamp_sec": float(task.get("timestamp_sec", 0.0) or 0.0),
            "label": str(label),
            "confidence": float(confidence),
            "label_confidence": float(confidence if label_confidence is None else label_confidence),
            "retrieval_confidence": float(confidence),
            "mobility_class": str(mobility_class),
            "mobility_confidence": float(mobility_confidence),
            "mobility_source": str(mobility_source),
            "is_known": bool(is_known),
            "rap_delay_ms": float(rap_delay_ms),
            "vlm_dispatch_status": str(vlm_dispatch_status),
            "rap_metadata": rap_metadata or {},
            "persistent_update": persistent_update or {},
        }
        self._safe_publish(self.rap_result_pub, String(data=safe_json_dumps(payload)))

    def dispatch_unknowns_to_vlm(
        self,
        frame: RsgFrame,
        rgb: np.ndarray,
        depth: np.ndarray,
        unknowns: List[Dict[str, Any]],
        classified: List[ClassifiedMask],
    ) -> List[Dict[str, Any]]:
        """Queue one VLM request per persistent unknown track.

        A frame may contain several unknown detections, and the same physical
        unknown may appear in many consecutive frames. The tracker assigns the
        persistent ``unknown_track_id`` and this function only queues the VLM
        task when the track is ready and has not already been queued/done.
        """
        dispatch_records: List[Dict[str, Any]] = []
        mask_lookup = {item.candidate_id: item for item in classified}
        image_area_px = int(rgb.shape[0] * rgb.shape[1]) if rgb.ndim >= 2 else None

        for unknown in unknowns:
            candidate_id = str(unknown.get("candidate_id", ""))
            classified_mask = mask_lookup.get(candidate_id)
            bbox_2d = unknown.get("bbox_2d")
            _, context_bbox_2d = self.extract_crop_with_context(
                rgb,
                bbox_2d,
                context_ratio=float(self.config.vlm_crop_context_ratio),
            )
            rgb_crop = self._build_vlm_semantic_crop(
                rgb,
                None if classified_mask is None else classified_mask.mask,
                context_bbox_2d,
            )

            if not self.config.vlm_enabled:
                dispatch_info = {"vlm_dispatch_status": "vlm_disabled", "best_frame_score": 0.0}
                task = None
            else:
                task, dispatch_info = self.unknown_tracker.update_evidence_and_build_vlm_task(
                    unknown=unknown,
                    rgb_crop=rgb_crop,
                    frame_header=frame.header,
                    frame_id=frame.rsg_frame_id,
                    sequence=int(frame.sequence),
                    image_area_px=image_area_px,
                )

            dispatch_status = str(dispatch_info.get("vlm_dispatch_status", "not_queued"))
            if task is not None:
                dispatch_status = self.enqueue_vlm_task(task, dispatch_status)

            record = {
                "candidate_id": candidate_id,
                "unknown_track_id": str(unknown.get("unknown_track_id", "")),
                "status": dispatch_status,
                "track_seen_count": int(unknown.get("track_seen_count", 1) or 1),
                "best_frame_score": float(dispatch_info.get("best_frame_score", 0.0) or 0.0),
            }
            dispatch_records.append(record)

        return dispatch_records

    def enqueue_vlm_task(self, task: Dict[str, Any], previous_status: str) -> str:
        """Compatibility enqueue path for RAP-disabled legacy unknown tracking.

        Normal RAP-enabled operation uses :meth:`enqueue_vlm_track` and stores
        only a track ID.  This path remains bounded for compatibility with older
        launch configurations that bypass RAP.
        """
        track_id = str(task.get("unknown_track_id", ""))
        task["enqueued_monotonic"] = time.perf_counter()
        try:
            self.vlm_queue.put_nowait(task)
            self.unknown_tracker.mark_vlm_queued(track_id)
            self.unknown_vlm_count += 1
            self.record_vlm_queue_event(event="enqueued", task=task, queue_wait_ms=0.0, reason=previous_status)
            return "queued_for_vlm_fifo"
        except queue.Full:
            # Preserve the old non-RAP fallback behavior. The requested
            # no-drop guarantee applies to the RAP-enabled ID-only scheduler.
            self.vlm_queue_dropped_count += 1
            self.unknown_tracker.mark_vlm_queue_rejected(track_id, reason="vlm_fifo_queue_full")
            self.record_vlm_queue_event(event="dropped_queue_full", task=task, queue_wait_ms=0.0, reason=self.config.vlm_queue_drop_policy)
            return "vlm_fifo_queue_full_dropped"

    def record_vlm_queue_event(self, event: str, task: Dict[str, Any], queue_wait_ms: float = 0.0, reason: str = "") -> None:
        """Record VLM queue latency when timing diagnostics are enabled."""
        if not self.config.timing_enabled:
            return
        self.timing_recorder.add_sample(
            node="rsg_object_detection",
            event=event,
            sequence=int(task.get("sequence", 0) or 0),
            frame_id=str(task.get("rsg_frame_id", "")),
            unknown_track_id=str(task.get("unknown_track_id", "")),
            candidate_id=str(task.get("candidate_id", "")),
            queue_size=int(self.vlm_queue.qsize()),
            queue_max_size=int(self.config.vlm_queue_size),
            queue_wait_ms=float(queue_wait_ms),
            track_seen_count=int(task.get("track_seen_count", 0) or 0),
            best_frame_score=float(task.get("best_frame_score", 0.0) or 0.0),
            reason=reason,
        )

    def _vlm_loop(self) -> None:
        """Run VLM on the latest crop available when its track ID is dequeued."""
        while not self._stop_event.is_set():
            self._pump_vlm_deferred()
            try:
                queued_item = self.vlm_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._pump_vlm_deferred()

            is_track_id_task = isinstance(queued_item, str)
            track_id = str(queued_item) if is_track_id_task else str(queued_item.get("unknown_track_id", queued_item.get("persistent_track_id", "")))
            if is_track_id_task:
                task = self._snapshot_track_task(track_id, "vlm_dequeue")
                if task is not None:
                    self.persistent_tracker.set_labeling_status(track_id, "vlm_dequeued")
                if task is None:
                    fallback = {
                        "persistent_track_id": track_id,
                        "unknown_track_id": track_id,
                        "hydra_slot_id": 0,
                        "timestamp_sec": 0.0,
                        "object_metadata": {},
                    }
                    self.get_logger().warn(f"VLM track {track_id} has no active crop; finalizing as unknown.")
                    self._finish_unknown_without_vlm(track_id, fallback, "vlm_missing_crop")
                    continue
                task["unknown_track_id"] = track_id
                if not self._is_vlm_crop_eligible(task, track_id):
                    self._defer_vlm_for_better_crop(
                        track_id,
                        "deferred_for_better_crop",
                        current_timestamp_sec=float(self._latest_processed_timestamp_sec),
                    )
                    self.record_vlm_queue_event(
                        event="quality_deferred",
                        task=task,
                        queue_wait_ms=0.0,
                        reason="vlm_crop_quality_gate",
                    )
                    continue
            else:
                # Compatibility only for RAP-disabled legacy dispatch.
                task = dict(queued_item)
                track_id = str(task.get("unknown_track_id", task.get("persistent_track_id", "")))

            start = time.perf_counter()
            queue_wait_ms = max(0.0, (start - float(task.get("created_monotonic", start))) * 1000.0)
            self.record_vlm_queue_event(event="dequeued", task=task, queue_wait_ms=queue_wait_ms, reason="fifo_order")
            try:
                result = self.vlm_backend.identify(
                    task.get("vlm_rgb_crop", task.get("rgb_crop")),
                    task.get("object_metadata", {}),
                )
            except Exception as exc:
                # VLM availability is not allowed to strand a unique slot in
                # the pending state.  Publish an explicit unknown-object result
                # and allow the fuser to retain the stable physical slot.
                result = {
                    "success": False,
                    "label": "unknown_object",
                    "confidence": 0.0,
                    "label_confidence": 0.0,
                    "mobility_class": "unknown",
                    "mobility_confidence": 0.0,
                    "backend": self.config.vlm_mode,
                    "model": self.config.vlm_model,
                    "raw_response": f"vlm_error: {exc}",
                    "failure_reason": f"worker_exception:{type(exc).__name__}",
                    "validation_status": "rejected",
                    "validation_reason": f"worker_exception:{type(exc).__name__}",
                }
                if rclpy.ok() and not self._stop_event.is_set():
                    self.get_logger().error(f"VLM failed for track={track_id}: {exc}")

            # The VLM label must be paired with the exact immutable crop that
            # was supplied at VLM dequeue.  Do not replace it with a later crop
            # observed during inference: that later crop was not classified by
            # this request and could contaminate RAP memory with a mismatched
            # label/image pair.
            memory_task = task
            memory_metadata = dict(memory_task.get("object_metadata") or {})
            memory_metadata.update({
                "rsg_slot_id": int(memory_task.get("hydra_slot_id", memory_metadata.get("hydra_label_id", 0)) or 0),
                "hydra_slot_id": int(memory_task.get("hydra_slot_id", memory_metadata.get("hydra_label_id", 0)) or 0),
                "persistent_track_id": track_id,
                "crop_revision": int(memory_task.get("crop_revision", 0) or 0),
                "crop_score": float(memory_task.get("best_frame_score", 0.0) or 0.0),
                "memory_crop_stage": "best_available_when_vlm_completed",
                "label_confidence": float(result.get("label_confidence", result.get("confidence", 0.0)) or 0.0),
                "mobility_class": str(result.get("mobility_class", "unknown") or "unknown"),
                "mobility_confidence": float(result.get("mobility_confidence", 0.0) or 0.0),
                "mobility_source": "vlm",
            })
            rap_update_status = self.rap_memory_updater.update_from_vlm(result, memory_metadata)
            try:
                if self.config.rap_update_enabled and bool(result.get("success", False)) and float(result.get("confidence", 0.0)) >= float(self.config.rap_update_min_confidence):
                    label_for_rap = str(result.get("label", "")).replace("_", " ").strip()
                    memory_crop = memory_task.get("rgb_crop")
                    if label_for_rap and memory_crop is not None and hasattr(self.rap_backend, "add_image"):
                        self.rap_backend.add_image(memory_crop, label_for_rap, metadata={
                            "rsg_slot_id": memory_metadata["rsg_slot_id"],
                            "hydra_slot_id": memory_metadata["hydra_slot_id"],
                            "persistent_track_id": track_id,
                            "crop_revision": memory_metadata["crop_revision"],
                            "source": "vlm_label_with_rap_target_only_crop_at_dequeue",
                            "label_confidence": memory_metadata["label_confidence"],
                            "mobility_class": memory_metadata["mobility_class"],
                            "mobility_confidence": memory_metadata["mobility_confidence"],
                            "mobility_source": "vlm",
                        })
                        rap_update_status["rap_memory_live_update"] = "added_to_visual_rap"
                        rap_update_status["memory_slot_id"] = memory_metadata["rsg_slot_id"]
                        rap_update_status["memory_crop_revision"] = memory_metadata["crop_revision"]
            except Exception as exc:
                rap_update_status["rap_memory_live_update"] = "failed"
                rap_update_status["live_update_error"] = str(exc)
            result["rap_update"] = rap_update_status
            result["memory_slot_id"] = memory_metadata.get("rsg_slot_id", 0)
            result["memory_crop_revision"] = memory_metadata.get("crop_revision", 0)

            vlm_delay_ms = (time.perf_counter() - start) * 1000.0
            total_age_ms = (time.perf_counter() - float(task.get("created_monotonic", start))) * 1000.0
            msg = Phase1VlmResult()
            msg.header = task["frame_header"]
            msg.rsg_frame_id = str(task["rsg_frame_id"])
            msg.sequence = int(task["sequence"])
            msg.candidate_id = str(task["candidate_id"])
            msg.unknown_track_id = track_id
            msg.mask_id = str(task["mask_id"])
            msg.success = bool(result.get("success", False))
            msg.status = "vlm_done" if msg.success else "vlm_failed"
            msg.reason = "ok" if msg.success else str(result.get("raw_response", "vlm_failed"))
            msg.predicted_label = str(result.get("label", "unknown_object")) if msg.success else "unknown_object"
            msg.confidence = float(result.get("confidence", 0.0)) if msg.success else 0.0
            msg.label_confidence = float(result.get("label_confidence", msg.confidence)) if msg.success else 0.0
            msg.mobility_class = str(result.get("mobility_class", "unknown")) if msg.success else "unknown"
            msg.mobility_confidence = float(result.get("mobility_confidence", 0.0)) if msg.success else 0.0
            msg.backend = str(result.get("backend", self.config.vlm_mode))
            msg.model = str(result.get("model", self.config.vlm_model))
            msg.vlm_delay_ms = float(vlm_delay_ms)
            msg.total_age_ms = float(total_age_ms)
            msg.object_metadata_json = safe_json_dumps(memory_metadata)
            result["unknown_track_id"] = msg.unknown_track_id
            result["track_seen_count"] = int(memory_task.get("track_seen_count", task.get("track_seen_count", 0)) or 0)
            result["best_frame_score"] = float(memory_task.get("best_frame_score", task.get("best_frame_score", 0.0)) or 0.0)
            memory_task["vlm_failure_reason"] = str(result.get("failure_reason", ""))
            memory_task["vlm_raw_response"] = str(result.get("raw_response", ""))
            memory_task["mobility_class"] = msg.mobility_class
            memory_task["mobility_confidence"] = float(msg.mobility_confidence)
            memory_task["mobility_source"] = "vlm" if msg.success else "none"
            msg.vlm_metadata_json = safe_json_dumps(result)
            self.unknown_tracker.mark_vlm_result(msg.unknown_track_id, result)
            if self.config.persistent_tracking_enabled:
                persistent_update = None
                if msg.success:
                    persistent_update = self.persistent_tracker.apply_vlm_result(
                        msg.unknown_track_id,
                        msg.predicted_label,
                        msg.confidence,
                        msg.mobility_class,
                        msg.mobility_confidence,
                    )
                semantic_task = memory_task if is_track_id_task else dict(task.get("semantic_label_task") or task)
                completed = self.persistent_tracker.complete_semantic_labeling(
                    msg.unknown_track_id,
                    float(semantic_task.get("timestamp_sec", 0.0) or 0.0),
                    "vlm_known" if msg.success else "vlm_failed",
                )
                if completed is not None:
                    self._emit_semantic_label_result(completed, semantic_task, source="vlm" if msg.success else "vlm_failed")
                    persistent_update = completed
                if persistent_update is not None:
                    result["persistent_track_update"] = persistent_update
                    msg.vlm_metadata_json = safe_json_dumps(result)
            self._safe_publish(self.vlm_result_pub, msg)
            self.publish_vlm_timing(msg)
            self.record_vlm_queue_event(event="completed", task=task, queue_wait_ms=queue_wait_ms, reason=msg.status)

    @staticmethod
    def extract_crop(rgb: np.ndarray, bbox_2d: Any) -> Optional[np.ndarray]:
        """Extract a tight RGB crop from [x, y, w, h] metadata."""
        crop, _ = Phase1SemanticCoordinator.extract_crop_with_context(rgb, bbox_2d, context_ratio=0.0)
        return crop

    @staticmethod
    def extract_crop_with_context(
        rgb: np.ndarray,
        bbox_2d: Any,
        *,
        context_ratio: float,
    ) -> Tuple[Optional[np.ndarray], List[int]]:
        """Extract a clipped crop with symmetric context around an object box."""
        context_bbox = context_bbox_xywh(
            rgb.shape[:2], bbox_2d, context_ratio=context_ratio
        )
        if not context_bbox:
            return None, []
        x, y, width, height = context_bbox
        return rgb[y:y + height, x:x + width].copy(), context_bbox

    @staticmethod
    def make_candidate_id(frame: RsgFrame, mask_id: str, index: int, is_known: bool) -> str:
        """Build a reproducible identifier for one mask candidate."""
        prefix = "rsg_known" if is_known else "rsg_unknown"
        safe_frame_id = frame.rsg_frame_id.replace("/", "_")
        return f"{prefix}_{safe_frame_id}_{index:03d}_{mask_id}"

    def build_result_metadata(
        self,
        frame: RsgFrame,
        masks: List[SamMask],
        objects: List[Dict[str, Any]],
        unknowns: List[Dict[str, Any]],
        vlm_dispatch: List[Dict[str, Any]],
        track_records: List[Dict[str, Any]],
        semantic_dispatches: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build compact metadata for one processed RSG frame."""
        queued = [item for item in vlm_dispatch if str(item.get("status", "")).startswith("queued_for_vlm")]
        metadata = {
            "phase": "phase1_object_classification",
            "node": "rsg_object_detection",
            "sam_backend": self.config.sam_backend,
            "rap_backend": self.config.rap_backend,
            "rap_async": self.config.rap_async,
            "rap_result_topic": self.config.rap_result_topic,
            "rap_fifo_queue_size": self.rap_queue.qsize(),
            "rap_fifo_queue_max_size": self.config.rap_queue_size,
            "rap_queue_dropped": self.rap_queue_dropped_count,
            "rap_queue_deferred_total": self.rap_queue_deferred_count,
            "rap_deferred_pending": len(self._rap_deferred_track_id_set),
            "rap_completed": self.rap_completed_count,
            "vlm_enabled": self.config.vlm_enabled,
            "vlm_mode": self.config.vlm_mode,
            "num_masks": len(masks),
            "num_objects": len(objects),
            "num_unknown": len(unknowns),
            "num_unknown_tracks": len({str(item.get("unknown_track_id", "")) for item in unknowns if item.get("unknown_track_id")}),
            "num_new_tracks": len([item for item in track_records if item.get("track_event") == "new_track"]),
            "num_matched_tracks": len([item for item in track_records if item.get("track_event") == "matched_existing_track"]),
            "num_vlm_queued": len(queued),
            "vlm_dispatch": vlm_dispatch,
            "unknown_track_records": track_records,
            "semantic_label_dispatches": semantic_dispatches or [],
            "source_preprocessor_metadata": safe_json_loads(frame.metadata_json, default={}),
        }
        return metadata

    def build_failed_result(self, frame: RsgFrame, reason: str) -> Phase1ClassificationResult:
        """Build a failure result with empty label maps so downstream nodes can log it."""
        result = Phase1ClassificationResult()
        result.header = frame.header
        result.rsg_frame_id = frame.rsg_frame_id
        result.sequence = frame.sequence
        result.success = False
        result.status = "failed"
        result.reason = reason
        height = int(frame.rgb.height) if frame.rgb.height else 1
        width = int(frame.rgb.width) if frame.rgb.width else 1
        empty = np.zeros((height, width), dtype=np.uint16)
        result.semantic_labels = self.bridge.cv2_to_imgmsg(empty, encoding=self.config.semantic_label_encoding)
        result.semantic_labels.header = frame.header
        result.instance_labels = self.bridge.cv2_to_imgmsg(empty, encoding=self.config.instance_label_encoding)
        result.instance_labels.header = frame.header
        result.label_table_json = safe_json_dumps({"0": "background"})
        result.object_metadata_json = "[]"
        result.unknown_candidates_json = "[]"
        result.vlm_dispatch_json = "[]"
        result.metadata_json = safe_json_dumps({"phase": "phase1_object_classification", "status": "failed", "reason": reason})
        result.image_conversion_delay_ms = 0.0
        result.result_message_build_delay_ms = 0.0
        result.classifier_debug_record_delay_ms = 0.0
        return result

    def publish_timing_event(self, result: Phase1ClassificationResult, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record one simple classifier phase-latency row per processed frame."""
        if not self.config.timing_enabled:
            return
        metadata = metadata or {}
        stage_ms = metadata.get("diagnostic_stage_ms", {}) or {}
        if self.timing_pub is not None:
            msg = Float64MultiArray()
            msg.data = [
                float(result.sequence),
                float(result.classifier_delay_ms),
                float(result.sam_delay_ms),
                float(result.rap_delay_ms),
                float(result.num_unknown),
            ]
            self._safe_publish(self.timing_pub, msg)

        self.timing_recorder.add_sample(
            node="rsg_object_detection",
            sequence=int(result.sequence),
            frame_id=result.rsg_frame_id,
            status=result.status,
            reason=result.reason,
            input_age_ms=float(result.input_age_ms),
            classifier_delay_ms=float(result.classifier_delay_ms),
            sam_delay_ms=float(result.sam_delay_ms),
            sam_prepare_ms=float(stage_ms.get("sam_prepare_ms", 0.0)),
            sam_inference_ms=float(stage_ms.get("sam_inference_ms", 0.0)),
            sam_restore_ms=float(stage_ms.get("sam_restore_ms", 0.0)),
            rap_delay_ms=float(result.rap_delay_ms),
            geometry_metadata_ms=float(stage_ms.get("geometry_metadata_ms", 0.0)),
            frame_assignment_ms=float(stage_ms.get("frame_assignment_ms", 0.0)),
            track_association_ms=float(stage_ms.get("track_association_ms", 0.0)),
            crop_update_ms=float(stage_ms.get("crop_update_ms", 0.0)),
            label_map_delay_ms=float(result.label_map_delay_ms),
            metadata_delay_ms=float(result.metadata_delay_ms),
            image_conversion_delay_ms=float(getattr(result, "image_conversion_delay_ms", 0.0)),
            result_message_build_delay_ms=float(getattr(result, "result_message_build_delay_ms", 0.0)),
            classifier_debug_record_delay_ms=float(getattr(result, "classifier_debug_record_delay_ms", 0.0)),
            num_masks=int(result.num_masks),
            num_known=int(result.num_known),
            num_unknown=int(result.num_unknown),
            num_new_tracks=int(metadata.get("num_new_tracks", 0)),
            num_matched_tracks=int(metadata.get("num_matched_tracks", 0)),
            num_vlm_queued=int(metadata.get("num_vlm_queued", 0)),
        )

    def publish_vlm_timing(self, result: Phase1VlmResult) -> None:
        """Record one simple VLM-latency row per persistent unknown track."""
        if not self.config.timing_enabled:
            return
        vlm_meta = safe_json_loads(result.vlm_metadata_json, default={})
        self.timing_recorder.add_sample(
            node="rsg_object_detection",
            sequence=int(result.sequence),
            frame_id=result.rsg_frame_id,
            status=result.status,
            reason=result.reason,
            candidate_id=result.candidate_id,
            unknown_track_id=result.unknown_track_id,
            predicted_label=result.predicted_label,
            label_confidence=float(result.label_confidence),
            mobility_class=result.mobility_class,
            mobility_confidence=float(result.mobility_confidence),
            vlm_delay_ms=float(result.vlm_delay_ms),
            total_age_ms=float(result.total_age_ms),
            track_seen_count=int(vlm_meta.get("track_seen_count", 0) or 0),
            best_frame_score=float(vlm_meta.get("best_frame_score", 0.0) or 0.0),
            backend=result.backend,
            model=result.model,
        )


    def publish_status(self, status: str, frame_id: str, reason: str) -> None:
        """Publish queue depth, worker progress, and frame throughput as JSON."""
        if not self.config.publish_status:
            return
        payload = {
            "node": "rsg_object_detection",
            "status": status,
            "reason": reason,
            "frame_id": frame_id,
            "received": self.received_count,
            "processed": self.processed_count,
            "failed": self.failed_count,
            "dropped": self.dropped_count,
            "hydra_published": self.hydra_published_count,
            "frame_fifo_size": self.frame_fifo.qsize(),
            "frame_fifo_max_size": self.config.request_queue_size,
            "sam_output_fifo_size": self.sam_output_fifo.qsize(),
            "sam_output_fifo_max_size": 1,
            "sam_output_dropped": self.sam_output_dropped_count,
            "rap_queue_dropped": self.rap_queue_dropped_count,
            "rap_queue_deferred_total": self.rap_queue_deferred_count,
            "rap_deferred_pending": len(self._rap_deferred_track_id_set),
            "rap_completed": self.rap_completed_count,
            "rap_fifo_queue_size": self.rap_queue.qsize(),
            "rap_fifo_queue_max_size": self.config.rap_queue_size,
            "vlm_queued": self.unknown_vlm_count,
            "vlm_queue_dropped": self.vlm_queue_dropped_count,
            "vlm_queue_deferred_total": self.vlm_queue_deferred_count,
            "vlm_deferred_pending": len(self._vlm_deferred_track_id_set),
            "vlm_quality_deferred_pending": len(self._vlm_quality_deferred_track_ids),
            "vlm_fifo_queue_size": self.vlm_queue.qsize(),
            "vlm_fifo_queue_max_size": self.config.vlm_queue_size,
        }
        self._safe_publish(self.status_pub, String(data=safe_json_dumps(payload)))

    def destroy_node(self) -> bool:
        """Stop intake and persist diagnostics inside launch's SIGINT window."""
        # Stop new work first. Background loops and semantic fan-out inspect
        # this event and return without producing more slot-level messages.
        self._stop_event.set()

        try:
            self.timing_recorder.save()
        except Exception as exc:
            print(f"Failed to save Phase 1 timing CSV: {exc}", flush=True)

        # Do not wait for a long HTTP timeout. Threads are daemon threads and
        # will terminate with the process after a brief cooperative join.
        for thread in (
            self._segmentation_thread,
            self._tracking_publish_thread,
            self._rap_thread,
            self._vlm_thread,
        ):
            try:
                if thread.is_alive():
                    thread.join(timeout=0.20)
            except Exception:
                pass

        return super().destroy_node()


def main(args: Optional[list[str]] = None) -> None:
    """Start the semantic labelling node and release resources on shutdown."""
    rclpy.init(args=args)
    node = Phase1SemanticCoordinator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
