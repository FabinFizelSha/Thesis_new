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

import os
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64MultiArray, String
from tf2_ros import TransformBroadcaster

from rsg.msg import Phase1ClassificationResult, Phase1VlmResult, RsgFrame, RsgHydraFrame

from nodes.support.phase1.backends import SamMask, make_rap_backend, make_risk_vlm_backend, make_sam_backend, make_vlm_backend
from nodes.support.phase1.bbox_diagnostics import BboxDiagnosticsLogger
from nodes.support.phase1.crop_evolution_tracker import CropEvolutionTracker
from nodes.support.phase1.frame_cache import BoundedFrameCache, CachedFrame, EvidenceBuffer
from nodes.support.phase1.tracker_state_store import load_tracker_state, save_tracker_state
from nodes.support.phase1.tracking_quality_recorder import TrackingQualityRecorder
from nodes.support.phase1.tracking_crop_manager import TrackingCropManager
from nodes.support.workspace_paths import workspace_path
from nodes.support.phase1.json_utils import safe_json_dumps, safe_json_loads
from nodes.support.phase1.label_map_builder import ClassifiedMask, LabelMapBuilder
from nodes.support.phase1.object_geometry import ObjectGeometryEstimator, filter_metadata
from nodes.support.phase1.phase1_config import Phase1Config
from nodes.support.phase1.phase1_timing_recorder import Phase1TimingRecorder
from nodes.support.phase1.persistent_object_tracker import PersistentObjectTracker
from nodes.support.phase1.vlm_test_diagnostics import VLMTestDiagnostics
from nodes.support.phase1.rap_memory import RapMemoryUpdater
from nodes.support.phase1.risk_vlm_diagnostics import RiskVlmDiagnostics
from nodes.support.phase1.rap_accuracy_diagnostics import RapAccuracyDiagnostics
from nodes.support.phase1.periodic_crop_diagnostics import PeriodicCropDiagnostics
from nodes.support.phase1.frame_mask_overlay_diagnostics import FrameMaskOverlayDiagnostics
from nodes.support.phase1.time_utils import stamp_to_float
from nodes.support.phase1.unknown_tracker import UnknownObjectTracker
from nodes.support.phase1.loop_closure import loop_closure_delta, quat_to_rot
from nodes.phase1_pipeline import (
    SegmentationStage,
    TrackingStage,
    SemanticsStage,
    PublishingStage,
    RiskVlmDispatchStage,
    RapDispatchStage,
    VlmDispatchStage,
    LocalSegmentPresenceStage,
    TrackCropRegistry,
    SemanticLabelDispatchStage,
)
from nodes.phase1_pipeline.crop_utils import make_candidate_id


class Phase1SemanticCoordinator(Node):
    """Coordinate Phase 1 segmentation, tracking, RAP retrieval, and VLM fallback.

    The class is intentionally a single ROS 2 node. SAM, RAP, and VLM workers
    remain internal threads so RGB-D crops do not cross ROS process boundaries.
    Final slot-to-label events are published only after RAP or VLM reaches a
    terminal decision.
    """


    def __init__(self) -> None:
        super().__init__("rsg_phase1_semantic_coordinator")

        # Clear Hydra cache on startup for fresh session (no pre-existing maps)
        try:
            import shutil
            hydra_cache = os.path.expanduser("~/.hydra/uhumans2")
            if os.path.exists(hydra_cache):
                shutil.rmtree(hydra_cache)
                self.get_logger().info(f"Cleared Hydra cache at startup: {hydra_cache}")
        except Exception as exc:
            self.get_logger().warn(f"Failed to clear Hydra cache at startup: {exc}")

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
        # Held locally, then handed to RapDispatchStage/VlmDispatchStage
        # below once their diagnostics writers exist.
        _rap_backend = make_rap_backend(self.config, self.get_logger())
        _vlm_backend = make_vlm_backend(self.config)
        # Deliberately a separate backend instance/model from vlm_backend --
        # risk assessment always runs against its own configured
        # endpoint/model, never the object-detection VLM's. Held locally,
        # then handed to RiskVlmDispatchStage below once its diagnostics
        # writer exists.
        _risk_vlm_backend = make_risk_vlm_backend(self.config)
        self.rap_memory_updater = RapMemoryUpdater(
            enabled=self.config.rap_update_enabled,
            output_path=self.config.rap_memory_path,
            min_confidence=self.config.rap_update_min_confidence,
            logger=self.get_logger(),
        )
        self.geometry_estimator = ObjectGeometryEstimator(self.config)
        self.label_map_builder = LabelMapBuilder(self.config)
        self.unknown_tracker = UnknownObjectTracker(self.config, self.get_logger())
        self.persistent_tracker = PersistentObjectTracker(self.config, self.get_logger(), coordinator=self)
        self.tf_broadcaster = TransformBroadcaster(self) if self.config.publish_hydra_tf else None

        # Initialize bounding box diagnostics logger for post-run analysis.
        # Gated by phase1.diagnostics.enabled together with every other per-run
        # diagnostic writer below.
        self.diagnostics_enabled = bool(getattr(self.config, "diagnostics_enabled", False))
        self.bbox_diagnostics_logger = BboxDiagnosticsLogger(
            enabled=self.diagnostics_enabled,
            output_dir=os.path.expanduser(getattr(self.config, 'bbox_log_dir', str(workspace_path('debug', 'bbox_diagnostics'))))
        )

        # Initialize modular pipeline stages
        self.seg_stage = SegmentationStage(self.sam_backend, self.config, self.get_logger())
        self.track_stage = TrackingStage(self.persistent_tracker, self.config, self.get_logger(), geometry_estimator=self.geometry_estimator)
        self.sem_stage = SemanticsStage(self.config, self.get_logger())
        self.pub_stage = PublishingStage(self.config, self.get_logger(), bridge=self.bridge, tf_broadcaster=self.tf_broadcaster)
        self.presence_stage = LocalSegmentPresenceStage(self, self.config, self.get_logger())
        self.crop_registry = TrackCropRegistry(self, self.config, self.get_logger())
        self.semantic_dispatch = SemanticLabelDispatchStage(self, self.config, self.get_logger())

        # Hydra receives one fixed slot ID per physical object.  Semantic names
        # are applied by the downstream scene-graph fuser, therefore Phase 1 does not rewrite or
        # reserve Hydra label-space entries across sessions.
        self.static_hydra_label_ids = dict(self.config.hydra_label_lookup)
        self.static_hydra_label_names = dict(self.config.hydra_label_names)
        self.persistent_tracker.set_reserved_slot_ids(set())

        # Restore a previous session's tracks, if enabled. Done here on purpose:
        # the tracker exists, TrackingStage already holds the reference, but no
        # subscription has been created and no worker thread has started, so
        # nothing can race the load. Restored tracks carry timestamps shifted
        # into the past, which is what routes them through the tracker's
        # revisit-association branch instead of the recent one.
        self.tracker_state_path = (
            Path(self.config.session_persistence_state_path).expanduser()
            if self.config.session_persistence_state_path
            else None
        )
        # Restored tracks that already carry a label: their label must be
        # republished to the fuser when they are re-observed, because the
        # fuser's overlay cache is per-process and the label otherwise only
        # ever arrives from a RAP/VLM completion that will never happen for
        # them. Emptied as each track is seen again.
        self._restored_label_pending: set = set()
        self._restored_presence_pending: set = set()
        if self.config.session_persistence_enabled and self.tracker_state_path is not None:
            try:
                load_tracker_state(
                    self.persistent_tracker, self.tracker_state_path, self.get_logger()
                )
                self._restored_label_pending = {
                    track_id
                    for track_id, track in self.persistent_tracker._tracks.items()
                    if (track.semantic_label or track.canonical_label)
                }
                # Every restored track needs its slot -> internal_object_id
                # mapping republished, labelled or not, so the fuser can group
                # a multi-segment object instead of drawing solid contact edges
                # between its own segments.
                #
                # Tracked per *slot*, not per track. The ordinary heartbeat
                # republishes only the one slot observed in the current frame
                # (_publish_active_local_segments keys on record's
                # hydra_slot_id), so retiring a whole track the moment any one
                # of its segments is re-observed strands the rest: they stop
                # being published having possibly never been published at all.
                self._restored_presence_pending = {
                    int(slot_id)
                    for track in self.persistent_tracker._tracks.values()
                    for slot_id in track.segments
                    if int(slot_id) > 0
                }
                self.get_logger().info(
                    f"{len(self._restored_label_pending)} restored tracks carry a label "
                    "and will republish it to the fuser on re-observation"
                )
            except Exception as exc:
                self.get_logger().error(f"Tracker state restore failed, starting fresh: {exc}")
        self.semantic_reuse_enabled = False
        # RAP/VLM scheduling is intentionally decoupled from Hydra output.  A
        # persistent track receives a unique Hydra slot immediately, while the
        # workers receive only its track ID.  The current best crop remains
        # mutable until the relevant worker dequeues that ID; an immutable crop
        # snapshot is then used for that worker's single inference request.
        self.rap_runs_async = bool(self.config.rap_enabled)
        self.rap_runs_synchronously = False
        # Best-crop-per-track store lives on TrackCropRegistry.
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

        # RAP queue, worker-thread bookkeeping, and accuracy diagnostics
        # writer live on RapDispatchStage (constructed further below, once
        # its diagnostics object exists). The worker FIFO stores only
        # persistent track IDs -- crops are never held in the queue.

        # VLM queue, worker-thread bookkeeping (including the quality-defer
        # and post-failure retry pools), backend, and diagnostics writer
        # live on VlmDispatchStage (constructed further below, once its
        # diagnostics object exists).

        # Risk-VLM queue, worker-thread bookkeeping, backend, and diagnostics
        # writer live on RiskVlmDispatchStage (constructed further below,
        # once its diagnostics object exists).

        self._stop_event = threading.Event()
        # Segmentation (GPU-bound SAM) and tracking/publish (CPU-bound) run on
        # separate threads so they can overlap: SAM backends release the GIL
        # for most of their wall time while waiting on the GPU, the same
        # property the RAP/VLM worker threads below already rely on.
        self._segmentation_thread = threading.Thread(target=self._segmentation_loop, daemon=True)
        self._tracking_publish_thread = threading.Thread(target=self._tracking_publish_loop, daemon=True)

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
        # Same JSON-over-String-topic pattern as semantic_label_result, kept
        # on its own topic since risk arrives later (after a second,
        # independent VLM call) and carries a different payload shape
        # (risk_score/risk_factors, not label/mobility).
        self.risk_result_pub = self.create_publisher(
            String,
            self.config.risk_result_topic,
            semantic_label_qos,
        )
        self.active_segments_pub = self.create_publisher(
            String,
            self.config.persistent_active_segments_topic,
            output_qos,
        )
        self.status_pub = self.create_publisher(String, self.config.status_topic, output_qos)
        self.unknown_pub = self.create_publisher(String, self.config.unknown_candidates_topic, output_qos)

        # Loop-closure re-anchoring: watch ``map -> odom`` for a drift-correction
        # step and rigid-transform the whole persistent-object cache when it
        # jumps.  Off unless a front end actually publishes a non-identity
        # ``map -> odom`` (LCD on, split map/odom frames).  When enabled it adds
        # only two plain ``/tf`` subscriptions on this node's own executor -- no
        # extra node, no second executor -- and a per-frame check that
        # early-returns until a matching ``map -> odom`` transform is seen.
        self._loop_closure_enabled = bool(getattr(self.config, "loop_closure_enabled", False))
        self._lc_map_frame = str(getattr(self.config, "loop_closure_map_frame", "map"))
        self._lc_odom_frame = str(getattr(self.config, "loop_closure_odom_frame", "odom"))
        self._last_map_odom: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._pending_map_odom: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._map_odom_lock = threading.Lock()
        self.loop_closure_pub = None
        if self._loop_closure_enabled:
            from tf2_msgs.msg import TFMessage

            tf_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST, depth=100,
                reliability=ReliabilityPolicy.RELIABLE,
            )
            tf_static_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST, depth=100,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._tf_sub = self.create_subscription(TFMessage, "/tf", self._on_tf, tf_qos)
            self._tf_static_sub = self.create_subscription(
                TFMessage, "/tf_static", self._on_tf, tf_static_qos
            )
            self.loop_closure_pub = self.create_publisher(
                String, self.config.loop_closure_event_topic, output_qos
            )
            self.get_logger().info(
                "Loop-closure re-anchoring enabled: watching "
                f"{self._lc_map_frame} -> {self._lc_odom_frame}"
            )

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

        # Comprehensive crop evolution diagnostics for debugging overlaps and tracking issues
        # (Path is imported at module scope. A function-local `from pathlib
        # import Path` here would make Path a local for the whole of __init__,
        # so every earlier use of it -- e.g. the tracker state path -- would
        # raise UnboundLocalError.)
        crop_evolution_dir = Path(self.config.timing_csv_path).parent / "crop_evolution"
        self.crop_evolution_tracker = CropEvolutionTracker(
            enabled=self.diagnostics_enabled,
            output_dir=str(crop_evolution_dir),
            logger=self.get_logger(),
        )

        # Tracking quality evaluation diagnostics. Gated on the object-tracking
        # sub-switch as well as the master one, so these can be turned off
        # without also silencing the risk-VLM / RAP writers.
        self.tracking_diagnostics_enabled = (
            self.diagnostics_enabled
            and bool(getattr(self.config, "diagnostics_log_tracking", True))
        )
        tracking_quality_dir = workspace_path("debug/object_tracking_experiment_part2/tracking_quality")
        self.tracking_quality_recorder = TrackingQualityRecorder(
            enabled=self.tracking_diagnostics_enabled,
            output_dir=str(tracking_quality_dir),
            logger=self.get_logger(),
        )

        # RAP-VLM diagnostic crops (best updates, RAP dequeues, VLM dequeues).
        # The manager's functional helpers (crop scoring, mask filtering,
        # contour highlighting) always run; only the disk writes are gated.
        rap_vlm_crops_dir = workspace_path("RAP-VLM crops")
        self.tracking_crop_manager = TrackingCropManager(
            output_dir=rap_vlm_crops_dir,
            enabled=self.diagnostics_enabled,
        )

        # VLM testing diagnostics (crops + per-call CSV for manual verification).
        # During a prompt experiment, write under <output_root>/<run_id>/
        # session_<timestamp>/ so repeated runs of the same matrix row never
        # overwrite each other. The two experiment toggles are mutually
        # exclusive in practice (object_detail_experiment checked first since
        # it is the active one as of its introduction; the label-prompt
        # experiment is frozen/complete -- see phase1_config.py).
        if self.config.vlm_object_detail_experiment_enabled and self.config.vlm_object_detail_experiment_output_root:
            _vlm_diag_dir = Path(self.config.vlm_object_detail_experiment_output_root) / (
                self.config.vlm_object_detail_experiment_run_id or "unnamed_run"
            )
            _vlm_diag_run_id = self.config.vlm_object_detail_experiment_run_id
            _vlm_diag_prompt_version = self.config.vlm_object_detail_experiment_prompt_version
            _vlm_diag_crop_format = self.config.vlm_object_detail_experiment_crop_format
        elif self.config.vlm_prompt_opt_enabled and self.config.vlm_prompt_opt_output_root:
            _vlm_diag_dir = Path(self.config.vlm_prompt_opt_output_root) / (
                self.config.vlm_prompt_opt_run_id or "unnamed_run"
            )
            _vlm_diag_run_id = self.config.vlm_prompt_opt_run_id
            _vlm_diag_prompt_version = self.config.vlm_prompt_opt_prompt_version
            _vlm_diag_crop_format = "jpg"
        else:
            _vlm_diag_dir = workspace_path("VLM-Test-Session")
            _vlm_diag_run_id = self.config.vlm_prompt_opt_run_id
            _vlm_diag_prompt_version = self.config.vlm_prompt_opt_prompt_version
            _vlm_diag_crop_format = "jpg"
        self.vlm_test_diagnostics = VLMTestDiagnostics(
            output_dir=_vlm_diag_dir,
            run_id=_vlm_diag_run_id,
            model_profile=self.config.vlm_active_profile,
            prompt_version=_vlm_diag_prompt_version,
            enabled=self.diagnostics_enabled,
            crop_format=_vlm_diag_crop_format,
        )
        if self.diagnostics_enabled:
            self.get_logger().info(
                f"VLM crop diagnostics -> {self.vlm_test_diagnostics.get_session_dir()}"
            )
        else:
            self.get_logger().info("Phase 1 per-run diagnostics disabled (phase1.diagnostics.enabled=false)")
        self.vlm_stage = VlmDispatchStage(
            self, self.config, self.get_logger(),
            backend=_vlm_backend, test_diagnostics=self.vlm_test_diagnostics,
        )
        self._vlm_thread = threading.Thread(target=self.vlm_stage._vlm_loop, daemon=True)

        # Risk VLM diagnostics (crops + per-call CSV). Gated on risk_vlm
        # actually being enabled, not just the diagnostics master switch --
        # otherwise this creates an empty session_<timestamp>/ folder on
        # every single run even while the feature itself is off.
        self.risk_vlm_diagnostics = RiskVlmDiagnostics(
            output_dir=workspace_path("debug/risk_assessment_feature"),
            enabled=self.diagnostics_enabled and self.config.risk_vlm_enabled,
        )
        self.risk_stage = RiskVlmDispatchStage(
            self, self.config, self.get_logger(),
            backend=_risk_vlm_backend, diagnostics=self.risk_vlm_diagnostics,
        )
        self._risk_thread = threading.Thread(target=self.risk_stage._risk_loop, daemon=True)

        # RAP accuracy diagnostics: crop + outcome for every RAP attempt
        # (hit or miss), for manual cross-checking of RAP's real-world
        # accuracy in one complete, self-contained CSV. Gated on RAP
        # actually being enabled, for the same reason as risk_vlm above.
        self.rap_accuracy_diagnostics = RapAccuracyDiagnostics(
            output_dir=workspace_path("debug/rap_accuracy_test"),
            enabled=self.diagnostics_enabled and self.config.rap_enabled,
        )
        self.rap_stage = RapDispatchStage(
            self, self.config, self.get_logger(),
            backend=_rap_backend, accuracy_diagnostics=self.rap_accuracy_diagnostics,
        )
        self._rap_thread = threading.Thread(target=self.rap_stage._rap_loop, daemon=True)

        # Periodic per-track crop diagnostics: saves every Nth observation
        # of every track (raw crop + both the raw per-frame geometry and
        # the accumulated local-segment envelope), independent of "best
        # crop" selection -- for tracing exactly when/how a track's mask or
        # bounding box starts absorbing a different object over time.
        self.periodic_crop_diagnostics = PeriodicCropDiagnostics(
            output_dir=workspace_path("debug/object_tracking_experiment_part2/periodic_crop_diagnostics"),
            enabled=self.tracking_diagnostics_enabled,
            interval=self.config.periodic_crop_interval,
        )

        # Full-frame diagnostic: every Nth frame saved whole, with every SAM
        # mask that frame outlined -- answers "did SAM detect this object at
        # all, and starting when" directly, without first having to know
        # which (if any) track it ended up under.
        self.frame_mask_overlay_diagnostics = FrameMaskOverlayDiagnostics(
            output_dir=workspace_path("debug/frame_mask_overlays"),
            enabled=self.diagnostics_enabled and self.config.frame_mask_overlay_enabled,
            interval=self.config.frame_mask_overlay_interval,
        )
        if self.diagnostics_enabled:
            self.get_logger().info(
                f"Risk VLM diagnostics -> {self.risk_vlm_diagnostics.get_session_dir()}"
            )

        self.received_count = 0
        self.processed_count = 0
        self.failed_count = 0
        self.dropped_count = 0
        self.hydra_published_count = 0

        self._segmentation_thread.start()
        self._tracking_publish_thread.start()
        if self.rap_runs_async:
            self._rap_thread.start()
        if self.config.vlm_enabled:
            self._vlm_thread.start()
        if self.config.risk_vlm_enabled:
            self._risk_thread.start()
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
            f"Risk VLM enabled={self.config.risk_vlm_enabled}, mode={self.config.risk_vlm_mode}, "
            f"model={self.config.risk_vlm_model}, endpoint={self.config.risk_vlm_endpoint}, "
            f"yield_to_object_vlm={self.config.risk_vlm_yield_to_object_vlm}"
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

                # Log bounding boxes for post-run diagnostic analysis
                try:
                    if self.bbox_diagnostics_logger.enabled and result.success:
                        objects = safe_json_loads(result.object_metadata_json, default=[])
                        if objects:
                            tracks_by_id = {obj.get("persistent_track_id"): obj for obj in objects if obj.get("persistent_track_id")}
                            if tracks_by_id:
                                self.bbox_diagnostics_logger.log_frame_tracks(frame.sequence, tracks_by_id)
                except Exception as exc:
                    if rclpy.ok() and not self._stop_event.is_set():
                        self.get_logger().debug(f"Failed to log bbox diagnostics: {exc}")
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
        hydra_msg, hydra_stage_ms = self.pub_stage.build_hydra_frame(frame, result, build_start, cached)
        hydra_build_delay_ms = (time.perf_counter() - build_start) * 1000.0

        publish_start = time.perf_counter()
        publish_success = True
        if self.config.publish_hydra_combined:
            publish_success = self._safe_publish(self.hydra_frame_pub, hydra_msg) and publish_success
        if self.config.publish_hydra_separate_topics:
            publishers = {
                "hydra_rgb_pub": self.hydra_rgb_pub,
                "hydra_depth_pub": self.hydra_depth_pub,
                "hydra_camera_info_pub": self.hydra_camera_info_pub,
                "hydra_pose_pub": self.hydra_pose_pub,
                "hydra_semantic_pub": self.hydra_semantic_pub,
                "hydra_instance_pub": self.hydra_instance_pub,
                "hydra_metadata_pub": self.hydra_metadata_pub,
            }
            publish_success = self.pub_stage.publish_separate_hydra_topics(hydra_msg, publishers) and publish_success
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
                # Wall-clock (not bag/sim time). This is the real one-row
                # -per-processed-frame event -- added for the revisit/FPS
                # experiment (debug/revisit_experiment): frame throughput
                # over a run is (count of event=="frame_trace" rows) /
                # (max - min of this column among them), and it lines up
                # against monitor_resources.py's tegrastats samples, which
                # are also real wall-clock time.
                wall_clock_unix_sec=time.time(),
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
        self.pub_stage.add_evidence_record(hydra_msg, result, self.evidence_buffer)
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
        sam_masks, sam_prep, seg_timing = self.seg_stage.run(rgb, depth)
        sam_delay_ms = (time.perf_counter() - sam_start) * 1000.0
        # Timing metrics from segmentation stage
        sam_prepare_delay_ms = float(seg_timing.get("sam_prepare_ms", 0.0))
        sam_inference_delay_ms = float(seg_timing.get("sam_inference_ms", 0.0))
        sam_restore_delay_ms = float(seg_timing.get("sam_restore_ms", 0.0))
        sam_other_ms = max(
            0.0,
            sam_delay_ms - sam_prepare_delay_ms - sam_inference_delay_ms - sam_restore_delay_ms,
        )
        # Keep only lightweight scalar metadata here -- this dict is JSON-serialized
        # into every per-frame Hydra message (metadata["sam_input_processing"]).
        # _prepare_input returns the full processed rgb/depth/valid-mask arrays in
        # the same dict; serializing those was ~8 MB and ~1 s of GIL-held json.dumps
        # per frame on the tracking/publish thread, which starved the segmentation
        # thread's SAM calls (see the SAM-throughput investigation).
        _sam_prep_drop = {"rgb", "depth", "valid_depth_mask_sam"}
        sam_prep_summary = {
            k: v
            for k, v in sam_prep.items()
            if k not in _sam_prep_drop and not isinstance(v, np.ndarray)
        }

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

        # Log frame start for tracking quality evaluation
        self.tracking_quality_recorder.log_frame_start(
            frame_id=frame.rsg_frame_id,
            sequence=frame.sequence,
            sam_mask_count=len(sam_masks)
        )

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
            self.presence_stage._publish_active_local_segments(frame, track_records, current_timestamp_sec)
            # Push restored labels out early so the resumed map opens labelled,
            # instead of each object staying blank until it is revisited.
            self.semantic_dispatch._drain_restored_semantic_labels(frame, current_timestamp_sec)
        frame_stage_ms["active_segments_publish_ms"] = (time.perf_counter() - stage_start) * 1000.0
        stage_start = time.perf_counter()
        if self.config.persistent_tracking_enabled and self.config.semantic_labeling_enabled:
            semantic_label_dispatches = self.semantic_dispatch._dispatch_tracks_after_settling(current_timestamp_sec)
        frame_stage_ms["semantic_dispatch_ms"] = (time.perf_counter() - stage_start) * 1000.0
        stage_start = time.perf_counter()
        if self.config.persistent_tracking_enabled and self.config.semantic_labeling_enabled:
            self.vlm_stage._release_quality_deferred_vlm_if_expired(current_timestamp_sec)
        frame_stage_ms["quality_deferred_release_ms"] = (time.perf_counter() - stage_start) * 1000.0
        if self.config.persistent_tracking_enabled:
            self._prune_expired_dynamic_tracks(frame, current_timestamp_sec)
        rap_delay_ms = (time.perf_counter() - rap_start) * 1000.0

        label_start = time.perf_counter()
        semantic, instance, label_table, objects, unknowns = self.label_map_builder.build(rgb.shape[:2], classified)
        label_map_delay_ms = (time.perf_counter() - label_start) * 1000.0
        metadata_start = time.perf_counter()
        # VLM dispatch is driven entirely by _dispatch_tracks_after_settling
        # (above), for both RAP-enabled and RAP-disabled operation.  When RAP is
        # off, settled tracks are routed straight to the VLM FIFO by ID there.
        # The old per-frame dispatch_unknowns_to_vlm path is inert on this branch
        # (legacy unknown_tracker._tracks is never populated) so it is not called.
        vlm_dispatch: List[Dict[str, Any]] = []
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

    def _on_tf(self, msg) -> None:
        """Capture the latest ``map -> odom`` transform (loop-closure only).

        Runs on this node's executor, same as ``frame_callback``.  Does nothing
        but store the newest matching transform; all decision logic is deferred
        to ``_maybe_reanchor_on_loop_closure`` on the tracking thread.
        """
        for tr in msg.transforms:
            if (
                tr.header.frame_id.lstrip("/") != self._lc_map_frame
                or tr.child_frame_id.lstrip("/") != self._lc_odom_frame
            ):
                continue
            t = tr.transform.translation
            q = tr.transform.rotation
            reading = (
                quat_to_rot(q.x, q.y, q.z, q.w),
                np.array([t.x, t.y, t.z], dtype=np.float64),
            )
            with self._map_odom_lock:
                self._pending_map_odom = reading

    def _maybe_reanchor_on_loop_closure(self, timestamp_sec: float) -> None:
        """Re-anchor the persistent-object cache after a ``map -> odom`` jump.

        Called once per frame on the tracking/publish thread, immediately
        before ``begin_frame``.  Early-returns unless loop-closure is enabled
        *and* a ``map -> odom`` transform has actually been received.  The first
        reading only records the baseline; a later step beyond the configured
        thresholds triggers ``PersistentObjectTracker.reanchor_all`` and,
        optionally, ``merge_reanchor_duplicates``.
        """
        if not self._loop_closure_enabled:
            return
        with self._map_odom_lock:
            pending = self._pending_map_odom
        if pending is None:
            return  # no map -> odom seen yet -- nothing to react to

        rot_new, trans_new = pending
        if self._last_map_odom is None:
            self._last_map_odom = (rot_new, trans_new)
            return

        rot_old, trans_old = self._last_map_odom
        delta = loop_closure_delta(
            rot_old, trans_old, rot_new, trans_new,
            min_translation_m=float(self.config.loop_closure_min_translation_m),
            min_rotation_deg=float(self.config.loop_closure_min_rotation_deg),
        )
        if delta is None:
            self._last_map_odom = (rot_new, trans_new)
            return
        rot_delta, trans_delta, trans_norm, angle_deg = delta

        n_tracks = self.persistent_tracker.reanchor_all(
            rot_delta, trans_delta, stamp=timestamp_sec
        )
        n_merged = 0
        if self.config.loop_closure_merge_duplicates:
            n_merged = self.persistent_tracker.merge_reanchor_duplicates(
                correction_translation_m=trans_norm,
                now_sec=timestamp_sec,
                recent_window_sec=float(self.config.loop_closure_merge_recent_window_sec),
                distance_slack_m=float(self.config.loop_closure_merge_distance_slack_m),
            )
        self._last_map_odom = (rot_new, trans_new)
        self.get_logger().warn(
            f"loop-closure re-anchor: |dt|={trans_norm:.3f} m dtheta={angle_deg:.2f} deg "
            f"tracks={n_tracks} merged={n_merged}"
        )
        if self.loop_closure_pub is not None:
            msg = String()
            msg.data = safe_json_dumps(
                {
                    "timestamp_sec": float(timestamp_sec),
                    "delta_translation_m": [float(v) for v in trans_delta],
                    "delta_translation_norm_m": trans_norm,
                    "delta_rotation_deg": angle_deg,
                    "tracks_reanchored": int(n_tracks),
                    "tracks_merged": int(n_merged),
                    "map_frame": self.config.loop_closure_map_frame,
                    "odom_frame": self.config.loop_closure_odom_frame,
                }
            )
            self.loop_closure_pub.publish(msg)

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
        # Populated per-mask below as track association runs, then rendered
        # onto the frame-level overlay diagnostic after the loop -- the
        # overlay needs each mask's final track_id, which isn't known until
        # persistent_tracker.associate() runs for that mask.
        mask_track_ids: Dict[str, str] = {}

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
            self._maybe_reanchor_on_loop_closure(timestamp_sec)
            self.persistent_tracker.begin_frame()

        # Build geometry for every SAM observation before mutating any track.
        # This enables track-aware mask redundancy analysis (A2) and one global
        # frame-level assignment (E), eliminating SAM-output-order bias.
        prepared: List[Dict[str, Any]] = []
        # Masks with too little in-range depth to yield 3D geometry carry
        # nothing usable -- no centroid, no box, nothing to add to the
        # semantic map for those pixels. The threshold is the geometry
        # estimator's own `min_valid_depth_points`, not zero: a mask whose
        # object lies beyond max_depth_m can still pick up a handful of
        # in-range points from near-field speckle elsewhere in the same
        # contour, which is enough to clear a ==0 test but not enough to
        # produce geometry. Those fell through as phantom single-observation
        # tracks (16 of 101 in run 194600), because a candidate without 3D
        # can reach at most 2 evidence votes (image + temporal) and so can
        # never satisfy a quorum of 3 -- it can only ever start a new track.
        # Tracked separately from keep_mask (rather than skipping the
        # `prepared` append outright) so `prepared`/`sam_masks`/`keep_mask`
        # stay strictly index-aligned for the second loop below.
        depth_range_keep: List[bool] = []
        for idx, mask in enumerate(sam_masks):
            stage_start = time.perf_counter() if timing_enabled else 0.0
            candidate_id = make_candidate_id(frame, mask.mask_id, idx, False)
            metadata, filtered_mask = self.build_object_metadata(
                frame=frame, mask=mask, depth=depth, tx=tx, rot_m=rot_m,
                label="unknown_object", label_id=0, instance_id=idx + 1,
                confidence=0.0, status="collecting_best_crop",
                candidate_id=candidate_id, rap_metadata={},
                geometry_stage_ms=geometry_stage_ms,
            )
            prepared.append({
                # "mask" carries the filtered (largest-island-only) mask so
                # every downstream consumer -- geometry above, and the
                # semantic pixel image below -- agrees on the same object
                # footprint. See build_object_metadata's docstring.
                "metadata": metadata, "mask": filtered_mask,
                "timestamp_sec": timestamp_sec, "desired_hydra_label_id": 0,
            })
            # `depth_valid_points` is absent when the depth gather never ran
            # (geometry disabled, or an empty mask). Keep those: this gate is
            # about depth range only and must not silently drop every mask
            # when object geometry is switched off.
            depth_valid_points = metadata.get("depth_valid_points")
            if (
                self.config.reject_masks_fully_outside_depth_range
                and depth_valid_points is not None
                and int(depth_valid_points) < int(self.config.min_valid_depth_points)
            ):
                depth_range_keep.append(False)
            else:
                depth_range_keep.append(True)
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
        if self.config.reject_masks_fully_outside_depth_range:
            keep_mask = [keep and in_range for keep, in_range in zip(keep_mask, depth_range_keep)]

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
            candidate_id = make_candidate_id(frame, mask.mask_id, idx, False)
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
            mask_track_ids[str(mask.mask_id)] = external_track_id or "unassigned"
            # Keep updating the shared best crop. RAP/VLM receive only this
            # track ID and retrieve the latest crop when each worker dequeues it.
            stage_start = time.perf_counter() if timing_enabled else 0.0
            self.crop_registry._remember_track_crop(external_track_id, rgb, metadata, frame, mask.mask)
            if timing_enabled:
                crop_update_ms += (time.perf_counter() - stage_start) * 1000.0

            if external_track_id:
                try:
                    self.periodic_crop_diagnostics.log_observation(
                        external_track_id,
                        int(frame.sequence),
                        rgb,
                        mask.mask,
                        metadata,
                        timestamp=timestamp_sec,
                    )
                except Exception as e:
                    self.get_logger().warn(f"Failed to log periodic crop diagnostics: {e}")

                # A track restored from a previous session keeps its label and
                # is therefore never re-sent to RAP/VLM -- but the fuser's
                # overlay cache is per-process and starts empty, and the label
                # only ever reaches it from a RAP/VLM completion. Without this
                # the object would be tracked correctly and still render
                # unlabeled.
                #
                # _drain_restored_semantic_labels normally gets there first, so
                # this is just the fast path for a track re-observed before the
                # drain reached it. Emitting is idempotent: the first one to
                # run discards the track from the pending set.
                if external_track_id in self._restored_label_pending:
                    self.semantic_dispatch._emit_restored_semantic_label(
                        external_track_id, frame, timestamp_sec
                    )
                # Presence is retired per slot in _publish_active_local_segments,
                # not per track here: this stage runs before the publish stage,
                # so discarding the track now would strand every segment of it
                # that has not been re-observed yet.

            # Extract crop for diagnostic inspection
            try:
                # bbox_2d should be in metadata from object_geometry
                bbox_2d = metadata.get("bbox_2d")  # (x_min, y_min, x_max, y_max)
                if external_track_id and bbox_2d:
                    # Check if this is a new track (first observation)
                    is_new_track = track_record.get("persistent_match_reason") == "new_track"
                    # Crop saving disabled (diagnostic feature for Phase 2 optimization)
            except Exception as exc:
                if hasattr(self, '_crop_extraction_errors'):
                    self._crop_extraction_errors += 1
                else:
                    self._crop_extraction_errors = 1

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

            # Gate the REAL label from reaching Hydra's semantic pixel image
            # until this track has survived a few observations -- a one-off
            # spurious detection (bad SAM prompt, single-frame noise) then
            # never paints a real label at all and stays harmless background
            # forever (see MLESemanticIntegrator: background/label 0 is
            # skipped entirely, never accumulates or entrenches). Everything
            # else -- metadata["hydra_label_id"]/bbox_diagnostics, crops,
            # RAP/VLM dispatch -- still uses the track's real slot from
            # observation 1, since those aren't painted into Hydra's TSDF and
            # don't have the entrenchment problem this specifically guards
            # against. Default (1) is a no-op: paint from the first
            # observation, same as before this existed.
            min_obs_for_paint = int(self.config.persistent_min_observations_before_semantic_paint)
            hydra_paint_label_id = semantic_label_id
            if min_obs_for_paint > 1 and external_track_id:
                seen_count = self.persistent_tracker.get_seen_count(external_track_id)
                if seen_count < min_obs_for_paint:
                    hydra_paint_label_id = 0

            classified_mask = ClassifiedMask(
                mask_id=mask.mask_id,
                # Filtered (largest-island-only) mask, same one geometry was
                # computed from in the loop above -- so the semantic pixel
                # image Hydra integrates agrees with this object's own bbox,
                # instead of painting islands its own geometry ignored.
                mask=prepared[idx]["mask"],
                label=str(semantic_label_name),
                label_id=int(hydra_paint_label_id),
                instance_id=int(instance_id),
                confidence=float(rap_info.get("confidence", 0.0) or 0.0),
                status=str(metadata["status"]),
                candidate_id=candidate_id,
                metadata=metadata,
            )
            classified.append(classified_mask)

            metadata["rap_dispatch_status"] = "track_id_pending_rap" if self.config.rap_enabled else "rap_disabled"
            next_instance_id += 1

        try:
            # Overlay the filtered (largest-island-only) mask tracking actually
            # used, not the raw SAM output -- e.g. a mask split by an occluding
            # object shows only the surviving island, matching what geometry
            # and the semantic image saw, not what SAM proposed.
            overlay_masks = [
                SimpleNamespace(mask_id=mask.mask_id, mask=prepared[idx]["mask"])
                for idx, mask in enumerate(sam_masks)
            ]
            self.frame_mask_overlay_diagnostics.log_frame(
                int(frame.sequence), rgb, overlay_masks, track_ids=mask_track_ids,
            )
        except Exception:
            pass

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
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """Create configurable object metadata used by Hydra/fusion/risk nodes.

        Returns the metadata alongside the filtered mask (largest island
        only) so callers can reuse the exact same mask for the semantic
        pixel image sent to Hydra -- otherwise Hydra integrates the raw,
        unfiltered SAM mask (both islands) even though this object's own
        geometry was computed from one island only.
        """
        # Use filtered mask for geometry (only largest contour, no islands)
        filtered_mask = self.tracking_crop_manager.get_filtered_mask(mask.mask)
        geometry = self.geometry_estimator.estimate(filtered_mask, depth, frame.camera_info, tx, rot_m, stage_ms=geometry_stage_ms)
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
        if self.config.persistent_tracking_enabled:
            return metadata, filtered_mask
        return filter_metadata(metadata, self.config), filtered_mask

    def _prune_expired_dynamic_tracks(self, frame: RsgFrame, current_timestamp_sec: float) -> None:
        """Delete confirmed-dynamic tracks once presence confidence decays.

        See PersistentObjectTracker.prune_expired_dynamic_tracks for the
        decision (it already did the deletion by the time this runs) -- this
        only handles the two things that live in phase1.py: telling the
        fuser to drop the corresponding object from its output, and clearing
        this node's own per-track bookkeeping so nothing stale lingers.
        """
        deleted = self.persistent_tracker.prune_expired_dynamic_tracks(current_timestamp_sec)
        for record in deleted:
            track_id = str(record.get("track_id", ""))
            if not track_id:
                continue
            self.semantic_dispatch._finalize_track_queue_state(track_id)
            self.crop_registry._retire_track_crop(track_id)
            self.vlm_stage.clear_retry_state(track_id)
            with self._semantic_label_lock:
                self._semantic_label_pending_track_ids.discard(track_id)
            payload = dict(record)
            payload.update({
                "event": "track_deleted",
                "frame_id": str(frame.rsg_frame_id),
                "sequence": int(frame.sequence),
                "timestamp_sec": float(current_timestamp_sec),
            })
            self._safe_publish(
                self.semantic_label_result_pub, String(data=safe_json_dumps(payload)),
            )
            self.get_logger().info(
                f"Deleted dynamic track {track_id} (hydra_slot_id="
                f"{record.get('hydra_slot_id')}, presence_confidence="
                f"{record.get('presence_confidence'):.3f}, age_sec="
                f"{record.get('age_sec'):.1f})"
            )

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
            "rap_fifo_queue_size": self.rap_stage.queue.qsize(),
            "rap_fifo_queue_max_size": self.config.rap_queue_size,
            "rap_queue_dropped": self.rap_stage.queue_dropped_count,
            "rap_queue_deferred_total": self.rap_stage.queue_deferred_count,
            "rap_deferred_pending": len(self.rap_stage._deferred_track_id_set),
            "rap_completed": self.rap_stage.completed_count,
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
            # Wall-clock (not bag/sim time), added for the revisit/FPS
            # experiment (debug/revisit_experiment) so this row can be lined
            # up against monitor_resources.py's tegrastats samples.
            wall_clock_unix_sec=time.time(),
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
            # Wall-clock -- see the matching comment in publish_timing_event.
            wall_clock_unix_sec=time.time(),
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
            "rap_queue_dropped": self.rap_stage.queue_dropped_count,
            "rap_queue_deferred_total": self.rap_stage.queue_deferred_count,
            "rap_deferred_pending": len(self.rap_stage._deferred_track_id_set),
            "rap_completed": self.rap_stage.completed_count,
            "rap_fifo_queue_size": self.rap_stage.queue.qsize(),
            "rap_fifo_queue_max_size": self.config.rap_queue_size,
            "vlm_queued": self.vlm_stage.queued_count,
            "vlm_queue_dropped": self.vlm_stage.queue_dropped_count,
            "vlm_queue_deferred_total": self.vlm_stage.queue_deferred_count,
            "vlm_deferred_pending": len(self.vlm_stage._deferred_track_id_set),
            "vlm_quality_deferred_pending": len(self.vlm_stage._quality_deferred_track_ids),
            "vlm_fifo_queue_size": self.vlm_stage.queue.qsize(),
            "vlm_fifo_queue_max_size": self.config.vlm_queue_size,
            "risk_completed": self.risk_stage.completed_count,
            "risk_queue_dropped": self.risk_stage.queue_dropped_count,
            "risk_fifo_queue_size": self.risk_stage.queue.qsize(),
            "risk_fifo_queue_max_size": self.config.risk_vlm_queue_size,
        }
        payload.update(self.persistent_tracker.track_counts())
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

        try:
            self.crop_evolution_tracker.save_snapshot(suffix="_final")
            self.crop_evolution_tracker.save_analysis()
        except Exception as exc:
            print(f"Failed to save crop evolution diagnostics: {exc}", flush=True)

        try:
            self.tracking_quality_recorder.save_snapshots(suffix="_final")
            self.tracking_quality_recorder.generate_report(suffix="_final")
        except Exception as exc:
            print(f"Failed to save tracking quality diagnostics: {exc}", flush=True)

        # Tracker state for the next session. save_tracker_state takes the
        # tracker's lock: _stop_event is set above but the tracking thread is
        # not joined until further down, and that join is best-effort with a
        # 0.20 s timeout, so _tracks may still be mutating right now.
        # getattr: destroy_node also runs when __init__ failed partway through,
        # and an exception here would mask the original failure.
        state_path = getattr(self, "tracker_state_path", None)
        if self.config.session_persistence_enabled and state_path is not None:
            try:
                save_tracker_state(
                    self.persistent_tracker,
                    state_path,
                    self.config.session_persistence_time_shift_sec,
                    self.get_logger(),
                    source_run=str(getattr(self, "_session_id", "")),
                )
            except Exception as exc:
                print(f"Failed to save tracker state: {exc}", flush=True)

        # Crop saving disabled (diagnostic feature for Phase 2 optimization)

        try:
            self.bbox_diagnostics_logger.save()
        except Exception as exc:
            print(f"Failed to save bbox diagnostics: {exc}", flush=True)

        # Only RAP-VLM diagnostic crops are saved (best_update, rap, vlm)
        # No summary files or additional diagnostics

        # Do not wait for a long HTTP timeout. Threads are daemon threads and
        # will terminate with the process after a brief cooperative join.
        for thread in (
            self._segmentation_thread,
            self._tracking_publish_thread,
            self._rap_thread,
            self._vlm_thread,
            self._risk_thread,
        ):
            try:
                if thread.is_alive():
                    thread.join(timeout=0.20)
            except Exception:
                pass

        # Clear Hydra cache on shutdown for fresh start on next launch
        try:
            import shutil
            hydra_cache = os.path.expanduser("~/.hydra/uhumans2")
            if os.path.exists(hydra_cache):
                shutil.rmtree(hydra_cache)
                print(f"Cleared Hydra cache: {hydra_cache}", flush=True)
        except Exception as exc:
            print(f"Failed to clear Hydra cache: {exc}", flush=True)

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
