# Loop-closure experiment

Goal: make phase-1 object identity survive a **loop closure** — i.e. when the
SLAM back-end suddenly corrects accumulated odometry drift, the cached
persistent-object tracks (centroids / 3D boxes / spatial index) must be
re-anchored into the corrected frame so that a re-observed object re-associates
with its existing track instead of spawning a duplicate.

This folder holds (a) the odometry analysis that locates loop-closure
opportunities in the uHumans2 bag, (b) the small test bags cut from those
windows, and (c) — later — the test harness for the re-anchoring code.

---

## 1. Files

| file | purpose |
|---|---|
| `find_loop.py` | analyse `/tesse/odom` for spatial revisits; cut a smaller bag |
| `plot_loop.py` | render the trajectory + loop markers to `trajectory_loops.png` |
| `trajectory_loops.png` | XY trajectory of the full bag, coloured by time, loop points marked |
| `README.md` | this file |

### `find_loop.py`

```
python3 find_loop.py analyse [BAG]                     # list candidate loop windows
python3 find_loop.py extract T0 T1 OUT_BAG [BAG]       # cut [T0,T1] bag-clock seconds
```

`analyse` bins `/tesse/odom` to ~5 Hz and, for every pose, finds the earliest
earlier pose within `RADIUS = 1.5 m` separated by at least `MIN_GAP = 25 s` of
travel, then collapses consecutive hits into windows.

`extract` copies **all** topics for messages whose bag timestamp is in
`[T0, T1]`. Latched one-shot topics (`/tf_static`) are always copied and their
timestamp is clamped forward to `T0`, so `ros2 bag play` still delivers the
camera extrinsics before the first frame.

---

## 2. Source bag

`/home/student/datasets/uhumans2/uHumans2_office_s1_00h_ros2`
506.1 s, 742 621 msgs, 40.5 GiB, `/tesse/odom` = GT odometry
(`frame_id = world`, child `base_link_gt`), 101 213 samples.

Trajectory: path length 263.9 m, extent x ∈ [-19.9, 26.2], y ∈ [6.0, 45.2],
z ≈ 2.5 (constant — planar).

## 3. Candidate loop windows  (`find_loop.py analyse`, 2026-09-02)

Times are **bag-clock seconds** (bag starts at 11.49 s; the analyser's t=0 is
the first `/tesse/odom` sample). Closest approach is the minimum XY distance
between the first-visit and revisit pose.

| id | first visit | revisit | gap | closest | suggested cut | ~size | notes |
|----|------------:|--------:|----:|--------:|---------------|------:|-------|
| **L1** | ~23.8 s  | ~221.7 s | 182.3 s | 1.40 m | `[15.8, 229.7]` (213.9 s) | ~17 GB | **big loop** — leaves the start corridor, explores the whole floor, returns to the same corridor. Largest drift-accumulation window; the realistic loop-closure case. |
| L2 | ~239.9 s | ~274.2 s | 33.5 s  | 1.39 m | `[231.9, 282.2]` (50.3 s) | ~4 GB | mid-corridor drive-past; short gap |
| **L3** | ~348.8 s | ~378.0 s | 28.6 s  | 1.41 m | `[325.0, 388.0]` (63 s, start padded) | ~5 GB | **compact room re-entry** (top-right room, x≈22 y≈40). Fast iteration bag; padded start gives ~24 s of pre-visit mapping. |
| L4 | ~0.0 s   | ~506.0 s | 506.0 s | 0.41 m | whole bag | 40 GB | robot returns to its exact start; only exercisable on a full replay |

Tighter-radius rescans (`plot_loop.py`) — tightest true revisits:
`R = 0.6 m`: ~89.8→217.6 s (0.51 m), ~241.6→272.6 s (0.48 m), start/end 0.41 m.

## 4. Cut test bags

```
python3 find_loop.py extract 15.8  229.7 /home/student/datasets/uhumans2/uHumans2_loop_L1
python3 find_loop.py extract 325.0 388.0 /home/student/datasets/uhumans2/uHumans2_loop_L3
```

| bag | window | duration | msgs | size | use |
|---|---|---|---|---|---|
| `uHumans2_loop_L1` | L1 `[15.8, 229.7]` | 213.9 s | 313 499 | 18 GB | end-to-end realistic loop closure (large drift) |
| `uHumans2_loop_L3` | L3 `[325.0, 388.0]` | 63.0 s | 91 714 | 4.7 GB | fast dev-loop; revisit ~53 s in, first visit ~24 s in |

Both verified with `ros2 bag info`: `/tf_static` count = 1, 963 (L3) / 3494 (L1)
`/tesse/left_cam/rgb` frames. L3 bag-clock runs [336.4, 399.4] s.

Replay (same flags as a normal phase-1 run, e.g.):

```
ros2 bag play /home/student/datasets/uhumans2/uHumans2_loop_L3 --clock --rate 0.1
```

---

## 5. Why these bags do not yet trigger a loop closure

The current pipeline runs on **ground-truth odometry** with
`map_frame == odom_frame == world`, a static identity `world→odom` bridge and
Hydra `enable_lcd: false`. `map→odom` is therefore always identity — there is
no drift and no correction to react to, regardless of which bag is replayed.

To actually exercise the re-anchoring path, one of:

* **(A) real** — switch the front end to a drifting odometry source
  (VIO on `/tesse/imu/noisy/imu` + stereo, or dead-reckoned noisy IMU),
  give Hydra separate `map` / `odom` frames and `enable_lcd: true`, and let
  Hydra's LCD detect the loop and publish a non-identity `map→odom`.
* **(B) synthetic** — keep GT odom, and during the revisit window inject a
  scripted step on `map→odom` from a small test node (a known ΔT applied at a
  known bag time). This tests the re-anchor math directly without a working
  drifting SLAM stack, and is the path used by the step-4 harness.

The revisit geometry in these bags is what makes case (B) a *realistic* test:
after the synthetic jump, the re-anchored object cache should line up with
where the robot genuinely re-observes the same objects.

---

## 6. Implementation status (full detail in `IMPLEMENTATION_PLAN.md`)

**Landed**, all gated behind `phase1.loop_closure.enabled: false`:

* `PersistentObjectTracker.reanchor_all(R, t)` — rigid-transforms every
  track/segment `centroid_3d` / `bbox_3d_min/max` / `last_bbox_3d_min/max`
  (8-corner AABB) and rebuilds the spatial index.
* `PersistentObjectTracker.merge_reanchor_duplicates(...)` — drift pass (folds
  the revisit duplicate into its pre-closure identity) + overlap pass.
* `phase1.py` reads `map→odom` from TF each frame; on a step change beyond the
  configured thresholds it fires the re-anchor + merge and publishes
  `/rsg/phase1/loop_closure_event`. No `/hydra/backend/dsg` subscription.
* `phase1.loop_closure` config block; `src/rsg/tests/test_reanchor.py` (9 tests);
  `loop_jump_injector.py` (synthetic `map→odom` step for the harness).

**TODO**: `run_loop_test.sh` end-to-end + baseline; the front-end / split
`map`/`odom` frames prerequisite for a live run (see §5).
