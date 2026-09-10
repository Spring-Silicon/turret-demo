# Shared SAM backend policy

Historical v1 audit below. The current Vegas/Thor-2 branch uses
`sam-shared-v2-auto-target`: automatic selection, visible-ID retention after
centering, and fallback across applied classes when the previous class is absent.
Mem's temporary-occlusion retention is unchanged. See the README for current
keyboard controls. This UI/control update does not change model implementations.

Audited the installed Arc (`spring-edge-turret`) and Thor (`agxthor-5`) packages,
not just the checkout. Audit date: 2026-09-08 (deployment crosses UTC midnight).
Policy version: `sam-shared-v1`.

## Target behavior (now shared)

| Profile | Per-frame output | Automatic selection | Missing selected object | Aim point |
|---|---|---|---|---|
| SAM 3.1 | Stateless boxes; short-lived spatial IDs | Nearest selected class each frame | Hold if none; immediately reacquire nearest matching class | Box center |
| SAM 3.1 Mask | Stateless masks; short-lived spatial IDs | Nearest selected class each frame | Same as Box | Original binary-mask centroid |
| SAM 3.1 Tracking | SAM temporal masks and persistent IDs | Acquire nearest once, retain that ID | Hold through temporary occlusion; reacquire nearest class when ID retires | Original binary-mask centroid |

A click prefers that instance. In Box/Mask the preference ends when it disappears
or becomes centered; ordinary nearest-of-class tracking resumes. In Tracking it
persists through centering and occlusion. Selecting the class again permits a new
nearest-object acquisition. IDs are session-scoped; prompt/model/session changes
cannot reuse a stale selection.

Both temporal implementations use the same `ManagedSession`: retirement requires
**both 5 seconds and at least 16 processed-frame intervals without a usable
detection** (confidence >= 0.5, positive box area, nonempty mask). Reappearance
resets the grace. Retirement compacts SAM's CPU IDs, GPU row metadata and temporal
object state together. IDs are never reused. Capacity is 16 objects per prompt.

Other shared rules:

- Full camera frame, latest-frame scheduling, no inference FPS cap, up to 8 prompts.
- Box and non-temporal Mask use confidence > 0.5; Tracking uses >= 0.5.
- Mask produces at most 10 masks per prompt and reports overflow, without claiming
  temporal identity. Temporal Tracking uses SAM's detector/association/memory.
- Both masked modes use uncomposited mask centroids; empty masks cannot control
  motors. Selection distance is in camera pixels, not distorted square space.
- Calibrated geometry and capture-time encoder interpolation; absolute bounded
  joint-angle goals, 0.003 normalized centering deadband, no guessed-geometry
  fallback when commissioned geometry fails validation.
- Start/Stop/manual motion/model/prompt transitions, hold-on-loss, control wakeup
  on new completed inference, fault handling and API/status fields are shared.
- Completed JPEG + mask + IDs stay paired. A new graph preparation does not
  invalidate the previous completed result or repeatedly feed it to the motors.
- Model startup timeout is 600 seconds; this is not a per-frame allowance.
- Shared worker lifecycle first drains in-flight work, closes stdin for normal
  exit, then uses bounded TERM/KILL fallback for stalled workers. A cancelled
  receive wakes even when native GPU code never returns; it cannot wait the full
  900-second inference deadline. Automatic aiming is held during a model change.
- Both startup defaults are now SAM 3.1 Mask. Reload testing restores the user's
  active selection separately from that boot default.
- Viewer runs in a separate process and does not generate tracking setpoints.
- No browser heartbeat or detection-age timeout. Start-time and encoder-pose
  validity checks remain. Physical joint limits remain +/-90 degrees.
- Device configuration retains camera identity, motor identity/zeros, geometry,
  explicit servo gain settings and packed-versus-register feedback transport.
  These are assembly parameters, not alternate target-selection programs.

## What actually differed before

| Area | Arc before | Thor before |
|---|---|---|
| Box target loss | Strict world-bearing lock: 3-degree reacquire gate, size/ambiguity gates and 3-frame confirmation; could hold indefinitely | Nearest-of-class fallback |
| Mask target loss | Already changed to nearest-of-class, but centroid aiming | Nearest-of-class, box-center aiming |
| Temporal selection | Automatic ID lock plus strict bearing gate | Nearest each frame unless clicked; click ID cleared when centered |
| Temporal loss | Hold active ID; strict geometry could prevent reacquisition | Clicked missing ID held even after retirement |
| Temporal retirement | Deployed frozen adapter lacked the newer lifetime wrapper | Deployed adapter also lacked it, despite newer local docs |
| Tracking mask output | Centroid included | Centroid missing |
| Centering deadband | 0.003 | 0.012 |
| Geometry | Bounded inverse-kinematics solution | Unconstrained solve followed by goal clamp |
| Capture pose | 256 samples, per-axis interpolation after inference | 16 samples, prior complete sample selected before inference |
| Graph progress | Preserved running state and processed pair | Changed state and paused control |
| Startup ready timeout | 600 seconds | 120 seconds |
| Mask graph-shape cache | Eight prompt-count shapes | Last shape only |

Servo code is now shared as well, including one retry for failed/corrupt *reads*,
never blind write retries. Arc's explicitly configured P1200/I0/D1600 remains;
Thor's existing gain registers are not overwritten. That is not evidence that
their physical response has been jointly tuned.

## Hardware/model differences intentionally retained

- Arc Box: current Sleepy-Joe W4A4 optimized image/grounding bundle.
- Arc Mask: current Sleepy-Joe attention8/skip4 fast mask bundle.
- Arc Tracking: pinned Israel native672-v24, W4A4 image/W8A8 heads/BF16 attention,
  672x672 network input, half spatial temporal KV. Its recorded qualification is
  limited (person prompt, 208 overlapping frames); it is not dense equivalence.
- Thor: dense Torch/CUDA implementations with compilation and CUDA graph replay;
  temporal network input is 1008x1008. The model can produce different
  detections/masks from Arc even with identical surrounding policy.
- Arc tracking uses bounded startup graph warmup then replay-or-direct cache
  misses. Thor retains its existing CUDA region capture implementation; new
  shapes can still require graph preparation. This affects latency, not which
  processed result the common controller consumes.
- SAM temporal builder thresholds were already the same: detection 0.4,
  NMS 0.1 (IoM), association IoU 0.1, new detection 0.65, hotstart 15,
  unmatched/duplicate hotstart thresholds 8, reconditioning every 16 frames,
  masklet confirmation on; 7 mask memories and up to 4 conditioning frames.
  Frozen model artifacts were not rewritten by this policy refactor.

Temporal sessions reset on prompt/session-revision, camera identity or image-size
change, reversed capture time, or a capture gap greater than
max(2 seconds, previous worker duration + 2 seconds). Frame-scoped Box/Mask have
no equivalent neural history to reset. Both temporal adapters retain 32 recent
reporting/nonconditioning frames and the required conditioning state, with 16
object pointers, 7 mask memories and up to 4 conditioning frames; this pruning
is separate from the five-second missing-object retirement policy.

Additional unchanged temporal builder policies: suppress unmatched objects beyond
hotstart too; recent-occlusion overlap threshold 0.7; suppress detections near
image boundaries; no hole filling; IoM-based reconditioning threshold 0.5;
reconstruction box IoU threshold -1 and detector score threshold 0.8; postprocess
batch size 1 and unbatched grounding. These are model-internal policies rather
than nearest-object motor selection.

## Code boundary and verification

`hardware.py` exposes the same `InferenceRuntime.prepare() -> WorkerSpec`
interface with `XpuRuntime` and `CudaRuntime` implementations. It selects model
artifacts, runtime libraries, caches and launch arguments only.
`interfaces.py` defines camera, motor and detection ports consumed by the common
controller. The same Dynamixel driver is appropriate for these identical servo
models; separate duplicate drivers are not needed.

`policy.py`, `tracking.py`, `detection.py`, `session_policy.py`, geometry/pose
code and public API/viewer programs are common. The native tracking engine now
inherits the common per-frame output loop rather than duplicating it.

`tests/test-shared-policy.py` replays identical observations through both device
configurations and requires identical selection/status/motor-command traces in
all three profiles. It also exercises native/dense session retirement, click
selection, temporary loss, retirement, session revision and Stop. Hardware-free
tests cannot qualify model accuracy or mechanical tuning; live deployment checks
are recorded separately.

The [live validation record](shared-policy-validation.json) includes all three
profiles on each host, 26 matching shared-module hashes, preserved settings and
the initial Arc GPU-fault recovery. These are smoke checks, not a new accuracy
qualification or a controlled speed comparison: Arc's temporal sample had no
visible detections, whereas Thor's had one. Nonempty mask/identity/retirement and
control parity are covered by deterministic regression tests.

YOLO was removed from the registry, UI, launch path, worker, dedicated requirements
and smoke test. Unknown/legacy YOLO requests return 400 without changing target or
armed state. Historical artifacts/backups are retained for rollback.
