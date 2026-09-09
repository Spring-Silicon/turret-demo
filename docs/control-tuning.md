# Arc turret control qualification (in progress)

This is measured control tuning, not model-accuracy qualification. Do not copy
assembly-specific gains, zeros or intrinsics to another turret without testing.

## Accepted moving-target settings (2026-09-09)

Arc now retains **P400/I0/D0 on both axes**, matching the actual gains read from
Thor. A cued 60-second comparison confirmed the user's reported improvement:
Arc is clearly smoother; very fast motion can still jerk, with more jerking on
Thor in that condition. Both devices had zero new servo retries or faults.
The persisted, adapter-bound profile and full live evidence are in
[the accepted gain comparison](arc-thor-gain-trial.md).

The original P1200/D1600 setting below describes historical stationary/step
qualification. The intermediate pan-P800/D1600 trial failed live and was reverted;
see [the earlier measurements](arc-moving-response.md). Neither is the retained
setting. Model, confidence, filtering, profiles and calibration were not changed
by the accepted gain update.

## Non-tracking mask reacquisition correction (2026-09-08)

The old Arc-only world-bearing lock could hold forever after a mask dropout if
the object moved outside the 3-degree reacquisition gate. Thor instead resumed
nearest-of-class detection selection. `sam3.1-mask` now uses that same fallback
on Arc, independent of `hold_reacquire`: prefer a clicked ID while visible, hold
once when no valid class mask exists, then immediately reacquire the nearest
valid mask centroid of that class. A new detector-associated ID is allowed.
The status reports `continuity="nearest-of-class"`. This can choose a different
instance of the selected class; no appearance re-identification is implied.

SAM Tracking and the other profiles retain their previous continuity policy.
Geometry, model/thresholds, servo gains, zero positions and angle limits are
unchanged. Regression coverage includes dropout with a moved/new ID, wrong-class
rejection, click preference/fallback, no motion while stopped, and unchanged
temporal-identity holding. All 38 controller and 12 association tests also pass
against the installed Arc package. Prior qualification records below describe
the earlier fixed-bearing policy, not the corrected non-tracking mask policy.

## Hold and reacquire policy (2026-09-08)

The user approved retaining the current object through misses, with explicit
click-to-retarget preserved. `tracking.hold_reacquire=true` now enables a small
world-bearing association layer on Arc. It does not change the mask model,
confidence, centroid calculation, geometry, PD gains, servo speed or smoothing.
The independent `bearing_filter` remains disabled. Thor is not changed.

Class selection acquires once; centering does not release the identity. A gap
holds measured motor position once. The anchor survives arbitrarily long gaps,
without widening the search gate or falling back to another central object.
Reacquisition requires three consecutive, spatially unambiguous observations
within 3 degrees and 0.5–2x angular size of the last accepted object. Same-ID
continuous updates are immediate and unsmoothed; jumps over 6 degrees or gross
size changes are treated as lost measurements. Explicit click/class reselection
clears that anchor. A still-active temporal identity cannot be replaced by a
different one; retired/new IDs can be spatially reacquired.

Limitations: this is spatial association, not appearance re-identification.
Two similar objects at the same bearing can be indistinguishable; an object that
moves far during an occlusion can require a click. These thresholds are a
conservative baseline, not a qualification of every moving scene.

Tests: 12 association unit cases plus six controller integration cases cover
centering, long loss, repeated/invalid frames, changed IDs, confirmation resets,
ambiguity, mask jumps, unchanged raw continuous motion, temporal expiry, explicit
retargeting, and +/-12-degree camera motion compensation. UI tests cover holding
and reacquiring labels and suppress false target highlights. The full validation
suite passes (two pre-existing dependency-gated mask-output skips).

Recorded-light replay preserves all 790 post-selection samples in the first ten
clean trials, without any new holds. Both first incorrect target switches in the
last two trials become holds. Later nearby replacement IDs can be reacquired;
the replay does not establish their physical identity, or a counterfactual
closed-loop trajectory. On the Mac the association-only replay median/p90 was
0.115/0.136 ms; this is not a live Arc end-to-end timing result.

Replay artifact: `work/control-tuning.8q0fam/target-lock-replay.json`.
Deployment rollback: `/home/spring/.local/share/turret-demo/control-hold-deploy-20260908.C8Gm6v`.

### Live follow check: natural loss held; full recovery test incomplete

The real-worker check selected a light detection at the unchanged 0.65 test
threshold. It disappeared during the initial centering attempt. Of 32
control-aligned results, one was pre-selection/waiting, three followed ID53,
and 28 held the same identity rather than choosing another light. The last 25
held samples had exactly constant reported X/Y positions **and** goals. There
were no servo errors or read retries.

The harness correctly failed its initial-settling requirement, so none of the
planned injected-gap/ID-change/outlier phases ran. This is evidence of natural
loss holding, **not** a passed complete live reacquisition qualification. The
original failure and images are retained under `work/control-tuning.8q0fam/hold-live-first/`.
Reacquisition is covered by synthetic/controller tests and recorded-ray replay;
successful live reacquisition in this particular run is not claimed.

The normal service is restored to `sam3.1-mask` / `person`, tracking off, original
goals X44.3/Y27.42 and the original armed state. The Arc-local unified frontend
and Firefox kiosk were restarted to load the updated UI; Thor's backend was not
restarted or altered. Its route remains wired via enp7s0 / 192.168.249.2.
Final verification: backend142861 / worker143398, advancing frames2–159, model
latency70–74 ms, packed feedback, zero retries/errors, all three Arc services
active. `hold-restored.json` contains the restoration receipt. Configuration
comparison confirms that only `tracking.hold_reacquire=true` changed. No new
GPU fault/reset was found in the checked deployment/recovery interval.

Installed hashes: tracking `30ffa089f5041dc995464bfa5c0856dd64e3a119e8e9beba1a4cd1b8e1a46afe`,
target lock `9fb202ec49c716858175d03baf1789732bdfb22d6a28cc1e3390740ebd41ab62`.
Servos and geometry retain the previously qualified hashes below.

## Stationary-light follow qualification (2026-09-08)

After the occupied-chair runs, a no-motion prompt survey tested 24 frames each
of `cabinet`, `shelving unit`, `desk`, and `ceiling light`. Only the light prompt
consistently exceeded the unchanged 0.65 test-selection threshold. The ordinary
runtime detection threshold remained 0.5. The survey restored `person` and did
not issue any motor commands.

Twelve alternating, eight-second trials then compared the existing unsmoothed
controller with the optional world-ray filter (minimum cutoff 0.5 Hz, speed gain
40). Both used packed feedback, the same compiled mask worker, gains, calibrated
geometry and starting offsets (0/0, +6/0, 0/+6, repeated twice). All twelve trials
completed: **960 control-aligned samples**, no servo faults or read retries, no
initial target-matching rejections. No filter was enabled in the deployed config.

The first ten trials retained the same light instance. In their last four
seconds, all motor goals were constant; nine had exactly constant reported
encoder positions, and one toggled by a single encoder count on Y. Unfiltered
trial-median centroid errors were 0.94–3.11 px, with trial p90s 1.30–3.63 px.
Filtered medians were 0.66–3.54 px, p90s 1.01–3.82 px. Four paired medians improved
by 0.12–0.87 px and one worsened by 0.42 px. Acquisition below 10 px took about
0.49–0.56 s; filter effects ranged from 27 ms faster to 42 ms slower in these
five pairs. This does not establish a consistent, worthwhile smoothing win.

An independent check fitted camera rotation to mutual SIFT image features,
then transferred a fixed reference point without using each frame's SAM
centroid. For the first ten trials, sparse late-image point ranges were below
0.15 px on each axis. This supports a steady camera, not a claimed 0.15-pixel
absolute accuracy or high-frequency vibration bound. The proxy reference was
chosen from one mask observation, not a labeled physical light centroid. Across
accepted saved images, encoder/visual-pose disagreement had median 0.021 degrees
and max 0.179 degrees. Sparse images do not qualify peak-motion exposure timing.

### Remaining failure is target/measurement continuity, not cured by smoothing

In the last two trials, detections of the selected light disappeared and the
non-temporal policy immediately chose other available instances. The first
switches occurred at 2.02 s and 4.18 s, with measured world-ray changes of 14.9
and 8.8 degrees. The original ID was absent in those frames. The filtered run
also had a 10.2-degree centroid-ray change while its original ID still existed:
an ID alone does not guarantee a stable mask measurement. People moving through
the scene/occluding fixtures are visible in the saved images; the light fixtures
themselves remained stationary.

The independent fixed-point check confirmed that the camera subsequently aimed
away from the original light. Both smoothing modes failed continuity. Their
late reported centroid p90s of 21/62 px and 77/133 px maxima are retained, not
discarded or described as successful same-object tracking. Runtime `error=null`
does not mean this behavior is correct for a persistent-target use case.

The then-current non-tracking selection policy deliberately resumed nearest-of-class
when an instance disappears (and releases a click's temporary preference after
centering). The user subsequently approved retaining/reacquiring an object
through misses; see the policy and qualification above. That is a separate choice
from servo/PID tuning. No smoothing, model or confidence change was deployed.

Artifacts under `work/control-tuning.8q0fam/`: `static-survey/`,
`filter-light/filter-ab.json`, `filter-light/analysis.json`,
`filter-light/visual-steadiness.json`, and `filter-light/target-discontinuities.json`.
The analysis now emits null for ratios against zero movement, not NaN or ratios
of floating-point roundoff. The final normal-service restoration reported backend
PID114296/worker PID114513, `sam3.1-mask`/`person` running at 70–71 ms, packed
feedback, motors armed/error-free, tracking off, goals X44.3/Y27.42, and unchanged
zeros/limits. No new GPU fault/reset appeared in the checked recovery window.

Read-only follow-up: the attached USB device is 1a86:55d3 using `cdc_acm`.
[WCH's driver source](https://github.com/WCHSoftGroup/ch343ser_linux/blob/main/driver/ch343.c)
identifies that product as the CH343 family. This is not a qualified end-to-end
high-baud link test. No serial baud rate, driver, or servo EEPROM was changed.

## Latest increment: fresh packed feedback (2026-09-08)

Arc now enables `servo.packed_feedback = true`; other installations retain the
default individual-register path. Eight RAM-only indirect aliases combine each
servo's signed position, torque state, hardware fault, status-return-level
sentinel and bus watchdog into one acknowledged read. There is still a **new
read of both axes before every command**, not a cached health check. The SDK's
device-error byte is checked. No baud, EEPROM, speed, current, geometry, gains,
zeros, model or angle-limit change is part of this increment.

The [XL330 control table](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/)
has different indirect-data bases before/after firmware V53. Both physical
servos report V53. Setup verifies both models and torque-off states, saves the
previous mappings, writes/read-verifies each RAM alias, and validates the full
mapping again before arming. Stop/fault cleanup attempts to restore both saved
mappings, reports restoration failures, and never writes an unexpected model.
A reset/invalid-map sentinel, torque mismatch, device/hardware fault or armed
watchdog mismatch rejects the command and clears pose history. Lost/corrupt
reads retain the existing single retry; retried positions do not enter history.

A proposed cached-read shortcut was rejected before deployment: fault injection
showed it could permit a command after a new read failure. The original failing
test was preserved. Twenty additional packed-path tests cover fresh-command
reads, either-axis faults before either goal/limit recovery, packet errors,
short/invalid payloads, retries/history, signed positions/timestamps, mixed
firmware, partial/ineffective setup writes, reset while armed/unarmed, and
restoration including changed models. The full existing validation suite passed
(its two dependency-gated mask-output tests remain skipped).

Initial torque-off hardware probe: 80 alternating comparisons, no encoder
differences, no errors, all saved aliases restored. Median complete feedback
read: **33.34 -> 13.19 ms**.

Then an exclusive-port **ABBA** motion test alternated legacy/packed/packed/legacy
with unchanged P=1200/I=0/D=1600. Each block used 13 poses: +/-6-degree single-axis
steps, +/-6-degree diagonals and returns around X44.3/Y27.42. The candidate was
not installed into the normal backend until this test passed.

| Measurement | Legacy | Packed |
| --- | --- | --- |
| Median command time, two blocks | 39.09 / 39.24 ms | 19.12 / 19.22 ms |
| Median encoder sample interval | 43.46 ms | 23.33 ms |
| X/Y steps settled within 0.5 degrees | 16/16 each | 16/16 each |
| X/Y median sampled settling | 175 / 149 ms | 139 / 128 ms |
| X/Y median absolute settled error | 0.085 / 0.045 deg | 0.085 / 0.045 deg |
| X/Y maximum sampled overshoot | 0.091 / 0.086 deg | 0.261 / 0.086 deg |

All four blocks completed without servo errors/retries and restored their
starting mappings. Faster sampling changes settling/peak observability: these
are sampled measurements, not proof that physical overshoot improved or that
the whole observed settling difference is real. The reliable latency result is
about **20 ms less command overhead**. Motor gains were identical.

Artifacts: `work/control-tuning.8q0fam/packed-motion/motion.json` and
`motion-summary.json`. Rollback configuration and installed/source servo modules:
`/home/spring/.local/share/turret-demo/control-packed-deploy-20260908.oOjpt9`.
Real-inference deployment verification: 177 atomic processed JPEG/metadata
pairs over 11 bounded API-commanded poses. All 177 had interpolated historical
encoder poses, none missing; median/p90 bracket spans were 23.38/23.51 ms.
Median reported model latency remained 72 ms. No servo faults or retries.
The active calibration produced median/p90 frame-median feature reprojection
errors of 0.615/1.518 px (all frames) and 0.592/1.401 px (74 strictly settled
frames); no frames failed the mutual-feature-match count gate. This is an
independent frame/pose-alignment check, not an automatic-following accuracy
claim. Artifacts: `packed-live/`, `packed-live-summary.json`, and
`packed-live-validation.json` in the same staging directory.

Installed/source `servo.py` SHA-256:
`3d711ea2837a9beda91c385242bd5eb57457f4b8d13899e82a89dc1b13984d8b`.
The normal backend was restored to `sam3.1-mask`, `person`, tracking off, motors
armed, and goals X44.3/Y27.42 before subsequent automatic-follow experiments.

### Follow comparison: incomplete, scene changed

A no-motion 30-frame preflight detected an empty chair in every frame. During
the subsequent worker startup it became occupied and moved, as the trial
selection JPEGs show. Two eight-second trials completed (90 legacy / 88 packed
control-aligned samples); the next offset never reacquired the reference chair
within the unchanged 3-degree identity check and the experiment stopped. The
later available chair was 31 degrees from the reference. Do not weaken that
check or count the planned twelve-trial comparison as completed.

Late-window observed capture-to-command medians were 160.6 ms legacy / 130.0 ms
packed, and control-tick medians 45.1 / 18.9 ms. These observations are consistent
with the isolated overhead reduction, but different target motion and command
activity prevent attributing all of the difference to the read path. The lower
candidate endpoint/jitter errors are **not an accuracy A/B result**. No bearing
filter was used or enabled: the harness's legacy boolean `filter` field denotes
the candidate path when `comparison = feedback`.

The first harness launch also failed before any trial due to a servo-only
candidate package shadowing the full application. Explicitly selecting the
installed package fixed the harness; product code was not changed for that.
The stopped experiment returned home, quiesced its worker and restarted the
normal service. Receipts/images and explicitly rejected analysis are retained
under `work/control-tuning.8q0fam/feedback-follow/`.

Final restoration verified after this experiment: normal backend PID104317,
mask worker PID104445, `sam3.1-mask`/`person` running (frame329, 72 ms at the
check), tracking off, motors armed/error-free, zero retries, packed feedback,
goals X44.3/Y27.42 and unchanged zeros/limits. Frontend and kiosk services are
active; Thor still routes through `enp7s0` to 192.168.249.2. No new xe fault/reset
was reported during this test/recovery window. This is not a fix or qualification
of the earlier GPU teardown fault. Stationary-jitter and moving-target accuracy
qualification remain open; no smoothing or guessed near-field range was enabled.

## Hardware and baseline

2026-09-08: `spring-edge-turret`, USB controller `5B3D045331`, XL330-M288-T
servos X=2/Y=1, camera `0c45:0261:UC684`, 1280x720 at 30 Hz. Saved zeros
remain X=1005/Y=22. Angle limits remain +/-90 degrees. No EEPROM, baud,
current, profile-velocity or profile-acceleration changes were made.

Actual register readback on both axes: P=400, I=0, D=0, first/second
feed-forward=0. Thus the installed inner loop was P-only, even though the
firmware supports PID and feed-forward. The backend computes automatic angle
setpoints; browser controls do not drive the automatic correction loop.

Each initial comparison used 25 poses: +/-3 and +/-12 degree single-axis
steps, four +/-8-degree diagonal steps, and returns to the local start pose.
There are 16 nonzero steps per axis. Encoder reads were timestamped individually;
settling means all remaining measured samples within 0.5 degrees through the end
of a 1.2-second step. The roughly 43 ms feedback interval limits temporal and
peak-overshoot resolution. These are measured samples, not high-bandwidth
oscilloscope bounds. Gains were restored after each experiment.

| Register gains P/D (I=0) | X/Y median absolute settled error | X/Y max sampled overshoot | X/Y median settling |
| --- | --- | --- | --- |
| 400/0 baseline | 0.352 / 0.306 deg | 0.093 / 0.263 deg | only 8/16 and 9/16 steps reached 0.5 deg |
| 800/0 | 0.131 / 0.091 deg | 1.492 / 1.410 deg | 221 / 214 ms |
| 800/100 | 0.086 / 0.134 deg | 1.142 / 1.057 deg | 194 / 170 ms |
| 800/400 | 0.088 / 0.088 deg | 0.613 / 0.529 deg | 173 / 151 ms |
| 1200/800 | 0.046 / 0.087 deg | 0.529 / 0.617 deg | 134 / 151 ms |
| 1200/1600 | 0.129 / 0.002 deg | 0.085 / 0.090 deg | 179 / 152 ms |

The baseline worst settled errors were 0.877 deg X and 0.967 deg Y. All listed
candidates after baseline settled all measured steps within 0.5 degrees.
Command calls cost about 39 ms for one-axis updates and 45 ms for both axes,
including the controller's synchronous feedback checks. This is separate from
model inference time.

## CAD geometry

Source: [author's STEP repository](https://github.com/AnthonyZJiang/dynamixal-pan-tilt-camera-cad/tree/c3b8d3f714c613a7871ae5e679be1cb5761e76f9),
commit `c3b8d3f714c613a7871ae5e679be1cb5761e76f9`. Both standard and lite STEP
assemblies were read as named XCAF assemblies, retaining occurrence transforms.
The shortest separation of the two horn axes is **44.5 mm in both variants**;
they do not intersect. The lens is also displaced from the tilt axis.

Current runtime geometry uses calibrated fisheye rays, camera mounting rotation,
and coupled `Ry(pan) Rx(tilt)` rotations. It does not model translated camera
centers or finite-distance parallax. CAD geometry alone cannot supply an unknown
target distance or the lens's entrance-pupil position. Do not silently substitute
an arbitrary distance and label it exact geometry.

The saved calibration's earlier +/-5-degree qualification is not full-range
qualification. A new 951-frame recording extends the local test to +/-12 degrees
around pan 44.3/tilt 27.4. On 152 settled frames, the median per-frame feature
reprojection error was 0.663 px (90th percentile of frame medians: 2.430 px).
These metrics differ from the old per-correspondence calibration metrics.

## Timing and visual-loop tests

The same recording measured GStreamer frame delivery about 32.1 ms after its
buffer timestamp. Fitting independent static image features during movement
put the best visual pose about 25 ms after that timestamp. This is evidence
against blindly subtracting the entire 32 ms from JPEG receipt time: exposure,
sensor readout and driver timestamp semantics need separate accounting.

The baseline production tracking controller was also exercised with a stationary
image feature in place of segmentation, with 90 ms simulated inference delay.
Six offsets (6/0, 0/6, -6/-6, 10/-8, -10/8, 0/0 degrees) ended at mean-last-five
errors 1.77, 2.64, 9.20, 7.24, 12.17 and 6.06 pixels respectively. This exposes
the original 1.2%-of-frame deadband; it is not a segmentation-model benchmark.

One initial visual-harness attempt encountered a transient bus disconnect before
its first trial. The repeat completed all six trials. Do not count the failed
attempt as a passing hardware test or hide intermittent communication failures.

With P=1200/I=0/D=1600 and deadband=0.003, a complete six-offset visual run
ended at mean-last-five errors 1.24, 0.70, 1.91, 1.93, 1.77 and 1.78 px.
Its reference feature differed from the first baseline run, so this is not a
strict paired A/B comparison. A subsequent identical-reference comparison
completed the baseline's six trials (6.24, 4.98, 13.15, 3.56, 4.68, 4.17 px)
and the candidate's first three (0.86, 0.41, 1.93 px), then rejected its feature
measurements. Exposure/scene-change and ambiguous patch-matching failures are
retained as failed/partial trials; they do not establish a full six-trial paired
win. Another NCC-gated repeat also rejected its feature. Broader validation
requires a more robust reference feature measurement, not relaxed pass criteria.

## First deployed increment

Arc only: `servo.axes.{x,y}.position_gains = {p:1200,i:0,d:1600}` and
`tracking.deadband = 0.003` (3.84 px horizontally, 2.16 px vertically at 720p).
Gains are written to RAM and read back before torque is enabled; unspecified
gains on other installations remain unchanged. The code includes one retry of
the same idempotent register read only for SDK receive-timeout/corrupt-response
errors. Device faults and writes are not retried; persistent read failures still
disarm. A read-retry counter and exact error codes are exposed for diagnosis.

Rollback copies of the previous Arc config and both installed/source servo
modules: `/home/spring/.local/share/turret-demo/control-deploy-20260908.cpKxMW`.
The geometry file, zeros, +/-90 degree limits, model bundle, Thor backend and
wired frontend routing are unchanged. This increment is locally tested, not
full-workspace or full-angle commissioning.

The encoder interpolation helper was subsequently integrated and deployed;
see the second increment below. Camera timestamps still mean JPEG receipt,
not independently qualified hardware exposure times.

After deployment, the unchanged mask worker again exceeded the backend's
120-second startup deadline and was terminated while CPU-active. The ready-wait
allowance was raised to 600 seconds in the live module and repository; per-frame
inference deadlines and the model are unchanged. The old detection module is
also in the rollback directory. Model-ready/processed-frame restoration must
be verified; a live PID alone is not sufficient.

Restoration verified: backend PID 49107, model PID 49354; `sam3.1-mask` with
prompt `person` returned to `running`, frame sequence 473 at the first check,
73 ms reported model latency, camera online, servos armed/error-free and zero
read retries. Target selection remains off, matching the pre-test state.
Arc frontend/kiosk services remain active; Thor traffic remains routed through
`enp7s0`, source `192.168.249.1`, destination `192.168.249.2`.

An attempted real-model static-object follow test was not started: both `chair`
and `table` prompts produced zero boxes/masks in the current view. No target was
selected and no automatic movement was made for this test. Prompt `person` was
restored. This is an unresolved limitation of end-to-end qualification, not
evidence that model-driven tracking passes. A later repeat did detect chairs
and the live model follow test below supersedes that temporary limitation.
Do not substitute the feature harness for model/control-loop qualification.

## Second deployed increment: historical pose and bounded pointing

Each axis's position read now has its own midpoint timestamp and transaction
duration. A position read that needed a retry is excluded from pose history,
because the whole retry interval does not identify the successful sample time.
The history retains 256 checked polls. Detection resolves its historical pose
after inference: interpolation requires bracketing measurements on each axis,
with matching identity/zero/direction and no more than 100 ms between reads.
It never substitutes the current angle or extrapolates. Stop and faults clear
history. Fast inference without brackets can use the prior checked snapshot.

An offline replay against independent SIFT-derived visual poses used 307 frames,
including 34 moving frames. Moving-frame median angular disagreement improved
from 0.244 to 0.206 degrees; p90 improved from 1.466 to 1.345 degrees. The old
poll-completion reconstruction optimistically excludes torque/fault-read time.
No time shift or calibration was fitted for this comparison. This modest result
does not resolve exposure/readout timing or near-field parallax.

A live `sam3.1-mask` run at P=1200/I=0/D=1600 recorded 183 processed frames over
11 API-commanded poses (returns and +/-6-degree single-axis/diagonal steps).
All 183 had interpolated poses, none missing; no control faults. Median bracketing
span was 43.63 ms, pose lookup 0.31 ms, and reported model latency 72 ms.
Independent image-feature checks accepted 180 frames (three failed fit-quality
checks). For 34 moving frames: median pose disagreement 0.126 degrees, median
per-frame reprojection 1.028 px, p90 3.701 px. Settled frames: 0.075 degrees,
0.808 px median, 2.181 px p90. This validates image/pose association through the
real inference worker, not automatic target following by itself.

The old inverse-kinematics solver selected the nearest unrestricted branch, then
let the servo clamp it. A 150-case synthetic audit found 23 cases where a bounded
optimum reduced angular error by more than one degree. The worst case clamped to
(-90,-90) with 74.8-degree pointing error despite an exact reachable alternative.

The new solver maximizes ray alignment within the configured joint rectangle.
It enumerates exact solutions, singular configurations, edge extrema and corners;
periodic equivalents and calibration zero/direction transformations are included.
Equal-error solutions prefer less joint travel. Unreachable targets receive the
closest reachable direction and `limited` status. Hardware angle enforcement
remains independent. This is still a rotation-only geometric model.

All 150 original cases and 600 additional randomized directions/mounts/bounds
matched or exceeded a multi-start SciPy optimizer (largest cosine shortfall
2.22e-16). Sixty more randomized unit cases dominate a 13x13 grid in each domain;
tests also cover poles, periodicity, alternate branches, reversed directions,
zero shifts and unreachable targets. Average solver cost on the Mac was 0.0113 ms;
this is not a measured Arc end-to-end speed claim.

Rollback copies:
- Timing: `/home/spring/.local/share/turret-demo/control-timing-20260908.boIs9f`
- Bounded geometry: `/home/spring/.local/share/turret-demo/control-geometry-20260908.rvjdVz`

Artifacts: `pose-timing-comparison.json`, `live-timing/live-timing.json`,
`live-timing-analysis.json`, `joint-limit-audit.json` and
`bounded-pointing-validation.json` in `work/control-tuning.8q0fam/`.

### Recovery during verification

The first automatic chair-follow attempt was interrupted before it selected a
target (zero trials). Kernel logs recorded `xe` engine memory faults at 17:26:51,
a GPU reset at 17:28:03, and a timed-out job in model PID 59109. That process
remained CPU-active inside native weight preparation after its GPU context was
reset. CPU activity alone did not mean loading was making progress.

The waiting test was explicitly interrupted; its `finally` restored the original
pose and `person` prompt. The backend was then restarted after the completed
driver reset, without rebooting the host or changing model/runtime/driver versions.
New worker PID 67695 progressed past weight preparation into compilation, with
no new kernel faults at that check. Full running/follow-test verification follows;
this recovery is not proof that the underlying driver/native teardown bug is fixed.
The failed test receipt remains at
`/home/spring/.local/share/turret-demo/control-model-follow-20260908/model-follow.json`.

### Actual mask-model follow tests after recovery

Worker 67695 completed compilation/capture and all three six-second chair-follow
trials completed without a control fault. Only model/class/instance selection
was sent over HTTP during following; the backend computed the angle commands.
Starting offsets were (0,0), (+6,0), and (0,+6) degrees relative to 44.3/27.42.
The tests included measured tilt travel down to -7.38 degrees, about 35 degrees
from the original pose. This is not a full +/-90-degree qualification.

| Starting offset | Frames observed | Initial centroid error | First error below 10 px | Final reported error | Last-two-second median / max error |
| --- | --- | --- | --- | --- | --- |
| 0 / 0 deg | 68 | 229.6 px | 1.04 s | 7.07 px | 13.02 / 20.72 px |
| +6 / 0 deg | 70 | 248.2 px | 0.82 s | 6.17 px | 5.65 / 17.35 px |
| 0 / +6 deg | 71 | 272.2 px | 0.74 s | 6.01 px | 10.24 / 17.55 px |

Do not report the final six-to-seven-pixel endpoints as a sustained accuracy
bound: residual jitter reached 21 px, and the first trial included two sampled
`lost` states. HTTP tracking telemetry can repeat between processed frames;
these statistics use observed status snapshots, not a high-rate servo trace.
The object was a real chair, partly occluded by a table support; changing mask
extent changes its visible-mask centroid. Separating segmentation noise,
visibility/parallax and control lag is the next investigation, not an assumption
that all residual error is a gain problem.

Receipt and sample images: `work/control-tuning.8q0fam/model-follow/`, copied from
`/home/spring/.local/share/turret-demo/control-model-follow2-20260908/`.
Restoration verified afterward: `sam3.1-mask`, prompt `person`, running at 72 ms
reported model latency, tracking off, servos armed/error-free, zero read retries,
goals X=44.3/Y=27.42, zeros X=1005/Y=22, unchanged +/-90 limits. Both frontend
and kiosk remained active; Thor still routes through `enp7s0` to 192.168.249.2.
No new GPU timeout/reset was found in the four-minute recovery/test window.

## Mask-noise isolation and translation experiment

A fixed-pose run (X=35.5/Y=-6 degrees, target off) recorded 160 processed
chair-mask frames. Both reported encoder positions were unchanged throughout.
Nevertheless, the nearest chair's mask centroid spanned 24.0 x 26.6 pixels
(standard deviations 4.81/4.40 px). Thus not all remaining pointing jitter is a
motor-gain problem. This is a partly occluded chair under the current scene and
exposure conditions, not a claim about intrinsic model randomness or every class.
Receipt and exact processed JPEGs: `work/control-tuning.8q0fam/static-mask/`.

The CAD-derived translated camera-center hypothesis was tested offline against
independent image features from the 183-frame timing run. It uses the measured
44.5 mm pan/tilt axis separation and three assumed lens-pupil offsets: 38, 45,
and 56 mm. The entrance pupil is not physically measured. Inverse depths for
631 multiply observed scene features were fitted using +6 pan, +6 tilt, and
+6/+6 diagonal poses; -6 pan and -6 tilt were held out.

Results were mixed. With the 38 mm pupil hypothesis, held-out pan median
reprojection improved from 0.937 to 0.852 px (400 points), but held-out tilt
worsened from 1.150 to 1.196 px (379 points). The other pupil hypotheses showed
the same qualitative tradeoff. No guessed target distance or translation
correction was enabled. `cad-parallax-validation.json` retains all hypotheses,
including worse training and held-out results. Translation remains unresolved;
CAD dimensions alone do not qualify a finite-distance pointing model.

A second analysis jointly refitted the camera intrinsics/mount and landmark
inverse depths, comparing rotation-only and 45 mm CAD-offset models using the
same correspondences and training/held-out views. The translated model reduced
training residuals and held-out pan median (0.937 to 0.624 px), but worsened
held-out tilt median (1.150 to 1.607 px). Its focal/principal-point parameters
also reached declared fit bounds. Rotation-only refitting worsened held-out tilt
too. Neither fit was deployed. `translated-calibration-refit.json` records both
fits, bounds/convergence indicators and all view-level errors; broader independent
pose data is needed before interpreting a smaller training loss as calibration.

## Adaptive world-bearing filter candidate

The optional estimator filters unit target rays after camera motion has been
removed with calibrated kinematics; it does not filter moving image coordinates.
It uses a shared, speed-adaptive cutoff across vector components, followed by
normalization, inspired by the [1 Euro filter](https://gery.casiez.net/1euro/).
Initial acquisition, explicit retarget, gaps over 0.5 seconds, and angular jumps
over 10 degrees pass the full measured direction immediately. Duplicate or
out-of-order samples do not advance filter state. No motor speed cap, image
modification, or mask change is introduced. Public centering error remains the
**raw detected centroid error**, never the filtered error.

The candidate is `min_cutoff_hz=0.5`, `speed_gain=40`. An offline split of the
fixed-pose recording (80 train/80 held-out frames, discarding ten warmup samples
in each half) reduced held-out bearing RMS from 0.720 to 0.551 degrees. The
reference is the recording's mean ray, not an independently measured physical
ground truth. A noiseless 90-degrees/second synthetic unit test has less than
0.3 degrees added lag error; this is not moving-object hardware qualification.
See `bearing-filter-sweep.json` for all candidates and synthetic-motion results.

The candidate was installed for live comparison with a rollback at
`/home/spring/.local/share/turret-demo/control-filter-20260908.vb266f`.
The alternating live comparison did **not** qualify the filter:

- Attempt 1 completed one unfiltered trial, then rejected the next acquisition
  because the chosen chair's world direction differed by 6.93 degrees. Saved
  images show that the chair was occupied and moving. The completed trial had
  changing mask extents/identity and cannot be used as a stationary jitter test.
- Attempt 2 sought a fixed tool chest, and attempt 3 a fixed table leg. Neither
  produced a qualifying detection within the 30-second ready-model search window;
  zero automatic follow trials ran in those attempts. No thresholds were relaxed
  to turn those failures into passes.
- The revised harness records exact detection/control pairs in the backend,
  retains partial/failed receipts, and can alternate both modes within one GPU
  worker lifetime. Matching acquisitions must remain within three degrees of the
  original target's world bearing. It is ready for a suitable test scene.

Artifacts on Arc: `control-filter-ab-20260908`, `control-filter-ab2-20260908`,
and `control-filter-ab3-20260908` under the application data directory. The first
run is also copied to `work/control-tuning.8q0fam/filter-ab-rejected/`.
**The filter was disabled again in the live configuration.** The helper and
tests remain available, but offline noise reduction is not claimed as a measured
closed-loop win. The previously validated gains, history interpolation, bounded
solver, saved zeros and camera calibration are retained.

Each mask worker was quiesced with empty prompts and confirmed sleeping before
shutdown. No new `xe` reset/timeout was observed over these three attempts. This
is an operational workaround, not a fix for the earlier driver/native teardown
failure. The complete local validation suite passed, with the two existing
Torch-dependent skips; no tests were disabled for this experiment.

## Broader camera-refit qualification and live trial

The smaller positive-motion-only calibration fit above did not generalize. A
subsequent fit uses balanced +/-3-degree single-axis and +/-8-degree diagonal
views from the earlier 951-frame sweep. All four +/-12-degree single-axis views
are held out. No frame from the separate 183-frame real-model timing recording
is used to fit the camera parameters.

The rotation-only refit improves all 23 settled, sampled held-out +/-12-degree
frames: median of frame medians 2.160 to 1.307 px; p90 of frame medians 2.994 to
1.732 px. Across 71 settled frames in the separate +/-6-degree recording, median
of frame medians is nearly unchanged (0.467 to 0.464 px), while p90 improves
1.974 to 1.193 px. 48/71 frames improve; the largest individual frame-median
increase is 0.0064 px. Repeated returns to the reference pose dominate the median.
Settled means more than 650 ms after completion of the most recently **started**
command; a new command can begin moving X before its later Y write completes.
The initial analysis accidentally included some such transitions as settled;
these corrected counts replace it. Reference selection also differs from the
earlier timing analysis, so their aggregates are not interchangeable.

Regression check against the original commissioning images near physical zero:
4,206 held-out feature correspondences have median 0.639 to 0.749 px and p90
1.396 to 1.463 px. This small local regression is explicitly retained as a
tradeoff, not hidden by the larger-pose improvements. The candidate still passes
the original geometric qualification thresholds against the legacy linear
mapping. Old qualification metrics were not copied onto the new calibration.

The fitted lens parameters are fx=528.791, fy=526.795, cx=652.591, cy=351.892,
k1=-0.0276526, k2=-0.00317360, with a jointly fitted camera-mount rotation.
No servo zeros, mechanical directions, IDs, gains, angular limits or model
weights change. The runtime still uses rotation-only rays and bounded IK.

The joint translated-camera variant also improved the larger sweep when fitted
on balanced views, but it needs per-feature inverse depth to realize that gain.
Its camera parameters alone do not consistently outperform the rotation-only
fit. Therefore those parameters and assumed 45 mm entrance-pupil offset were
not activated in the live depth-free controller.

The exact runtime ray and bounded-pointing code passes 1,000 randomized pixel/
pose cases with the new lens file: 715 reachable, 285 limited, maximum ray
round-trip error 1.20e-8 px, maximum reachable centering error 8.98e-9 px. These
are numerical consistency checks, not physical accuracy measurements.

Arc calibration trial deployed with rollback:
`/home/spring/.local/share/turret-demo/control-camera-refit-20260908.6yTcbG/geometry.json`.
The previous file was SHA-256 checked before replacement; camera/controller
bindings are unchanged. The filter remains disabled.

### Fresh post-deployment check

The first fresh sweep collected 84 frames before the recorder hit HTTP 404 while
fetching a frame after its separate status request. It returned to the original
pose; this partial run is retained at `control-refit-live-20260908`. The recorder
was changed to consume the existing atomic JPEG/metadata event stream instead
of racing a bounded image cache. No backend/frame retention policy was changed.

The repeat completed all 11 commanded poses and recorded 176 processed frames
without a control fault or recording error. No parameters were fitted to this
recording. Mutual SIFT matches exclude detected-person boxes (with a 10 px
margin); all 176 frames had enough matches, with none rejected.

Across all frames, median of frame reprojection medians improves 0.854 to 0.588
px, p90 improves 2.361 to 1.442 px. For 70 strictly settled frames, median changes
0.585 to 0.570 px and p90 improves 2.127 to 1.375 px. The largest settled-frame
median increase is 0.160 px. Nonzero-pose median errors:

| Offset X/Y | Previous calibration | Refit |
| --- | --- | --- |
| +6 / 0 deg | 1.178 px | 0.631 px |
| -6 / 0 deg | 0.906 px | 0.572 px |
| 0 / +6 deg | 2.112 px | 1.372 px |
| 0 / -6 deg | 1.027 px | 1.002 px |
| +6 / +6 deg | 2.351 px | 1.430 px |

The -6 tilt pose has a small p90 regression (1.076 to 1.178 px); this is not a
uniform improvement claim. These are geometry/frame alignment measurements,
not mask-centroid tracking error or an automatic-follow acquisition benchmark.
The refit remains enabled based on the broader independent checks; finite-depth
parallax and moving-target performance remain open.

Restored and verified: backend 85294/model worker 85319, `sam3.1-mask`, prompt
`person`, running at 72 ms reported model latency at the check; servos armed,
tracking target off, goals X=44.3/Y=27.42, zeros X=1005/Y=22, no current servo
error/read retries. Frontend/kiosk remain active, and Thor still routes through
`enp7s0` to 192.168.249.2. No new GPU reset/timeout appeared in the checked window.

Artifacts in `work/control-tuning.8q0fam/`: `translated-calibration-baseline-refit.json`,
`cross-record-rotation-refit.json`, `cross-record-cad-refit.json`,
`refit-full-recording-validation.json`, `geometry-refit-candidate.json`,
`fresh-refit-validation.json`, and `refit-live/` (fresh exact model-frame pairs).

## Remaining qualification

- Reduce and qualify residual model-to-control centering jitter. The first real
  mask-follow tests now pass initial acquisition, not sustained sub-pixel or
  low-jitter following.
- Qualify camera exposure/readout timing beyond the deployed per-axis historical
  interpolation; delayed inference no longer freezes the pre-submission pose.
- Validate translated-camera geometry, including the limits of monocular depth
  observability. Exercise the mathematically tested joint-limit solver across a
  wider physical envelope before calling it full-range commissioned.
- Test wider poses, moving targets, overshoot/settling, and the real model/backend
  workload; retain independent held-out trials.
- Investigate graceful GPU-worker teardown and the observed Intel driver reset;
  process recovery is not a root-cause fix.
- Deploy only measured improvements, verify saved zeros/limits and running state,
  and retain a rollback of code and configuration.

Temporary experiment programs and receipts currently live in the workspace's
`work/control-tuning.8q0fam/` and on the Arc under
`/home/spring/.local/share/turret-demo/control-*-20260908/`. Measurements above
are reproducible from the JSON encoder/frame/command logs; they do not establish
that the full tuning objective is complete.

Register meanings and units: [ROBOTIS XL330 manual](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/#position-pid-gain80-82-84-feedforward-1st2nd-gains88-90).
