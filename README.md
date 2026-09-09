# Spring turret demo

Camera feed, switchable SAM 3.1 / SAM 3.1 Mask / SAM 3.1 Tracking, and manual or opt-in automatic
X/Y camera framing.

For the current **spring-edge-turret (Arc) + agxthor-5** pair, start with the
[reproducible deployment and recovery guide](docs/reproduce.md). It pins the
actual runtimes, private model artifacts, physical calibration, wired networking
and boot services. Older host-specific qualification notes below are historical.

Optional qualified native SAM image execution from `sleepy-joe` is documented
in [the native image setup](docs/sam31-native.md). Text/grounding and the default
compiled image path remain available; activation is an explicit device setting.
The faster, **accuracy-unqualified** Israel W8A8 candidate has a separate
explicit [development setup](docs/sam31-w8a8.md); dense-reference gates remain unchanged.
The approved [sleepy-joe W4A4 setup](docs/sam31-w4a4.md) provides the retained
67.8 ms GPU-model candidate, with its explicitly accepted measured accuracy tradeoff.

The optional [SAM 3.1 Tracking profile](docs/sam31-tracking.md) uses the full
Object Multiplex network, temporal memory and native object IDs on Arc and Thor. It is
dense BF16 by default, with an optional pinned Israel native672 tracking build
on Arc. Both are separate from the fast detector-only profiles. See the
[native tracking tradeoffs](docs/sam31-tracking.md#israel-native672-tracking-opt-in).

NVIDIA Thor uses [dense SAM 3.1 with CUDA compilation and graphs](docs/agxthor.md).
The same camera/servo UI is retained; Intel binaries are not used on Thor.

[SAM 3.1 mask](docs/sam31-mask.md) is a separate per-frame mask detector: the
qualified sleepy-joe native mask recipe on Arc, and the same SAM checkpoint and
mask head through standard `torch.compile` and CUDA graph replay on Thor.
It is not the temporal Tracking profile.

## Current hardware status

- Arducam 1080P Low Light (`0c45:0261`, serial `UC684`): 1280x720 MJPEG at
  30 fps verified on `spring-edge-2`.
- USB Single Serial adapter (`1a86:55d3`, serial `5B61036033`): enumeration and
  stable device naming verified.
- Two ROBOTIS DYNAMIXEL XL330-M288-T servos share one Protocol 2.0 bus at
  57,600 baud: **X/pan = ID 2**, **Y/tilt = ID 1**. The original single-servo
  qualification in `hardware.json` is historical; it does not qualify the
  assembled two-axis mechanism.

Current assembly readbacks and bounded motion results are recorded separately
in [`hardware-pan-tilt.json`](hardware-pan-tilt.json). Stock motor tuning showed
up to 1.86° of residual error in ±3° tests. While running, slider values indicate
the commanded angle; `/api/status` reports measured and goal angles separately.

The service starts torque-off with no startup movement. **Start** writes both
current positions as hold goals before enabling either motor; it never homes
the assembly. **Stop** (or Escape) attempts to release both motors even if one
does not answer. Failed communication or a partial Start cancels motion and
attempts torque-off on both axes. An unconfirmed stop is reported explicitly.
Start intent survives recoverable device faults: reconnection rechecks model,
position mode, drive mode, secondary ID, homing offset, range and hardware faults,
then enables torque at the newly measured position. Stop cancels that intent even
while unplugged. A fresh backend process still starts torque-off. See
[device recovery](docs/recovery.md).

Start stays active when the browser tab is backgrounded, closed or disconnected;
there is no browser-heartbeat timeout. Use **Stop** (or Escape in the focused
page) to release the motors. Service shutdown and hardware faults also attempt
torque-off; restarting the service never re-arms. Each servo retains its one-second bus watchdog:
if the process or bus stops sending traffic it stops motion, **but retains
holding torque**. These controls are not a physical emergency stop. A released
tilt axis can fall under gravity; support the camera before disconnecting power.

## UI

The frontend can run on your workstation while all hardware and inference stay
remote: `node scripts/frontend.mjs --arc http://ARC:8080 --thor http://THOR:8080`.
Open `http://127.0.0.1:8080/` for Arc and Thor side by side, each with the same
independent controls. HTML, CSS and JavaScript are served locally;
API commands and paired camera/detection streams stay scoped to their device.
No Python, GPU runtime or model files are needed on the workstation. See
[local frontend setup](docs/local-frontend.md) for Arc/Thor side-by-side use.

The HTTP viewer is a separate, restartable Python process. The backend owns
camera capture, inference and servo control; a viewer crash/reload/disconnect
does not restart the model, lose tracking memory or stop armed motors. The viewer
reads an atomic latest-snapshot file in private tmpfs and keeps its own bounded
JPEG cache. There is no shared Python lock/GIL, viewer acknowledgment or video
queue that can backpressure control. All HTTP, SSE encoding and network writes
are in the viewer; the backend's single snapshot publisher does fixed work
regardless of viewer count. Only explicit commands cross a bounded private Unix
socket. See [process isolation](docs/process-isolation.md).

Open `http://HOST:8080/`. Below the camera are an **X degree slider**, a central
**triangle/square Start/Stop** button, and a **Y degree slider**. Sliders move
their respective axes while running; requests are coalesced during a drag.
Errors appear only when needed.

The status line shows **detection FPS**, **model** time and full processing
**loop** time (capture wait, pose lookup, worker and result publication; not
network/browser display time). FPS is measured
from completed-frame sequence changes over a rolling two-second browser-time
window, including frames completed between polls. It includes pipeline overhead
and is not `1000 / model_ms`, the camera's capture rate, or browser rendering FPS.
It resets on model/prompt changes and shows `—` during startup/unavailable data.

**Recalibrate zeros** saves the current pan and tilt encoder positions as X=0°,
Y=0°. Stop the motors, support the camera and position it at the intended zero,
then click and confirm. No motion or EEPROM writes occur. Both fresh readings
must be stationary and torque-off; an active motor, missing axis or failed save
rejects the operation. Tracking selection is cleared and Start remains manual.
The numeric angle limits, axis directions and motor settings are unchanged;
the allowed physical travel is now relative to the new zero.

For persistent zeros, set `servo.calibration_file` to a writable state path,
for example `/var/lib/spring-turret-demo/servo-zeros.json`. A systemd service can
provide this directory with `StateDirectory=spring-turret-demo` and
`StateDirectoryMode=0750`. Both zeros are atomically saved in that file and
loaded on restart; no write access to `/etc` is needed. Existing commissioned
zeros remain the fallback until the first save. Invalid saved data or changed
axis IDs/directions fail closed; the button does not replace initial hardware
commissioning (`servo.calibrated` must already be true).

Enter
one object category per row (for example `person`, `cup`, `keyboard`). Each row
shows its detected instance count and a trash button. The single **+** below
the list adds another row. Counts show `—` while unavailable or for unapplied
prompts; `0` means no instances were detected in the current result. Select **Update
prompts** to apply every row together without starting stopped motors. Pressing
**Enter** in a text box applies the prompts and requests Start; Enter elsewhere
also starts the affected panel (both panels from shared controls).
Edits do not change active detection until submitted. Up to eight categories
are supported; blank and duplicate prompts are ignored. Remove/empty all rows
and update to return to the raw feed. Detection alone never moves either servo.

Select the **target icon** beside an applied object class to follow it; only one
class can be selected. Click it again to return to manual control. While **Start**
is active, the camera follows whichever matching aiming point is nearest
the frame center (distance in image pixels, not apparent object size or depth).
A dashed white box previews that instance and the center marker shows the framing
goal. Mask-producing profiles aim at the original mask centroid; box-only models
use the bounding-box center. An empty mask has no aiming point. The preview and
both calibrated axis corrections use the same point. Non-tracking class mode
reconsiders the nearest instance each frame; SAM temporal tracking retains its ID.

**Click a bounding box** to temporarily retarget to that object, including another
instance of the same class. The white dashed outline follows the clicked object
while centering it. Once centered, the original nearest-to-center class tracker
continues automatically. This is an additional retarget control, not a persistent
instance-lock mode. The boxes also
support keyboard focus and Enter/Space. Overlapping boxes prioritize the smaller
box. Click the class icon to return to nearest-of-class mode, or click another box
to switch objects. Selection never starts stopped motors.

Instance IDs use conservative class/position/size matching between detections,
with camera-motion compensation from the encoder/frame pairs. This is not SAM
video tracking or appearance-based re-identification: occlusion, fast movement or
crossing similar objects can lose the association. If the clicked ID disappears,
the original nearest-of-class tracker takes over immediately. With no matching
detections it holds, then reacquires automatically when that class returns; it
does not remain stuck on an expired ID. Updating prompts clears a pending
retarget. The server validates a click against the exact displayed
JPEG's cached detections and rejects unavailable frames or fabricated object IDs.
Click metadata is kept independently of the eight-JPEG cache for the last 128
results, without an age timeout. Pointer presses stay attached to the stable overlay if a box
changes ID before release. Ambiguous old IDs are retired, so one crossing cannot
cause continuous ID churn after objects separate. Rejected-click errors remain
visible for five seconds rather than disappearing on the next video frame.

Selecting a class never starts stopped motors. Stop/Escape still releases both
motors; a manual slider move cancels automatic tracking. Editing/removing the
selected prompt also cancels tracking. No target, a camera/inference fault, or
a prompt change pauses corrections and holds position;
there is no automatic search/sweep. New frames resume tracking while Start is
still active, including when the browser is backgrounded or disconnected.

Tracking has no step-size cap, encoder-to-goal lead cap, or settling delay. Each
new inference result wakes the controller immediately, including frames captured
during a preceding move. Absolute pointing goals are clamped only to the X/Y angle
limits. `inference.max_fps: 0` (the default) runs inference as fast as the pipeline
can process fresh camera frames, without an added FPS throttle. The controller
uses qualified fisheye/servo geometry when `tracking.geometry_file` is configured;
otherwise it estimates the full correction as normalized image error times the
degrees-per-frame scale. Both command **sampled camera angle + correction**.
It does not repeatedly add delayed image errors to the previous goal. Fresh
frames refine that absolute destination; a stationary, unchanged goal allows
learning the small load/stiction holding bias. The 1.2% centering deadband remains.
Reused frames and images from before Start/class selection are still ignored.
There is no maximum age for a completed detection or a box click: slow inference
does not pause corrections solely because its camera frame is old. This can
command an outdated object position; later results provide corrections.
Stop, exact-frame validation, and hardware fault protections
are unchanged. The existing uncapped motor profile registers are also unchanged.
The tracker reports `angle limit` when centering would require travel outside
the configured range. Faster corrections can be more abrupt.

Camera-axis direction must be commissioned separately from the mechanical zero:
add `"tracking": {"calibrated": true, "x_direction": 1, "y_direction": -1,
"x_degrees_per_frame": 161.6, "y_degrees_per_frame": 82.8}` to
the device config **only after checking the assembly**. These signs were measured
on spring-edge-2: +X moves the background left, +Y moves it down. Defaults remain
uncalibrated so another installation cannot move on assumed camera directions.
Those scales are **initial linear estimates**, derived from the earlier small
encoder/phase-correlation measurements (640 × 360): 640 × 2.02 / 8 and
360 × 1.15 / 5 degrees per frame. They are not measured full lens fields of view
or a full optical/gimbal calibration; wide-angle distortion and cross-axis
coupling can leave residual errors that later frames correct. Commission scales
for another camera rather than copying these blindly. `deadband` remains
configurable. Legacy `max_frame_age_seconds` is accepted but ignored. The old `x_gain`, `y_gain`,
`max_step_degrees` and `settle_seconds` settings have been removed.

Inference matches the latest camera frame to a bounded 16-snapshot X/Y history
from the independent hardware monitor, using the newest read completed before
that frame's receipt timestamp;
it does not block on the serial bus. Every published snapshot has passed the
same position, torque and hardware-fault checks for both axes. A JPEG must arrive
after those reads complete and within 100 ms of their start, including bus time.
The snapshot is carried through the model. Missing or stale
pose pairing holds motion rather than guessing from the current motor position.
JPEG receipt time is not a hardware exposure timestamp: bus/camera buffering,
model latency and physical travel still matter. At high speed this pose is an
estimate, with subsequent frames providing feedback. A newer encoder poll no
longer forces the detector to discard a fresh frame and wait for the next one.
It waits for capture only when no unprocessed camera frame is available; no
added motion-settling wait or relaxed pose-age/fault gate is introduced.

SAM's Torch/W8A8 path decodes RGB/grayscale JPEG directly into a tensor, with
pixel parity against the original PIL reference (CMYK/non-JPEG diagnostics
retain PIL). It keeps torchvision's uint8 antialiased resize, then uploads
bytes directly into the graph input and performs float32 normalization with
a compiled GPU lookup table and SYCL replay. This avoids CPU float32 passes and
the four-times-larger float32 upload. All 256 input values map to the original
CPU float32 bits; first-frame exact parity is a hard gate, independent of the
W8A8 development accuracy opt-in. `preprocess_validation` reports that check.
The separate dense native runner retains its CPU-input path. Run
`tests/qualify-sam31-preprocess.py --jpeg /path/to/camera.jpg` in the inference
venv with an idle XPU for exhaustive byte-value and changed-frame parity tests.

SAM advertises `jpeg-bytes-v1` in its worker handshake: a bounded JSON header
with `jpeg_length`, followed by exactly that many unmodified JPEG bytes.
Responses stay JSON lines, and legacy base64 JSON requests remain accepted.
Older workers retain the base64 path unless they advertise the capability.
This removes base64 encoding/decoding and its 33% payload expansion from SAM's
local input pipe; it does not reorder inference requests/results.

The Torch/W8A8 worker overlaps CPU JPEG decode/resize with GPU inference. During
steady, validated prompt batches the parent checks for a new camera frame every
5 ms while awaiting the GPU result, sends each candidate only once, and keeps a
bounded, latest-only preparation buffer, not a FIFO of inference frames. Cold
compilation and changed-prompt setup do not start new background preparation.
On the next pass the controller still chooses the newest camera frame;
reuse requires matching revision, request/sequence, JPEG bytes and the existing
100 ms freshness bound. A newer frame, changed prompt or late preparation falls
back immediately to preparing the actual latest frame. GPU operations remain
on the inference thread; pose pairing, angle limits and motor stop behavior are
unchanged. Set `inference.sam31_cpu_prefetch: false` to disable this optimization.
`preprocess_prefetched` reports reuse; `timing.cpu_prepare_ms` measures CPU work
even when overlapped, so it must not be added again to `worker_total_ms`.

### Pattern-free fisheye calibration

Version 0.14 supports a qualified equidistant fisheye model with two radial terms,
unequal focal lengths, optical-center offset and a measured camera-to-tilt mount
rotation. Pixel rays are transformed through the sampled pan/tilt pose, and both
joint angles are solved together to place the target at the **image center**.
This is not simply `atan(pixel_error/focal_length)` or advertised diagonal FOV
divided by image width. Instance motion prediction uses the same camera model.

Set `tracking.geometry_file` to an absolute path to a **device-specific qualified**
JSON file. An invalid/unqualified file prevents startup. A camera serial,
resolution, motor ID or direction mismatch holds tracking with a visible error;
it does not silently revert to the linear mapping. The old linear path remains
available when no geometry file is configured. The API reports `tracking.mapping`
as `fisheye-kinematics` or `linear` and `camera.identity` as USB vendor:product:serial.
Changing servo zeros preserves the physical mapping by translating angles back
to the calibration's encoder-origin reference. Moving the camera mount, changing
its lens/focus, or modifying mechanical axes requires new calibration.

The tools below target the commissioned +X/right, +Y/up, positive-encoder pan/tilt
assembly. They use **measured**, settled encoder angles, not requested angles.
`capture-scene.py` and `check-scene-pointing.py` move motors: only run after an
operator clears the mechanism and explicitly authorizes the small sweeps. They
start with motors off, clear automatic tracking, restrict excursions to ±8° pan
and ±6° tilt around the start, issue steps no larger than 3°, abort on Stop or goal
changes, return home on success, then disable torque. On an abort they stop
without overriding the operator with a return move. They do not change zeros,
limits, EEPROM, model or prompts. Do not interact with sliders while they run.

```bash
# Service must be running on localhost:8080; output path must not exist.
python3 tools/capture-scene.py --output /path/to/new-capture --allow-motion
# Use the inference venv (OpenCV + numpy + scipy); this step never moves motors.
python tools/fit-scene.py /path/to/new-capture --output /path/to/candidate.json
# Only after fit passed, with operator clearance still valid:
python tools/check-scene-pointing.py --geometry /path/to/candidate.json \
  --output /path/to/new-pointing-check.json --allow-motion
```

Fitting uses spatially distributed, reciprocal SIFT matches from a static room.
Eight poses train the model; six other poses test it. Median/p90 held-out feature
prediction errors must improve on the old mapping and pass absolute pixel-error
gates. Reject inconclusive calibration rather than relaxing the gates. This is
a rotational approximation: camera/axis offsets create depth-dependent parallax,
moving objects and backlash can add error, and small local sweeps do not establish
accuracy throughout the full ±90° mechanical range. Keep visual feedback active.
See [the measured qualification](docs/geometry-qualification.md).

On spring-edge-2, version 0.8.1 passed separate 6° commanded-offset checks against
a blue bag: first centered detection at 1.05 s (pan) and 0.86 s (tilt) after class
selection, with final errors under 3 px per axis and stable holding goals. These
are small-offset hardware checks, not full-range or moving-object benchmarks.

Configured command limits are **X: −90° to +90°** and **Y: −90° to +90°**.
They are enforced in the API as well as the sliders. Changing limits never
commands motion. If an axis is outside its new range, Start stays unavailable
until it is repositioned with torque off. Changing limits does not change the
calibrated zero; both axes' current ranges include zero.

While running, feedback outside these software limits no longer stops the
motors. The affected axis is commanded back to the nearest limit, with torque
remaining on. Recovery happens once per excursion so it does not repeatedly
reset the motion profile or overwrite a subsequent valid slider command. Both
axes' communication, hardware-fault and torque checks must still pass before
any correction. Stop and those fault shutdowns
are unchanged; recovery never starts a stopped motor. These are corrective
software limits, not a guarantee against physical overshoot.

Both axes now use `profile_velocity: 0` and `profile_acceleration: 0`.
In the required velocity-based drive mode, these are the XL330's documented
[uncapped profile values](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/#profile-velocity112),
not zero speed. The service writes them on every Start. Physical speed still
depends on the actuator, supply and load; current/PWM limits, hardware shutdown,
angle limits, the bus watchdog and Stop are unchanged. Uncapped motion can be
abrupt and has not been physically qualified; the recorded small-angle tests
used the earlier velocity 20 / acceleration 5 profile. Use bounded profiles
when commissioning a different mount.

There is no password or application-level access control. Run it only on an
isolated demo LAN. The software Stop is not an emergency stop; keep a physical
power disconnect available.

## Run

The OS is responsible for packages, permissions, stable device paths, and
service management. The matching integration lives in
[`Spring-Silicon/edge-image#19`](https://github.com/Spring-Silicon/edge-image/pull/19).

```bash
uv sync --frozen
uv run spring-turret --config config/spring-turret-demo.json
```

The configured `/dev/spring-turret-camera` and `/dev/spring-turret-servo` paths
must already exist and be accessible to the process. Check the live state:

```bash
curl http://127.0.0.1:8080/api/status
```

## Model selector

The demo offers SAM 3.1 (boxes), SAM 3.1 Mask (per-frame masks), and SAM 3.1
Tracking (temporal masks and IDs). All accept free-text prompts, retain multiple
instances, and support per-class counts and click retargeting. YOLO is removed
from the selector, API model registry, worker and installation dependencies.

Applied prompts and browser drafts are kept separately per model. Switching
models clears old results and instance IDs and holds automatic motion; it keeps
the selected class when that class remains in the applied prompts and
never arms the motors. Workers exit when switching models; no warm-worker pool
is enabled. Disk compiler caches survive but weights/live GPU graphs reload.
Start remains requested across device faults; reconnection automatically resumes
after checking the commissioned hardware. Stop clears that request even while
disconnected. Enter starts again (and applies prompts first when editing them).
See [reconnection behavior](docs/recovery.md).
See [the deployed policy audit](docs/sam-policy-audit.md) for current Arc/Thor
model/hardware differences; inference and control policy are shared.

## SAM 3.1 on Intel Arc

Inference is optional. Existing camera/servo configurations continue working
without PyTorch. Use a separate Linux x86_64 / Python 3.12 environment:

```bash
uv venv --python /usr/bin/python3 /var/lib/spring-data/turret-inference/venv
uv pip sync --python /var/lib/spring-data/turret-inference/venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/xpu \
  --index-strategy unsafe-best-match requirements-sam31.txt
```

Put your authorized Meta SAM 3.1 multiplex checkpoint at the path in
[`config/inference.example.json`](config/inference.example.json), then add that
object as the `inference` key in the service's JSON config. The checkpoint is
not included in this repository. The tested checkpoint SHA256 is
`0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`.

The checkpoint path is administrator configuration, never a browser-supplied path.

The host needs both Intel Level Zero and **GPU OpenCL** drivers. On the tested
Ubuntu 24.04 host, `libze-intel-gpu1` and `intel-opencl-icd` are both
`26.31.39395.13-1~24.04~ppa1`. `clinfo -l` must list the Arc GPU. Having an XPU
device alone is insufficient: oneDNN attention also needs OpenCL.
Install `python3-dev` and `build-essential` for Triton's native compilation helper.

OS/service integration remains in `edge-image`, not this repository. Its service
user needs `render` in addition to the existing video/serial groups, read access
to the model/environment, and write access to `inference.cache_dir`. A hardened
systemd service must allow JIT executable memory (`MemoryDenyWriteExecute=false`)
and the cache path (`ReadWritePaths=...`) while keeping the other restrictions.

Implementation:

- The exact SAM **3.1 detector** is loaded strictly from `detector.*` weights.
  The tracker, interactive branch, and mask output are unused; this is text
  grounding on sampled frames, not persistent video tracking.
- Fixed 1008x1008 inputs and 32-token prompts run through full-graph
  `torch.compile(backend="inductor")` on XPU, then `torch.xpu.XPUGraph` capture
  and replay (SYCL graphs). This does not use the separate Spring Graphs runtime.
- The image backbone runs **once per sampled frame**, regardless of prompt
  count. Text embeddings are cached for the active categories; unchanged or
  reordered prompts do not rerun the text encoder. Cached embeddings own their
  buffers so subsequent SYCL replays cannot overwrite another category's text.
- Prompt-conditioned detection replays the batch-one grounding graph for each
  category against the shared image features. This was faster than batches of
  two or four on the B580. Larger fixed batches remain in the benchmark path.
  All detected instances are merged into the exact source JPEG,
  with a label and color per category. Overlapping categories can label the same
  object; boxes are not suppressed across categories. This shares the image
  backbone, not the text-conditioned fusion encoder/decoder computation.
- Image, text, and grounding stages are individually full-graph compiled and
  SYCL-captured. Grounding captures use bounded fixed shapes, not one retained
  graph for every possible prompt list. First use of a grounding batch also
  checks the shared-feature output against independent full eager passes.
- First use validates eager versus compiled outputs and graph replay, and can
  take several minutes. The UI reports loading/compiling/capture separately.
  Errors are visible; there is no silent eager or CPU fallback.
  Eager/compiled validation bounds confidence differences to 0.03 for candidates
  within 0.03 of the threshold or above it, and retained XYXY coordinate
  differences to 0.01 (normalized). Rejected low-confidence queries are checked
  for finite values but not numerical parity. Decisions right at the threshold
  can differ with mixed precision. Graph replay is additionally compared to
  uncaptured compiled raw outputs at `atol=rtol=0.001`.
- The inference subprocess consumes only the latest available camera frame;
  there is no frame backlog. The browser draws boxes over their exact source
  JPEG only after that image loads (`client_overlay: true`), avoiding another
  JPEG decode/encode and base64 return trip in the worker. Standalone engine
  calls still return annotated JPEGs by default. A
  prompt change/clear invalidates prior results immediately. Stale output is
  replaced by the raw feed. Submitting an empty list stops new inference; the
  model remains loaded for the next prompt. An in-flight compilation/inference
  may finish.
- Confidence is `sigmoid(class logit) * sigmoid(presence logit)`, threshold 0.5
  by default. Boxes are normalized XYXY coordinates in `/api/status`.
- `latency_ms` measures the image encoder, grounding and box postprocessing;
  it excludes preprocessing, prompt setup and JPEG annotation. `timing` exposes
  each stage and `worker_total_ms`, which includes those worker-side overheads
  (including cold compile/validation when applicable). HTTP delivery, camera
  buffering and browser display remain outside the worker timing.
- `pipeline_timing` measures pose lookup, camera wait, worker round-trip and the
  complete detection cycle. A persistent event stream delivers detection metadata
  and its original JPEG together, independently of the 200 ms hardware-status
  refresh. This avoids two network round-trips per frame. Slow clients skip to
  latest on the server, and the browser holds only one pending image while
  decoding; JPEGs and boxes remain frame-matched. Stream disconnects reconnect
  automatically, with metadata long-polling and individual JPEGs as fallback.
- The connected Arducam advertises at most 30 FPS, including at smaller sizes.
  A 15 ms forward pass alone does not imply 60 distinct camera detections/second;
  capture, preprocessing, IPC and delivery also contribute. Do not count repeats
  of the same camera sample as additional detections.

Opt-in GPU validation (never controls the servo):

```bash
/var/lib/spring-data/turret-inference/venv/bin/python tests/smoke-sam31.py \
  --checkpoint /var/lib/spring-data/turret-inference/sam3.1_multiplex.pt
```

The smoke test benchmarks shared-feature grounding batches of 1, 2 and 4 for
1–8 categories, verifies one image-encoder replay per steady-state frame, text
cache reuse, changed/reordered prompts, and a changed image against independent
full passes. For nonempty regression coverage, supply `--image IMAGE.jpg
--require-detections` with a frame containing at least two detectable people.
Run it separately from the service's GPU worker to avoid loading two models.

B580 warm inference medians (10 frames/case, FP16 autocast, same 1280x720 camera
JPEG with two detected people; preprocessing/annotation excluded):

| Categories | Shared image, grounding batch 1 | Batch 2 | Batch 4 |
| --- | ---: | ---: | ---: |
| 1 | 138 ms | 138 ms | 138 ms |
| 2 | 151 ms | 164 ms | 186 ms |
| 3 | 164 ms | 177 ms | 186 ms |
| 4 | 178 ms | 203 ms | 186 ms |
| 8 | 231 ms | 281 ms | 247 ms |

Batch-one is the deployment default. The image stage is about 124 ms/frame;
each grounding replay about 13 ms. Three-category worker time including JPEG
preprocessing and annotation was about 188 ms, versus the old implementation's
432 ms **inference alone**. These are historical measurements; the later native
image backend and frame-driven delivery are separate improvements.

## API

- `GET /stream.mjpg`
- `GET /api/status`
- `GET /api/detection/events` streams JSON events with metadata and base64 `jpeg`
  together for each new frame; idle/progress heartbeats omit the image.
- `GET /api/detection/status?revision=1&sequence=25` waits up to one second for
  newer detection metadata, without acquiring hardware-status locks.
- `POST /api/servo/arm`
- `POST /api/servo/disable`
- `POST /api/servo/keepalive` compatibility no-op for older clients; no heartbeat required
- `POST /api/servo/position` with `{"axis": "x", "degrees": 10.5}` (or `"y"`)
- `POST /api/servo/gains` with `{"axis": "x", "p": 400, "d": 0}`
- `POST /api/servo/gains/reset` with `{"axis": "x"}` (or `"y"`)

Per-axis P/D sliders and Reset controls are available in each device panel.
See [servo gains](docs/servo-gains.md) for raw units, persistence, readback and
reset-baseline behavior. Gain changes do not arm motors or change servo zeros.
- `POST /api/tracking/instance` with `{"revision": 1, "frame_sequence": 25, "instance_id": 7}` temporarily retargets to a box from the displayed frame without arming.
- `POST /api/tracking/target` with `{"target": "cup"}` (an applied class), or
  `{"target": null}` to clear it. Selection is not persisted across restarts.
- `POST /api/detection/prompts` with `{"prompts": ["person", "cup"]}`; `[]` clears
- `POST /api/detection/prompt` with `{"prompt": "chair"}` (single-category compatibility)
- `GET /api/detection/frame/REVISION-SEQUENCE.jpg` (exact annotated frame URL
  returned in `status.detection.frame_url`; old revisions return 404)

## Validate

```bash
make validate
```

## Servo qualification still required

The sample config is deliberately `calibrated: false`: Start is blocked until
the actual assembly is commissioned. Use a stopped service and an exclusive bus
connection for commissioning. Never change an ID with two factory-ID-1 motors
connected: isolate the bottom/pan motor, torque off, assign ID 2, verify readback,
then reconnect the upper/tilt motor (ID 1). Both must respond separately at
57,600 baud with model 1200, drive mode 0, secondary ID 255, homing offset 0,
torque off and no hardware error. With torque disabled, program and verify
**Operating Mode (11) = 4** (extended position) on both motors.

With torque off, place the camera straight ahead and level. Read each present
position modulo 4096 into its axis's `center_position`; set `direction` to 1
or -1 for the mount's orientation. Use slow profiles and set `calibrated: true`
after confirming neutral and clearance. The sample reflects the operator's
requested X ±90° / Y ±90° limits, not a qualified full-travel envelope.
The [reference CAD](https://github.com/AnthonyZJiang/dynamixal-pan-tilt-camera-cad)
specifies ±90° maximum travel, but mounting and cable clearance must be checked
on each assembly. No full-travel sweep was performed when applying these limits.

Extended position mode is intentional: this assembly's neutral tilt is near
encoder rollover. A bounded tilt can therefore cross 4095/0. The controller
holds a continuous local coordinate while armed and chooses the nearest
equivalent zero after power cycling/reconnecting; it never commands a full turn
to recover zero. Signed positions are supported. Extended mode ignores the
servo's EEPROM min/max position limits, so the service rejects out-of-range
commands and corrects out-of-range feedback while running. Invalid readings
outside the actuator's extended-position range still stop both axes. Do not
turn the assembled mount through full revolutions.

The service does not rewrite EEPROM on startup. It rejects incompatible modes
instead of silently changing the coordinate system. The configuration format
changed in 0.6: migrate the old flat single-servo fields into `servo.axes` and
preserve the existing `camera`, `listen`, and optional `inference` sections.
Full mechanical qualification still requires checking travel endpoints, cable
clearance, direction, Stop during motion, current/temperature and recovery after
power loss. A small motion/readback test is not full-range qualification.
