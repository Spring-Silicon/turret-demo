# Spring turret demo

Camera feed, text-prompt SAM 3.1 bounding boxes, and manual X/Y pan/tilt controls.

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
attempts torque-off on both axes. An unconfirmed stop is reported explicitly;
reconnection never automatically re-arms. Start checks model, position mode,
drive mode, secondary ID, homing offset, range and hardware faults.

The browser sends a keepalive while running. After three seconds without one,
the service releases both motors. Each servo also has a one-second bus watchdog:
if the process or bus stops sending traffic it stops motion, **but retains
holding torque**. Neither safeguard is a physical emergency stop. A released
tilt axis can fall under gravity; support the camera before disconnecting power.

## UI

Open `http://HOST:8080/`. Below the camera are an **X degree slider**, a central
**triangle/square Start/Stop** button, and a **Y degree slider**. Sliders move
their respective axes while running; requests are coalesced during a drag.
Errors appear only when needed. Enter
one object category per row (for example `person`, `cup`, `keyboard`). Each row
shows its detected instance count and a trash button. The single **+** below
the list adds another row. Counts show `—` while unavailable or for unapplied
prompts; `0` means no instances were detected in the current result. Select **Update
prompts**, or press **Enter** in a text box, to apply every row together.
Edits do not change active detection until submitted. Up to eight categories
are supported; blank and duplicate prompts are ignored. Remove/empty all rows
and update to return to the raw feed. Detection never arms, aims, or moves either servo.

Configured command limits are **X: −90° to +90°** and **Y: 30° to 90°**.
They are enforced in the API as well as the sliders. Changing limits never
commands motion. If an axis is outside its new range, Start stays unavailable
until it is repositioned with torque off. The calibrated zero remains unchanged
even when zero is outside the allowed range (as it is for Y).

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
  there is no frame backlog. Boxes are drawn into their exact source JPEG. A
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
432 ms **inference alone**. Browser/network overhead and the 5 FPS cap are unchanged.

## API

- `GET /stream.mjpg`
- `GET /api/status`
- `POST /api/servo/arm`
- `POST /api/servo/disable`
- `POST /api/servo/keepalive` at least once a second while running
- `POST /api/servo/position` with `{"axis": "x", "degrees": 10.5}` (or `"y"`)
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
requested X ±90° / Y 30–90° limits, not a qualified full-travel envelope.
The [reference CAD](https://github.com/AnthonyZJiang/dynamixal-pan-tilt-camera-cad)
specifies ±90° maximum travel, but mounting and cable clearance must be checked
on each assembly. No full-travel sweep was performed when applying these limits.

Extended position mode is intentional: this assembly's neutral tilt is near
encoder rollover. A bounded tilt can therefore cross 4095/0. The controller
holds a continuous local coordinate while armed and chooses the nearest
equivalent zero after power cycling/reconnecting; it never commands a full turn
to recover zero. Signed positions are supported. Extended mode ignores the
servo's EEPROM min/max position limits, so the service enforces both commanded
and measured limits. Do not turn the assembled mount through full revolutions.

The service does not rewrite EEPROM on startup. It rejects incompatible modes
instead of silently changing the coordinate system. The configuration format
changed in 0.6: migrate the old flat single-servo fields into `servo.axes` and
preserve the existing `camera`, `listen`, and optional `inference` sections.
Full mechanical qualification still requires checking travel endpoints, cable
clearance, direction, Stop during motion, current/temperature and recovery after
power loss. A small motion/readback test is not full-range qualification.
