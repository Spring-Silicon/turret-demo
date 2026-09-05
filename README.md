# Spring turret demo

Camera feed, text-prompt SAM 3.1 bounding boxes, and manual servo controls.

## Current hardware status

- Arducam 1080P Low Light (`0c45:0261`, serial `UC684`): 1280x720 MJPEG at
  30 fps verified on `spring-edge-2`.
- USB Single Serial adapter (`1a86:55d3`, serial `5B61036033`): enumeration and
  stable device naming verified.
- ROBOTIS DYNAMIXEL XL330-M288-T: model 1200, firmware 53, Protocol 2.0, ID 1
  at 57,600 baud. Read-only PING and position are verified; motion is not yet
  qualified.

The service starts disarmed, sends no startup movement, rejects positions outside
1536 through 2560, and rejects every movement until the operator selects **Arm**.
Communication failure clears the armed state. Service shutdown attempts to turn
torque off. Arming first writes the current position as the goal, then enables
torque, preventing an immediate jump on Arm.

## UI

Open `http://HOST:8080/`. The UI contains only the live feed, camera/servo state,
position, bounded jog controls, Arm, Stop, and the current hardware error.
The motor controls sit below the camera. Enter
one object category per row (for example `person`, `cup`, `keyboard`). Each row
shows its detected instance count and a trash button. The single **+** below
the list adds another row. Counts show `—` while unavailable or for unapplied
prompts; `0` means no instances were detected in the current result. Select **Update
prompts**, or press **Enter** in a text box, to apply every row together.
Edits do not change active detection until submitted. Up to eight categories
are supported; blank and duplicate prompts are ignored. Remove/empty all rows
and update to return to the raw feed. Detection never arms, aims, or moves the servo.

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
- `POST /api/servo/center`
- `POST /api/servo/position` with `{"position": INTEGER}`
- `POST /api/detection/prompts` with `{"prompts": ["person", "cup"]}`; `[]` clears
- `POST /api/detection/prompt` with `{"prompt": "chair"}` (single-category compatibility)
- `GET /api/detection/frame/REVISION-SEQUENCE.jpg` (exact annotated frame URL
  returned in `status.detection.frame_url`; old revisions return 404)

## Validate

```bash
make validate
```

## Servo qualification still required

Before changing `hardware.json` to motion-qualified, confirm the actuator model,
electrical interface, supply voltage, ID, and baud rate. Then verify read-only
PING/position, mechanical center with linkage disconnected, conservative limits,
Stop under motion, communication-loss behavior, current, and temperature.
