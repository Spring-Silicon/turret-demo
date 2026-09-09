# Israel W8A8 development image

This is the retained **development candidate**, not an accuracy-qualified model.
Israel's 24-frame/72-case development set still fails five confidence and four
retained-box checks. Its 80% roofline target is unproven. The API exposes
that status; no threshold, dense reference, or servo safety limit is changed.

In the original profile, only the shared image encoder changes. The image linears use SmoothQuant W8A8
(alpha .65, original activation maxima and retained bias correction), grouped
native QKV/RoPE and queued native MLP kernels. Attention/convolutions remain
dense. This is **not one whole-image megakernel**. Text is still cached and each
class runs the unchanged compiled grounding heads. Everything executes in one
Torch XPU process with explicit SYCL graphs; the prior dense native image's
host IPC/upload/download boundary is absent. Release 0.15.4 also supports the
retained packed-head profile described below; it does change the grounding
implementation, while retaining the independent original dense reference.

Source: `spring@israel:/home/spring/sam3_1`, SAM commit
`660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`. Runtime code comes from the retained
`results/turret_megakernel/native_qkv_sources` snapshot. Files are pinned in
`src/spring_turret/w8a8_manifest.json`, including SAM source, both binaries,
calibration tensors, build metadata, report and saved feature/reference banks.
The base checkpoint is the same SHA256-pinned `sam3.1_multiplex.pt`; it is not
replaced with quantized reference weights. Do not publish the private bundle or
its checkpoint-derived tensors. Preserve its SAM license.

Retained binaries:

- `joint_qkv_rope_grouped.so`: `3f5f5d2f2ab49fe0ed5baf98078ace1ecaf17b2d030ea9e747cf0036aa0f848b`
- `joint_mlp_blockstore.so`: `ca5106ece670d91c053c3c1f959a84dac3ff0e40b6fcdfc1a372aec60e6392aa`

Source measurements: 74.02/74.49 ms image replay (GPU0/GPU1); GPU1 full worker
107.50/122.30/151.30/209.59 ms for 1/2/4/8 prompts. These are Israel results,
not guarantees for another host or browser/camera/servo end-to-end latency.

## Retained exact head-cast cache (2026-09-07)

Israel's current retained recipe is `engine_only_retained_cast_cached_config.json`,
SHA-256 `b7a9fa2c6fb283f8060c005cd9b6d5a3a6604a5e4c98b1c3dea749479be56539`.
This extends the packed profile below. The checkpoint, image kernels, calibration,
FP32 head masters, arithmetic, resolution, prompts and thresholds are unchanged.
It constructs 243 exact FP16 parameter-cast caches once, removing 293 repeated
cast launches per head pass. The four feature/position/text/mask inputs remain
dynamic; each requested class still gets its own head pass over shared image
features. No frames or classes are omitted.

`w8a8_cast_cache_manifest.json` pins the retained recipe, original helper and
qualification evidence. Place those files at their manifest-relative paths in
a **copy** of a verified packed bundle, then run destination qualification before
selecting that bundle. The adapter recognizes the retained config; it never
selects newer experimental files by timestamp. Keep the original bundle/cache
and application for rollback. To copy this extension, including its packed base:

```sh
bash scripts/copy-sam31-w8a8.sh spring@israel spring@spring-edge-2-1 cast-cached
```

The `packed` argument still copies only the older profile.

The pinned helper proves that only pure FP32-to-FP16 conversions and GEMM read
operands are changed. Construction runs before graph capture, with exact
retained/direct/replay checks; unsupported generated code aborts, without a
silent fallback. Original parameters and cache buffers remain owned for the
worker's lifetime. **Any weight/device/layout change requires a new worker and
new cache/graphs**; in-place parameter updates are not supported.

Runtime identity is `israel-w8a8-packed-cast-cached-development`. After successful
head capture, `native_image_validation.head_cast_cache` reports exact replay,
structural-proof and cache identities. These are copy-equivalence checks, not
full accuracy qualification. The existing five confidence/four box development
failures remain disclosed in the API. The destination qualification additionally
requires bitwise equality to saved current-release camera outputs.

Israel's alternating resident image-plus-one-head benchmark saved 0.75–0.88 ms
(about 85 ms to 84.2 ms). That measurement excludes host preprocessing, camera,
network and browser delivery, and is not a destination FPS guarantee.

Arc destination verification (2026-09-07): all 25 image replays and 72 source
head cases passed; all 28 camera-image/prompt outputs were bitwise identical to
the captured prior deployment. Source failures remain exactly five confidence
and four box checks, with the same identities. No incremental failures occurred.
Six alternating same-process rounds on B580 measured image-plus-head execution
at 86.85 ms control versus 86.12 ms cached (median paired saving 0.734 ms).
The deployed `hand` prompt delivered 336 paired frames in 30 seconds, 11.15 FPS;
mean image/head/worker times were 73.98/11.44/89.12 ms. This live sample is not an
isolated before/after FPS comparison or exposure-to-browser latency measurement.
Camera settings, servo calibration and UI were preserved; motors remained off.

## Previous retained packed profile (0.15.4)

Israel's retained recipe is `engine_only_retained_packed_config.json`, SHA-256
`c7985199c70ead9b4b6735b4c8ae9599135b1eeb597f759506b68b12a77e51b7`.
It adds cached-reciprocal vector-16 MLP execution and native split-512 decoder
attention with compact positional bias. It retains the same checkpoint, image
calibration, resolution, confidence threshold, text encoder and FP32 master
weights. The six decoder layers use corrected explicit FP16 rounding in the
axis MLP, not the rejected initial packed-mask prototype.

```sh
bash scripts/copy-sam31-w8a8.sh spring@israel spring@spring-edge-2-1 packed
```

The packed manifest extends (does not replace) the original manifest. All helper
sources, libraries, build metadata and source parity evidence are hash-pinned.
An original bundle without the retained config continues to select the old
implementation for rollback. The packed profile requires grounding batch size 1;
up to eight class prompts still share one image encoding, followed by a cached
text embedding and one packed head pass per prompt. No prompts are dropped.

New binaries:

- `joint_mlp_cached_recip_vec16.so`: `bbb4c548e66b46dffbafed93f8bf01656d86ad0d5490f67a93e3039ab4aa360e`
- `joint_head_attention_packed512.so`: `4c62c5864217e51918fa3d363def33949d0e980c521bd2fbeff596b10b8ac4db`

**The user-space GPU compiler/runtime is part of this profile.** On spring-edge-2,
the installed Intel 26.31 / IGC 2.40.13 runtime did not reproduce Israel's saved
head outputs (the image features still matched). Copying Israel's Intel 26.27 /
IGC 2.38.5 libraries restored bitwise head parity across all 72 source cases.
The copy script includes these five hash-pinned libraries under
`runtime/graphics`; `w8a8_graphics_manifest.json` records their identities.
The parent sets `LD_LIBRARY_PATH` before launching **only the packed SAM worker**.
No system package, installed driver, kernel, service-wide environment or YOLO
runtime is changed. The worker rejects an incorrectly loaded Level Zero library.
Packed SAM uses its own `sam31-israel-w8a8-packed` compiler cache.

Israel measured 85.00/84.98 ms (GPU0/GPU1) for a **single resident graph** containing
image, one cached-prompt head and device decode. Those numbers exclude input
copies and CPU/camera/network work. The demo retains its separately measured
image/head graphs and lossless latest-frame CPU overlap; use live measurements
for the delivered FPS rather than substituting the source engine-only number.

The new MLP preserves the old image features exactly. Native attention is **not
bitwise-identical to the original Torch heads**. Source development checks retain
the same 5 confidence and 4 box failure identities, with no changed detection
keep masks or new failures across 72 cases. These reused development checks do
not establish independent accuracy qualification or guarantee unseen-data parity.
No tolerances are widened. Dense-reference execution is outside the native-head
patch context; the context restores on success and failure and is not reentered
for graph replay.

Destination qualification additionally requires exact Israel image/head output
parity, direct/replay checks, old-head drift gates, identical detection sets and
unchanged failure identities. Pass `--images /path/to/camera-jpegs` to test old
versus packed heads and both versus dense on additional camera frames, using
`face`, `person`, `chair` and `bottle`. Any new gate or detection-set failure aborts
this test; the service's development report-only flag cannot waive it.
For packed-profile command-line tests, prepend
`/absolute/copied-bundle/runtime/graphics` to `LD_LIBRARY_PATH` **before Python
starts** and use a separate compiler cache. `tests/capture-sam31-reference.py`
can first capture the installed release's original outputs, using its normal
library path and installed package in `PYTHONPATH`. Pass that output with
`--baseline` to compare against the actually deployed model/runtime, not merely
the original head running under the candidate runtime.

Destination result: [qualification report](sam31-packed-qualification.json).
All 72 source head cases and 25 changed/return-to-first image replays matched
source bits. Seven camera images times four prompts passed with identical
detection sets (35 detections total), including comparison against captured
0.15.3 outputs using the original installed graphics runtime. Maximum meaningful
confidence drift was 0.000853; maximum normalized box-coordinate drift was
0.000123 (less than 0.16 pixels at 1280px). All 28 camera cases passed both dense
gates; the source set's pre-existing 5/4 failures and identities remain unchanged.

Live `face` prompt, same boot/B580, 1280x720@30, motors off: the pre-update 15s
sample delivered 10.58 FPS; the [30s packed sample](sam31-packed-throughput.json)
delivered **10.98 FPS** (331 results), approximately 3.8% higher. Mean image time
fell 75.26 to 74.07 ms, grounding 14.17 to 12.24 ms, and total processing loop
94.75 to 91.15 ms. Model plus device postprocessing was approximately 86.8 ms.
These are local paired-JPEG delivery measurements, not exposure-to-display
latency; preprocessing overlap and camera/network overhead remain present.
The live `face` startup dense-reference check also passed.

Release 0.15.4 retains the prior release/config/bundle for rollback. Its final
wheel (including the packed-backend accuracy label) has SHA-256
`e51695d80d22f400ac6ce41182fc37e54f1b1326fc47f806ee761e19279a7780`.
The UI-only final wheel's Python files were compared byte-for-byte with the
qualified/deployed wheel. Repository validation passes 128 Python tests and the
UI/JavaScript contracts. Independent full accuracy qualification remains pending.

## Copy and verify

```sh
bash scripts/copy-sam31-w8a8.sh spring@israel spring@spring-edge-2-1
```

Copy `src/` and `tests/` beside the new bundle. Stop motors and pause the active
detector before running the destination test to avoid competing for its GPU:

```sh
OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2 \
LD_LIBRARY_PATH=/var/lib/spring-data/turret-inference/venv/lib \
TORCHINDUCTOR_CACHE_DIR=/absolute/new-cache/inductor \
TRITON_CACHE_DIR=/absolute/new-cache/triton \
/var/lib/spring-data/turret-inference/venv/bin/python \
  tests/qualify-sam31-w8a8.py --bundle /absolute/copied-bundle \
  --checkpoint /var/lib/spring-data/turret-inference/sam3.1_multiplex.pt \
  --output /absolute/new-report.json --image /absolute/camera.jpg
```

The test requires source bitwise feature parity and direct/replay agreement on
24 changed frames plus return-to-first. It independently reports both accuracy
gates and requires the known failure counts, rather than relabeling failures as
passes. `copy_parity_passed` is distinct from `accuracy_qualified` (always false).
By default the optional camera test must pass the demo's unchanged live startup gates.
The quantized image has separate module topology and shares only immutable
masters with the original dense wrapper; quantization never changes the reference.

After verification, explicitly select development mode by removing
`inference.sam31_native_bundle` and setting
`inference.sam31_w8a8_development_bundle` to the copied absolute path. Both keys
together are invalid. Keep FP16 precision, existing inference venv, device,
camera, servo zeros and geometric calibration. Restarting never arms motors.
YOLO remains unchanged; its worker receives neither SAM bundle flag.

Keep the prior release and config for rollback. Remove the development key to
use Torch's dense image, or restore the prior native bundle key for the qualified
sleepy-joe image.

## Explicit unqualified execution

The user may opt into running the known-inaccurate development model by setting
`inference.sam31_allow_unqualified_w8a8: true` alongside the development bundle.
Default is false; the flag is invalid without that bundle and is never passed
to YOLO. This does **not** raise the 0.03 confidence or 0.01 box tolerances or
change the dense reference. Numerical failures remain `validation[].passed=false`
with the original error and `accuracy_policy=report-only-development` in results.
The API keeps this image marked accuracy-unqualified. Shape/dtype mismatches,
nonfinite outputs, file hash mismatches, native errors and graph qualification
failures still abort. Servo bounds, lease, calibration and fault checks are unchanged.

## Demo overhead follow-up (2026-09-06, release 0.15.1)

On spring-edge-2's B580, one `chair` prompt, same W8A8 bundle, 1280x720 MJPEG
at 30 Hz, uncapped inference, a local paired-JPEG SSE receiver measured:

| Run | Motors | Delivered FPS | Preprocess median | Model image + heads | Loop median |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.15.0, 20 s | Armed, no tracking target | 8.27 | 15.58 ms | 88.98 ms | 121.40 ms |
| 0.15.0, 12 s | Off | 9.12 | 18.10 ms | 88.67 ms | 110.76 ms |
| 0.15.1, 25 s | Off | 9.82 | 9.17 ms | 88.98 ms | 102.37 ms |

The original armed run spent a median 13.08 ms (p95 27.67 ms) waiting for
capture after the newest encoder read. The updated controller chooses the
newest completed encoder read before the latest frame from bounded history,
with the same 100 ms age limit. No physical movement was commanded to benchmark
the new version; its historical-pose, future/stale rejection and fault/stop
invalidation paths were unit-tested. Do not treat the armed/off rows as an
isolated preprocessing A/B test.

The GPU normalization lookup was bitwise-equal to original CPU pixels on ten
changed-image/reverse replays (camera, varied sizes, grayscale) plus exhaustive
all-256-byte-value replays. An alternating standalone CPU-vs-GPU preprocessing
comparison measured 14.245 vs 8.732 ms medians (25 measured iterations each).
Live input parity and W8A8 direct/replay checks passed. Quantized-model accuracy
qualification is unchanged: still a development candidate.

The 0.15.1 live worker median was 98.65 ms, loop p95 103.14 ms; receipt-to-result
frame age was median 122 ms, p95 138 ms, including age of the latest 30 Hz frame.
These are not exposure-to-browser-display measurements. No frame queue or
model simplification was introduced. The ~89 ms model still bounds serial
throughput to roughly 11.2 FPS even if all remaining overhead disappeared.

### Further overhead reduction (release 0.15.2)

Same host/model/prompt, motors off in both runs, with a fresh 15-second baseline
and 30-second upgraded paired-JPEG SSE measurement:

| Metric (median unless FPS) | 0.15.1 baseline | 0.15.2 |
| --- | ---: | ---: |
| Delivered FPS | 9.83 | 10.42 |
| Preprocessing | 9.24 ms | 5.89 ms |
| Image encoder | 75.12 ms | 74.96 ms |
| Grounding | 14.09 ms | 14.07 ms |
| Worker total | 98.97 ms | 95.50 ms |
| Worker round trip | 102.51 ms | 95.94 ms |
| Full processing loop | 102.63 ms | 96.06 ms |
| Receipt-to-result frame age | 123 ms | 114 ms |

The update decodes JPEG directly into a tensor, preserves the original uint8
resize, and copies directly to the normalization graph input. A negotiated raw
JPEG protocol removes base64 from the local SAM input pipe; YOLO retains its
existing transport. No pipeline queue, frame reordering, tracking change or
model modification was added. Non-model overhead (loop minus model including
postprocessing) is approximately 13.0 to 6.6 ms; GPU model time is unchanged.
Loop/frame-age p95 after the update were 96.94/130 ms, respectively. These remain
local receiver measurements, not exposure-to-browser-display measurements.

Pixel qualification covered 44 forward/reverse image cases including camera,
varied dimensions, grayscale, progressive JPEG/chroma subsampling and CMYK/PNG
fallbacks, plus exhaustive all-256-value graph replays. All matched the original
CPU float32 pixels and strides exactly. Alternating preprocessing medians were
8.509 ms (0.15.1 path) versus 5.671 ms (new path), 25 measured iterations each.
The live worker also passed exact first-frame pixels and W8A8 direct/replay
checks. The development model's unqualified accuracy status is unchanged.

### Latest-only CPU/GPU overlap (release 0.15.3)

CPU decode/resize can run while the GPU processes the preceding frame. During
steady validated batches, the parent checks for a new camera frame every 5 ms
while awaiting the result and submits each new candidate once. The worker keeps
only a bounded active/pending/ready preparation, not a queue of inference frames.
Next inference still selects the actual newest camera frame. Reuse requires the
same revision, request, camera sequence and JPEG bytes within 100 ms; late or
superseded preparation falls back without waiting. No model, weights, precision,
resolution, threshold, pose mapping or servo behavior was changed. The trade-off
is more CPU preparation work on camera frames the GPU may never need.

Same host, W8A8 bundle, `person with white shirt` prompt, 1280x720 at 30 Hz and
motors off: local paired-JPEG SSE samples were 55 s for the candidate (585
results), followed by a 20 s rollback baseline (210 results):

| Metric | 0.15.2 baseline | 0.15.3 |
| --- | ---: | ---: |
| Delivered FPS | 10.44 | 10.61 |
| Exposed preprocessing, mean | 6.01 ms | 4.05 ms |
| Image encoder, mean | 74.96 ms | 75.00 ms |
| Grounding, mean | 14.10 ms | 14.13 ms |
| Worker total, mean | 95.60 ms | 93.74 ms |
| Full processing loop, mean | 96.08 ms | 94.28 ms |
| Receipt-to-result frame age, median / p95 | 114 / 129 ms | 108 / 125 ms |

This is a modest **1.6% measured FPS gain**, not an additional model speedup.
Non-model critical-path time (loop minus image, grounding and postprocessing)
fell from approximately 6.6 to 4.7 ms. The candidate reused preparation on 40.7%
of frames; reused preprocessing was 0.94 ms median. Median loop time barely
changed (96.05 to 95.98 ms) because most frames still used the normal path; mean
loop time and delivered FPS capture the benefit. These are local receipt/result
measurements, not hardware exposure-to-browser-display latency.

An initial late-only preparation window was rejected in favor of latest-only
preparation throughout GPU execution. It produced only 19–38% reuse and little
throughput gain. Neither version substituted an old prepared frame for a newer
camera frame.

Accuracy-preservation qualification is **strict**, separate from the W8A8
development model's pre-existing accuracy limitations:

- 44 changed-image forward/reverse cases, 22 background-prepared buffers and
  exhaustive byte-value replay matched the original CPU normalized pixels.
- `tests/qualify-sam31-overlap.py` compared the installed 0.15.2 preprocessor
  with the new path on six actual camera images, using one and four prompts.
  During optimized inference, other changed frames were prepared concurrently.
  All **30,030 raw logits/box/presence values were bitwise identical** across
  12 comparisons. All published boxes, confidence scores and class counts also
  matched, including 32 detected boxes in total. No tolerance or report-only
  waiver was used for this comparison. See [the qualification result](sam31-overlap-qualification.json).
- Repository validation passed 124 Python tests plus UI/JavaScript contracts,
  including latest-generation ownership, late-preparation fallback, prompt
  changes, duplicate suppression and real subprocess pipe integration.

The tested deployment wheel SHA-256 was
`c078977075c556a325d60eb4278ceb7f1a8dfb2d6233855de5d89751e3d556c8`.
`inference.sam31_cpu_prefetch: false` disables overlap. Runtime metadata exposes
`preprocess_prefetched`, `prefetch_candidate_matched`, `prefetch_frames_sent` and
`prefetch_window_ms` (time from last candidate send to result receipt).
`timing.cpu_prepare_ms` includes overlapped CPU work; do not add it to worker
total. W8A8-versus-dense accuracy remains unqualified as before.
