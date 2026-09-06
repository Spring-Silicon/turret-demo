# Israel W8A8 development image

This is the retained **development candidate**, not an accuracy-qualified model.
Israel's 24-frame/72-case development set still fails five confidence and four
retained-box checks. Its 80% roofline target is unproven. The UI and API expose
that status; no threshold, dense reference, or servo safety limit is changed.

Only the shared image encoder changes. The image linears use SmoothQuant W8A8
(alpha .65, original activation maxima and retained bias correction), grouped
native QKV/RoPE and queued native MLP kernels. Attention/convolutions remain
dense. This is **not one whole-image megakernel**. Text is still cached and each
class runs the unchanged compiled grounding heads. Everything executes in one
Torch XPU process with explicit SYCL graphs; the prior dense native image's
host IPC/upload/download boundary is absent.

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
The UI always labels this image as accuracy-unqualified. Shape/dtype mismatches,
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
