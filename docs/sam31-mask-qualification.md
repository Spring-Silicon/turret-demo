# Mask profile deployment check — 2026-09-08

Arc: `spring-edge-turret`; Thor: `agxthor-5`.

## Newest sleepy-joe mask path, follow-up on 2026-09-08

Arc now uses `masks/w8a8/skip4_attention8-v0`; Thor is unchanged by this update.
The extension manifest is
`2ec94c43535ea3cfd4c384b0d2adcfbb26907d18bbc0734655c66f3cf6958640`.
It pins the source-selected native projection library, calibration, sources and
qualified four-block removal recipe. See `sam31-mask.md` for the measured
0.1416-point additional person-mask AP tradeoff and its dataset limitations.

Sixteen changed-frame/prompt fixture cases passed on the actual Arc: `person`,
`cup`, both together, and return to `person`, using three source image sizes.
They checked compiled-direct versus replay, decoded masks and exact integer
postprocessing. The first pass exposed repeated reconstruction of postprocessing
graphs; the worker now retains up to eight shape/prompt-count graph entries.
The complete rerun passed. Its timings are not a speed claim because the live
tracker was resumed during that fixture run and contended for the GPU.

A subsequent 90-frame **live** sample with only the demo inference worker on the
Arc measured:

| Stage | Median |
| --- | ---: |
| GPU model | 63.45 ms |
| Whole mask worker | 68.10 ms |
| Backend cycle | 69.38 ms |
| Observed processed-frame rate | 14.29 FPS |

Prompt: `arm`; observed counts: 0–2 objects; exact prepared-frame hit rate 92.2%.
This is a short live deployment sample, not controlled end-to-end before/after
testing. Five additional paired-frame fetches decoded successfully and advanced
frame sequence numbers. The installed API reports the new recipe, 28 retained
blocks, 56 W8A8 projections, `torch_compile`, SYCL graph replay, and successful
preprocessing/postprocessing validation.

Only three Arc mask-loader/worker Python files were installed. The new extension
is linked from the existing base bundle's `attention8` directory to the frozen
bundle under
`/home/spring/.local/share/turret-demo/sleepy-mask-update-20260909.bDg4xS/bundle/attention8`.
The original base bundle is unchanged; removing that extension link and reloading
the mask worker restores the prior numerical path. Predeployment source files,
fixture reports and live measurements are retained in that same update directory.
Local evidence is in `../sleepy-mask-update-20260909.5L4GSt`.

Backend/viewer processes were not restarted; hardware config, servo/geometry code
and tracking-input ownership-fix hashes are unchanged. The Arc frontend and kiosk
remain active, and Thor routing still uses `192.168.249.2` directly via `enp7s0`.
The live Arc selection is SAM3.1 mask / `arm`; clicked IDs from the old temporal
session are not transferred. Full repository validation passed (dependency-gated
local Torch tests skipped; actual-GPU fixture tests ran separately).

## Original mask-profile deployment

| Check | Arc | Thor |
|---|---:|---:|
| Alternating fixtures and changed prompts | 16 frames passed | 16 frames passed |
| Live camera and paired JPEG/mask API | 20 frames passed | 20 frames passed |
| Warm live GPU model median, one prompt | 70.98 ms | 111.30 ms |
| Warm live worker median, one prompt | 80.71 ms | 165.69 ms |
| Replay | One SYCL graph | One CUDA graph |

The live prompt was `person`. Medians exclude the first two sampled results.
Worker time includes preprocessing, model execution, transfers and mask encoding,
but not networking/browser presentation. These are short functional deployment
measurements, not a sustained benchmark or an accuracy requalification.

Fixture tests alternate three images and prompt sets `person`, `cup`, both,
then `person` again. Compiled-direct and replay outputs are compared after input
changes: exact discrete values, 0.001 relative/absolute floating tolerance.
All source manifests and the checkpoint were verified on Arc.

CPU tests passed: 33 detection tests, three mask profile tests, two fixed-output
tests, and five mask-display/encoding tests. Browser unit scenarios passed for
independent panels, shared model selection, per-frame masks without temporal
tracking, overflow notices and atomic JPEG/mask presentation. The frontend's nine
proxy tests also passed.

The actual fullscreen Firefox combined viewer was sampled 600 times over 30
seconds: **zero black frames on either panel**, zero presentation-canvas
replacements and zero raw-MJPEG requests. Both panels advanced through processed
mask frames, and the shared mask-profile option was enabled.
After testing, both devices' previous `sam3.1-tracking` / `face` selection and
`face` target were restored. Saved servo calibration hashes matched their
predeployment copies; motors were left stopped.

The broader pre-existing `test-mask-centroid.py` suite still has one unrelated
mock-fixture error: its `NativeTrackingEngine` mock lacks `cache_warmup_frames`.
The new mask profile does not use or modify that temporal worker.

Arc required Python 3.12 development headers for fresh compilation; installing
them also brought the matching Ubuntu Python patch packages from `.15` to `.16`.
The existing per-service OpenCL environment is retained. No GPU driver, servo
calibration, USB binding, angle limit or existing model weights were changed.

Raw host reports and predeployment backups are retained under
`/home/spring/.local/share/turret-demo/mask-20260908` on Arc and
`/home/spring/mask-20260908` on Thor. Thor image:
`spring-turret-demo:0.16.8-mask-agxthor-5`, layered over the previous `0.16.7`
image with the new profile and mask-capable viewer assets only.
