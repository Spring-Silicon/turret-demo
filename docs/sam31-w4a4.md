# Sleepy-joe W4A4 detector

Non-tracking `sam3.1` can use the explicitly approved
`model-fixed-fc1-barrier-gpu1` candidate from sleepy-joe. This does not modify
SAM 3.1 Tracking, camera geometry, or servo zeros. Motion policy is now shared
across backends; see [the audit](sam-policy-audit.md). Historical YOLO references
below describe earlier qualification, not the current three-SAM model registry.

The retained source measurement is **67.813 ms complete resident GPU model**
for one cached `person` prompt (60.173 ms image, 7.750 ms grounding measured
separately). Preprocessing, transfers, CPU decoding/filtering and display are
excluded. Do not present this as camera-to-screen latency.

This is **not accuracy parity**: person AP50:95 is 65.344 versus dense 66.081 on
512 COCO images, a 0.736-point loss. General text-prompt accuracy is not qualified.
The user explicitly accepted this tradeoff. No further quantization or routing
change is introduced in deployment.

## Model and execution

- Original SHA256-pinned SAM 3.1 checkpoint, FP32 masters, FP16 autocast.
- Group128 signed W4A4 MLPs with alpha0.5 SmoothQuant and native integer kernels.
- Gradient token routing retains87.5% of tokens from block16; full residual grid
  and original rotary coordinates remain.
- Exact selected fixed/unrolled FC1, barrier-removal attention, compensated
  norm/pack, K32 QKV, native routing, and gather/window normalization kernels.
- Full grounding fusion/decoder layers, all200 queries, packed-mask attention
  and the six oneDNN FFN epilogues. No mask or temporal tracker execution.
- `torch.compile(fullgraph=True,dynamic=False,emulate_precision_casts=True)`
  and one explicit SYCL graph spanning image plus all active prompt heads.
  One image pass per frame; each distinct text prompt gets a separate head pass.
  Text is encoded eagerly once and cached, matching the retained benchmark;
  it is not part of per-frame GPU inference. Only the active prompt-count
  graph is retained.
- Existing exact full-frame1008×1008 preprocessing, CPU prefetch and paired JPEG
  overlays remain. Neither camera cropping nor input resolution changes.

## Reproduction / isolation

`tools/collect-sam31-w4a4.py` runs on sleepy-joe without executing a GPU workload.
It freezes the benchmark's source snapshot,23 DSOs (including a transitive
normalization import), calibration, exported head, tokenizer, licenses and eight
source-result fixtures. Copy that output directory to the destination.

`tools/relocate-sam31-w4a4.py /absolute/bundle` verifies the original manifest,
creates a separate `runtime/` copy and replaces17 compile/load steps with exact
pinned DSO loads. Operator functions and all kernel binaries are unchanged.
The original snapshot remains intact. `relocation.json` records each change;
both original and runtime manifests are SHA256-pinned by `sam31_w4a4.py`.
Runtime verification includes the base checkpoint and every manifested file.
No download, native compilation, system library installation or source-host
dependency is needed at runtime. PyTorch must match the measured2.14.0+xpu build
and the GPU must be Arc B580.

Configuration under `inference`:

```json
{
  "model": "sam3.1",
  "sam31_w4a4_bundle": "/absolute/copied-bundle",
  "sam31_allow_w4a4_accuracy_tradeoff": true,
  "precision": "float16"
}
```

Remove the mutually exclusive `sam31_native_bundle`,
`sam31_w8a8_development_bundle` and `sam31_allow_unqualified_w8a8` settings.
Preserve `sam31_tracking_bundle` explicitly: it continues to identify the
unchanged temporal source and existing graphics runtime. The W4A4 worker gets
an isolated compiler-cache subdirectory and `TORCHINDUCTOR_FREEZING=1`; other
workers retain their previous settings. The existing inference venv is reused.

Qualification must compare changed images and return-to-first against source
predictions, verify direct/replay agreement, test fresh/multiple prompts and
measure the actual live worker separately. These are copy/replay checks, not a
new AP benchmark. Keep motors stopped during deployment and verification.
The source-result check requires identical retained query selection at0.5,
confidence error<=0.002 for either score>=0.47, and box error<=0.001 there.
All-query errors are recorded separately: unconstrained rejected query boxes
are not required to be bitwise identical across compiled deployments.

Run `tests/qualify-sam31-w4a4.py --bundle /absolute/bundle --checkpoint
/absolute/checkpoint --output /writable/report.json` in the inference venv with
the same environment and render-device group as the worker. It does not access
camera/servo devices. Stop other inference workloads before collecting timings.

## Arc deployment verification (2026-09-07)

Release`0.15.9-w4a4-verified` is active on spring-edge-2. On50 live1280×720 camera
results with one`table` prompt: median GPU model67.09ms, worker69.51ms, server
loop70.06ms and14.05FPS. CPU preprocessing is overlapped by the existing prefetch
path; these are not glass-to-glass timings. Warm fixture measurements were66.83ms
GPU/73.35ms worker for one prompt and74.58ms/81.22ms for two.
The original prompt/target, camera geometry and servo zeros were retained;
motors remained stopped. Tracking/YOLO source files were checksum-preserved.
