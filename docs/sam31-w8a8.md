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
