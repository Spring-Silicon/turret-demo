# Native SAM image from sleepy-joe

This is the qualified **dense FP16/BF16 native `graphs` image**, not Israel's
experimental W8A8 image. Only the once-per-frame image stage changes. The
cached text encoder and one grounding pass per class retain Torch 2.14 XPU,
`torch.compile`, and explicit XPUGraph replay. Every instance returned by a
class pass remains available. No mask/video/tracker branch is introduced.

## Source and identity

On `spring@sleepy-joe`, the source is
`/home/spring/springsilicon/graphs/outputs/sam31-turret-image/`:

- Artifact: `native-attention-prefetch1`, fingerprint
  `c98974a61db4e8e0a73b825c60b85862406d934cd8631c7189df8eb2b77c051a`.
- Modules: `runtime-attention-prefetch1/`, not the current experimental build.
- Artifact/compiler closure: `attention-prefetch1-build-closure.json`.
- Accuracy receipt: `tla-attention-box-validation/report.json` (20 passing
  image/prompt checks). The selected artifact inherits those checks through
  bitwise equality on five images plus return-to-first replay. These are
  development fixtures, not an independent accuracy dataset.
- SAM commit: `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`.
- Audited demo base: `777516b1d28a0ba4ce82afff57760b45ce9b0c6b`.

The worker pins the checkpoint, artifact manifest, runner, native modules and
bundled oneAPI runtime dependencies by SHA256 in `sam31_native.py`. Artifact
objects are additionally checked by the runner's content-addressed loader.
An updated source binary needs an explicit pin update and requalification;
copying a later experiment will not silently change the deployed model.

The app uses `graphs-runner-intel-turret`: the same image artifact and native
modules, with host-only dense-output and channels-last input copy fast paths,
plus actual per-response graph-count/timing receipts. The original runner spent
roughly 100 ms on per-element host layout conversion on this host.
`native/runner-host-copy.patch` retains the fix and CPU tests (offset/bounds,
transposed/dynamic shapes, and channels-last input with one/two batches).
No GPU kernel or model arithmetic changes. This was built in an isolated
snapshot, without editing sleepy-joe's active optimization checkout.

The runner binary, patch and complete build source are retained on sleepy-joe
at `/home/spring/turret-demo-native-runtime/20260906/` and copied with the bundle.
`runner-source.tar.gz` SHA256 is
`6a245f181b521b79347c3e3d159f6688de56af83cb2eecc2013336b571ca58f6`.
Extract it into an empty directory and use Cargo 1.98.1:

```sh
cargo test --locked -p cli --no-default-features --features intel --bin graphs-runner-intel
cargo build --locked --release -p cli --no-default-features --features intel --bin graphs-runner-intel
```

Native module builds are disabled; the original qualified `.so` files remain
unchanged. A rebuilt binary may have a different digest; requalify before
updating the pin. The native stage requires one command graph in every response.

## Execution boundary and latency

Input is normalized FP32 `[1,3,1008,1008]`; outputs are FP16 final feature and
position tensors, each `[1,256,72,72]`. FP32 master weights/residuals and the
upstream BF16 addmm-before-GELU rounding are retained.

The native runtime has different SYCL dependencies than PyTorch. A persistent
child process isolates its library loader and holds one resident native SYCL
graph. Private `/dev/shm` buffers transfer input and output between processes.
This **is not zero-copy GPU interop**: the runtime uploads the image, downloads
the two feature tensors, and Torch uploads those features for grounding.
`timing.image_encoder_ms` includes this IPC and transfer overhead. The source's
approximately **87.3 ms** is native image replay only, not app latency.

Before serving, the native stage runs direct and replay qualification on the
destination, requires exactly one actual command graph, and compares outputs
at atol=rtol=0.001. The app's independent dense full-pass confidence (0.03) and
retained-box (0.01) gates remain unchanged; no tolerance is relaxed and no
quantized model is substituted for the reference. A failure stays an error,
not a silent fallback. As before, runtime full-pass checks are startup/batch
qualification, not an every-frame accuracy guarantee.

`image_backend=graphs-native-sycl` and `native_image_validation` identify the
selected image implementation in results. `torch_compile=true` refers to the
unchanged text/grounding stages; the UI distinguishes native image execution.
The initial deployment contract is one B580 GPU and Torch `2.14.0+xpu`.

## Copy, qualify, activate

From a checkout, with SSH access to both hosts:

```sh
bash scripts/copy-sam31-native.sh spring@sleepy-joe spring@spring-edge-2-1
```

This creates a new directory below `/var/lib/spring-data/turret-inference`
and prints its path. It does not overwrite an active bundle, install an OS,
copy credentials, change system-wide libraries, restart the app, or move a
servo. Keep the included SYCL-TLA/oneAPI/oneDNN license notices with the bundle.
The bundle contains checkpoint-derived weights; do not publish it publicly.

Copy this checkout's `src/` and `tests/` to the destination, then run with its
existing inference venv (use a new output path and dedicated compile caches):

```sh
OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2 \
TORCHINDUCTOR_CACHE_DIR=/absolute/test-cache/inductor \
TRITON_CACHE_DIR=/absolute/test-cache/triton \
/var/lib/spring-data/turret-inference/venv/bin/python \
  tests/qualify-sam31-native.py \
  --bundle /absolute/copied-bundle \
  --checkpoint /var/lib/spring-data/turret-inference/sam3.1_multiplex.pt \
  --output /absolute/new-qualification.json
```

The test checks five changed images plus return-to-first, exact stored native
feature bits, 24 independent full-detector comparisons, and 1/2/4-class timing
with one image replay and no text replay on cache hits. No camera or servo is
accessed. Camera/JPEG/API latency needs a separate full-worker test.

After successful qualification, install the new app release and add only
`inference.sam31_native_bundle: "/absolute/copied-bundle"` to the existing
device config. Keep `precision: "float16"`, hardware calibration and all other
settings. Make the bundle readable/executable by `spring-turret`, back up the
config and release symlink, and restart `spring-turret-demo`. Startup is never
armed. Removing that config key restores the existing compiled image path;
reverting the config and release symlink restores the previous app release.

Local regression tests include bundle tampering, image boundary checks,
short reads/EOF/timeouts, SAM-only routing, and worker-group cancellation.
Workers and native descendants are stopped together; parent-owned shared
memory is reclaimed even when a compilation has to be killed.

## Destination verification: spring-edge-2, 2026-09-06

Installed release `0.11.0`; tailnet DNS is currently `spring-edge-2-1`.
The [qualification receipt](sam31-native-qualification.json) passes all 24
detector checks and five-image/return-to-first bitwise feature comparisons.
Largest checked confidence error is 0.006196; largest retained box-coordinate
error is 0.000039 (limits remain 0.03 and 0.01).

The [installed-worker receipt](sam31-native-worker-smoke.json) used the saved
1280x720 camera JPEG, the real JSON worker protocol, and the service's user,
groups and filesystem hardening. Medians after warmup (six samples):

| Boundary | One class | Two classes |
|---|---:|---:|
| Native image replay | 87.14 ms | 87.12 ms |
| Image stage including host IPC/transfers | 107.02 ms | 107.05 ms |
| Grounding | 15.50 ms | 29.22 ms |
| Full worker including JPEG processing/annotation | 145.73 ms | 158.24 ms |
| JSON request roundtrip | 148.12 ms | 160.79 ms |

Two person instances remain present; changing/reordering the class list keeps
counts and cached embeddings coherent. Stopping the worker removes its native
process and shared-memory directory. These figures exclude camera acquisition,
browser network/rendering and physical tracking; camera and servo were not
connected during deployment. No servo commands or hardware calibration changes
were part of this update.

Active bundle: `/var/lib/spring-data/turret-inference/sam31-sleepy.6zk2ky`.
Rollback config: `/etc/spring-turret-demo.json.before-native-0.11.0`.
Previous release remains `/opt/spring/turret-demo/releases/0.10.0/venv`.
