# NVIDIA Thor deployment

This is the same camera, two-axis controls, calibrated pointing, click retargeting,
multi-category boxes, counts and FPS UI, using dense SAM 3.1 on CUDA instead of
the Intel native/W8A8 implementation. No TensorRT export or additional quantization.
The current CUDA deployment does not expose the Intel-only YOLO worker.

## Inference contract

- Original `sam3.1_multiplex.pt`; SHA256
  `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`.
- SAM source commit `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`.
- FP32 master weights, FP16 autocast, TF32 and reduced-precision FP16 accumulation
  disabled. Original 1008×1008 input and confidence threshold retained.
- `torch.compile(backend="inductor", fullgraph=True, dynamic=False)` for image
  normalization, image encoder, text encoder and grounding head; explicit CUDA
  graph capture/replay on each stage. Inductor's own graph capture is disabled
  to avoid nesting graph systems. `emulate_precision_casts=True` is retained.
- Image features are computed once per camera frame and shared across categories.
  Text embeddings are cached; grounding runs once per category with batch size 1.
  Masks, video propagation and SAM's video tracker are not run.
- Original preprocessing is checked bitwise. Compiled graph replay is checked
  against compiled direct execution, and detection outputs are checked against
  the full eager detector. Failures stop inference; there is no unqualified opt-in
  for the CUDA setup. This is runtime parity checking, not a dataset-wide accuracy
  certification.

Intel configurations still default to `device_type: "xpu"`; CUDA requires
`"device_type": "cuda"` in `inference`. Status reports `cuda_graph: true`,
`sycl_graph: false` only after capture and validation complete.

## Container and host layout

`deploy/thor/Dockerfile` uses NVIDIA's aarch64 PyTorch 25.08 container, preserving
its vendor Torch/Torchvision/Triton packages. Build from the repository root:

```sh
sudo docker build -f deploy/thor/Dockerfile -t spring-turret-demo:0.16.0-thor .
```

The launch script assumes the verified agxthor-4 IDs: application UID/GID 1000,
video GID 44, dialout GID 20. Adapt these IDs before using another host. The
application runs without root or Linux capabilities; Docker receives only the
matched camera and serial device, not a privileged container or Docker socket.

Required host files:

- `/etc/spring-turret-demo.json`: demo configuration; inference Python
  `/usr/bin/python`, checkpoint `/models/sam3.1_multiplex.pt`, cache `/cache`,
  `device_type: cuda`, `model: sam3.1`, precision `float16`. Remove Intel bundle
  and YOLO checkpoint fields. Keep the existing camera/servo/tracking settings.
- `/opt/spring/turret-demo/models/sam3.1_multiplex.pt`: authorized checkpoint,
  mounted read-only. It is not included in Git or the container image.
- `/etc/spring-turret-geometry.json`: camera/mount calibration, read-only.
- `/var/lib/spring-turret-demo/servo-zeros.json`: physical assembly zero offsets.
- `/var/cache/spring-turret-demo`: writable compiler cache owned by UID 1000.
- Existing serial-specific udev rules providing `/dev/spring-turret-camera` and
  `/dev/spring-turret-servo`.
- Launch script at `/opt/spring/turret-demo/run-container.sh` and systemd unit
  `deploy/thor/spring-turret-demo.service` in `/etc/systemd/system/`.

Transfer geometry and zeros only with the same physical camera/servo assembly.
The device-number sysfs lookup supports container aliases and still verifies the
USB camera serial. New hardware must be calibrated separately. Motors start off;
installing or testing inference never auto-arms, homes or changes zero offsets.

## Verification and operations

Verified on `agxthor-4` (NVIDIA Thor, GPU
`GPU-a7c66ad2-6dbb-0ab8-c1a2-37ba6dba3600`, MAXN, driver 595.78),
with PyTorch `2.8.0a0+34c6371d24.nv25.08`, Torchvision
`0.23.0a0+428a54c9` and CUDA 13.0. The application is served at
`http://100.97.193.16:8080/` by the enabled `spring-turret-demo` service.

[Camera-frame qualification](sam31-thor-qualification.json) passed all replay,
shared-feature, multi-prompt, changed-image and cache checks. Fifteen warm
iterations per configuration gave these median model times:

| Categories per frame | Model time |
| --- | ---: |
| 1 | 112 ms |
| 2 | 127 ms |
| 4 | 158 ms |

The image encoder was approximately 96 ms. These are model timings, not camera
throughput. The standalone test also draws/encodes annotations; the live UI
instead uses client overlays and CPU prefetch. Live display was observed around
7.5 FPS with 113 ms model / 142 ms loop on one prompt; it varies with prefetch
hits and interactive prompt changes.

Camera and both servo identities, torque-off startup, saved zero loading and
the fisheye mapping were verified. Initial geometry and zero files were copied
bit-for-bit. Subsequent recalibration and active tracking were performed through
the UI; those user changes were left untouched. No automated motion test was run.

`tests/smoke-sam31.py --device-type cuda` tests the real detector, changed images,
multiple prompts, cache ownership and stage reuse. It has no servo access. For a
short hardware qualification, use `--batch-sizes 1 --prompt-counts 1 2 4` with
`--iterations 15 --image <camera.jpg> --report <report.json>` and the checkpoint.

```sh
sudo systemctl status spring-turret-demo
sudo journalctl -u spring-turret-demo -f
sudo systemctl restart spring-turret-demo
sudo systemctl stop spring-turret-demo
```

Cold compilation takes longer than normal startup. Submit a prompt, wait for
validation, then confirm actual `/api/detection/status` results contain both
`torch_compile: true` and `cuda_graph: true`, with current frame sequences. Use
the displayed loop time/FPS separately from model latency.
