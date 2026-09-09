# Offline-compiled Arc mask startup

`inference.sam31_mask_compiled_bundle` selects a package exported from the
qualified Arc mask worker. Startup loads its generated execution code, frozen
weights with their original strides, argument/output mappings, and kernel caches.
The model's native operators are registered normally. The live SYCL graph is
still captured in the new process; this package does not serialize device
pointers, streams, or an `XPUGraph` object. There is no dependency on the graphs
repository.

The package uses the exact existing `torch.compile` output, including weight
freezing and precision-cast settings. It bypasses Dynamo tracing and Inductor
schedule generation for the image, FPN, grounding, mask and output stages.
Image normalization and the small shape-specific postprocessor retain their
normal compilation/replay path. Text embeddings remain cached per prompt.

## Export and activation

Run `tools/export-sam31-mask.py` with the **inference venv's Python**, the same
graphics environment as the existing mask worker, and live inference stopped.
Use an existing JPEG; the exporter has no camera or motor access. For an existing
installation, `--worker-directory` must point at the exact proposed installed
Python sources, including the new artifact module and worker.

```sh
python tools/export-sam31-mask.py \
  --worker-directory /path/to/proposed/spring_turret \
  --checkpoint /path/to/sam.pt --mask-bundle /path/to/native-mask-bundle \
  --frame /path/to/existing-frame.jpg --prompt ball \
  --output /path/to/new-compiled-mask-package
```

Export never overwrites a package. It publishes a complete staging directory only
after successful normal execution and graph replay validation. Before activation,
compare a fresh process loading the package against the original worker on changed
images and prompt counts. Check exact masks, scores, query indices, counts and
boxes, and prohibit compiler calls inside the loaded stages. Measure startup and
steady inference separately.

Set the absolute package path in `inference.sam31_mask_compiled_bundle`, then
restart the backend. The package is XPU-only and is ignored by other model profiles.
The worker reports `compiled_mask_artifacts: true` on completed results. Removing
the setting and restarting restores ordinary compilation.

A package must be regenerated when its worker/model source hashes, native bundle
manifests, Torch/Python build, device or confidence setting change. Every packaged
file is checked before loading. A missing, corrupt or incompatible explicitly
selected package fails with an error; it does not silently compile a replacement.
Generated Python and compiler cache files are executable artifacts. Only use
packages created by the trusted offline exporter; hashes detect corruption and
drift, not a malicious publisher.

## Graceful Arc restarts

Install `deploy/startup/spring-turret-demo-graceful-stop.conf` as
`~/.config/systemd/user/spring-turret-demo.service.d/graceful-stop.conf` and run
`systemctl --user daemon-reload`. `KillMode=mixed` sends the initial stop signal
only to the backend, allowing its existing inference drain to finish before
systemd reaps descendants. The default cgroup-wide signal was observed to fault
Arc's driver while a mask kernel was in flight. The 45-second timeout retains
forced cleanup for a stalled process. Restart never re-arms motors.

## Arc deployment qualification — 2026-09-09

Implementation commit `421d665` was pushed on `user/demo-changes`, pulled on Arc,
and installed with the locally exported 408 MiB package at
`~/.local/share/turret-demo/sam-artifacts-20260909/package`. Its manifest SHA-256 is
`5776131844bf7f8a134b45ee3126e5d5df6b4701dea8924aecc7e56898a9730f`.
The sibling `deployment-backup` directory contains the previous configuration,
replaced sources and a receipt of installed file hashes. Only the worker,
artifact loader, hardware configuration and detection configuration modules were
replaced; the package matches the existing installed graph helper exactly.

Both boxes were rebooted before and after activation. Their startup services
launched the backends and default `person` inference without commands after boot;
Arc also launched its frontend and Firefox kiosk. Motors remained disarmed.

| Measurement | Original path | Saved stages |
| --- | ---: | ---: |
| Arc kernel boot to first processed frame | 84.4 s | 44.6 s |
| Arc kiosk painted to first processed frame | 72.4 s | 32.0 s |
| Thor kernel boot to first processed frame (unchanged control) | 102.2 s | 101.4 s |
| Arc steady model GPU time, median of boot observation samples | 65.08 ms | 64.83 ms |

These are one before/after reboot pair, with the existing persistent caches in
both cases. Kernel boot times exclude firmware and shutdown. A read-only observer
polled status every 200 ms plus request time; kiosk readiness came from the
service journal. Arc's first observed result had `compiled_mask_artifacts: true`,
`sycl_graph: true` and bitwise postprocessing validation enabled. All startup
units were active, Firefox remained held, and the Arc kernel journal had no GPU
reset or page-fault errors during the measured boot.

Separate fresh-process tests compared 21 cases using existing recorded images,
including changed/empty scenes and prompt counts 1, 2, 8, then 1 again. All raw
output tensors matched exactly, including masks, scores, counts, indices and
boxes. The candidate test made calls to `torch.compile`, Inductor `compile_fx`
and the Triton compiler entry point raise inside loaded stages. The original
path took 59.5 s from engine construction start to first result; the saved stages
took 26.1 s. A further candidate run with empty runtime cache directories passed
all 21 comparisons, taking 72.5 s to its first result. The package does not remove
all first-use runtime/cache initialization, and that empty-cache measurement is
separate from the normal reboot comparison.

The full repository validation passed. All six artifact tests also passed in
Arc's actual inference environment, including preservation of non-contiguous,
gapped, expanded, offset and empty weight layouts. An earlier exploratory
eight-prompt run of the original worker raised `Nonfinite mask result`; the
completed original and candidate qualification runs had no errors. This change
does not claim to fix that pre-existing model behavior.

This deployment accelerates the `sam3.1-mask` profile. The separate temporal
`sam3.1-tracking` profile retains its existing startup path.
