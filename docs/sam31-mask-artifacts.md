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
