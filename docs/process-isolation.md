# Backend / viewer process separation

`spring-turret` starts the backend, which owns the camera, servo monitor,
frame-driven tracking controller and GPU-worker subprocess. It does not bind
the public HTTP port. A separate `python -m spring_turret.viewer` child owns
the HTTP server, SSE/MJPEG delivery, image encoding for transport and viewer
caches. It does not instantiate a camera, servo controller or inference worker.

## Data and commands

- One backend publisher atomically replaces a complete metadata + raw-camera
  JPEG + exact detection-frame JPEG snapshot in a mode-0700 temporary directory.
  Linux prefers `/dev/shm`; other systems fall back to their temporary directory.
- There are no cross-process locks, queue acknowledgments, or waits for a reader.
  Filesystem work happens outside camera/servo/detector locks. A slow or stopped
  viewer simply misses intermediate snapshots. The producer has at most the
  current and in-progress files; each JPEG is capped at 8 MiB and metadata at 1 MiB.
- Control state is sampled at 5 Hz, independent of viewer count; frame snapshots
  are refreshed when camera or detection sequence changes. Publication does not
  pace inference or motor commands. The publisher and viewer check for updates
  every 5 ms; this is viewer latency, not an inference/control delay.
- The viewer owns its eight-frame JPEG cache, 32-connection bound, all network
  waits, SSE base64/JSON encoding and HTTP status serialization. GET requests do
  not send requests to the backend or acquire its locks.
- Explicit Start/Stop, sliders, prompt/model selection, target clicks and zero
  calibration use a private Unix socket. The backend validates the same API
  fields and hardware constraints as before. Input is bounded; queued commands
  older than two seconds are rejected. Commands are never automatically retried
  after a missing reply. This is not a detection-age or browser-heartbeat cutoff.
- The legacy keepalive endpoint remains a no-op. Neither camera publication nor
  inference depends on having a connected browser.

## Failure behavior

- Viewer exit: backend restarts only the viewer; inference/tracking state and
  motor arming remain unchanged. Restarting the whole service still disarms and
  resets the model session. It never automatically re-arms.
- Frozen/slow viewer: backend keeps running without waiting for it. Snapshot
  writes remain bounded; no backlog accumulates for that viewer.
- Backend failure: the viewer does not pretend that torque-off was confirmed.
  After two seconds of stale snapshots it reports backend unavailability and
  unknown motor state; it exits when its backend parent disappears. This is a
  viewer status check, not a motor timeout. The hardware bus watchdog remains.
- Explicit Stop, angle limits/recovery, fault detection and service-shutdown
  torque-off are unchanged. All physical motion remains backend-owned.

`/api/status.runtime` reports backend, viewer and model-worker PIDs plus snapshot
age for diagnosis. The processes still share the machine's CPU, memory and OS
scheduler: this is process isolation, not dedicated-core or hard-real-time
resource isolation, and not a security boundary between separate Unix users.

## Tests

`tests/test-isolation.py` exercises atomic frame pairing, stale-backend reporting,
Stop reply ordering, expired command rejection, and real backend/viewer process
separation with simulated hardware. It freezes and restarts the viewer while
checking that backend inference sequence and simulated arming continue, then
confirms explicit Stop. Existing hardware-fault, bounds, API and UI tests remain.
