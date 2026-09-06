# Live throughput check, 2026-09-06

Host: `spring-edge-2` (`100.95.161.99`), Arc B580, Torch 2.14.0+xpu,
YOLO26x, compiled FP16 with SYCL graphs, 640x640 model input.
Camera: Arducam 1080P Low Light, 1280x720 MJPEG at 30 FPS. V4L2 advertised
no mode above 30 FPS, including the smaller resolutions. No camera, model,
precision, thresholds, servo limits or persistent calibration settings changed.

| Measurement | Result |
| --- | --- |
| Before, motors armed, 6 seconds | 10.25 detection FPS; 15.51 ms model; 31.17 ms worker total |
| Before, motors stopped, 8 seconds | 29.93 detection FPS; 8.72 ms model; 25.25 ms worker total |
| Intermediate metadata long-poll + separate JPEG downloads, tailnet, 10 seconds | 30.07 detection FPS; 16.39 downloaded FPS |
| Final paired event stream, tailnet, 20 seconds, 601 distinct frame IDs | 30.11 detection FPS and 30.11 downloaded FPS |

The final measurement had motors **stopped**, with the `mouse` class selected
and no objects of that class visible. The small excess above 30 reflects network
arrival timing/window endpoints, not extra camera exposures. This measures full
JPEG arrival over the network, not browser compositor presentation or exposure-
to-screen latency. The browser showed about 30 detection FPS and a working image.
Hardware-active throughput after the change still needs an operator-started run;
do not attribute the entire armed-to-stopped improvement to software alone.

Final warm timing, milliseconds (median / p95):

| Stage | Median | p95 |
| --- | ---: | ---: |
| Preprocess | 8.43 | 8.58 |
| Model | 8.69 | 8.90 |
| Postprocess | 0.37 | 0.42 |
| JPEG annotation | 0.00 | 0.00 |
| Worker total | 17.51 | 17.79 |
| Worker round-trip | 19.79 | 21.05 |
| Camera wait | 12.64 | 16.74 |
| Detection cycle | 32.19 | 36.39 |

Changes remove serial reads from inference, move JPEG overlays into the browser,
and stream exact image/metadata pairs on one connection. Position, torque and
hardware-fault checks remain in monitor/command paths; stale encoder snapshots
still prevent tracking guesses. CPU tests cover nonblocking pose access, cache
expiry, stop/fault invalidation, exact-frame overlays, HTTP streaming, stale
result rejection, and bounded image decoding. Model tensor computation and
accuracy gates are unchanged. This scene's three startup validation frames had
zero retained detections, so they are not nonempty detection-parity evidence.

Deployment is version 0.13.1 with rollback releases retained. SHA-256 of the wheel:
`8a2fd15d83d14ded80e47703b95b151393e88cde9a8ebc561649ebb5edc4f311`.
Both config and saved servo-zero file hashes were checked unchanged across the
restart. Neither axis was armed by deployment or benchmarking.
