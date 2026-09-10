# Shared backend API (version 1)

Arc and Thor expose the same HTTP contract through a shared viewer-boundary
normalizer. Their inference implementations differ (CUDA versus XPU/native), but
target selection, control and lifecycle policy programs are shared. Assembly
calibration and hardware transport configuration can differ.

Full status, detection-only status, SSE metadata and command replies include the
same detection defaults, `execution`, `timing`, and `pipeline_timing` fields.
Unavailable measurements/capabilities are null, not invented zero/false values.
The additive `detection.progress.phase` vocabulary is `disabled`, `idle`,
`loading`, `preparing`, `capturing`, `validating`, `waiting_for_camera`, `running`
or `error`. `progress.stage` retains the worker detail. Legacy `state` and
`progress_stage` remain for compatibility. Graph preparation while running keeps
the completed frame and reports "showing last result"; it is not necessarily a
fresh compilation. Error/camera-loss/idle states override old preparation stages.

Both panels label explicit `timing.model_ms` as **model**, temporal
`timing.tracking_ms` as **tracking**, and unclassified legacy `latency_ms` as
**inference**. Backend `pipeline_timing.cycle_ms` is **total**, not browser latency.
Missing measurements are omitted. Both masked modes count masks, not boxes.

## Requests

| Endpoint | Method | Body / meaning |
|---|---|---|
| `/api/status` | GET | Camera, servo, detection and tracking state |
| `/api/detection/status` | GET | Optional `revision` and `sequence` cursor; latest result |
| `/api/detection/events` | GET | SSE: exact-frame JPEG + metadata; slow readers skip to latest |
| `/api/detection/frame/{revision}-{sequence}.jpg` | GET | Bounded exact-frame history; 404 if evicted |
| `/stream.mjpg` | GET | Raw camera, independent of inference |
| `/api/detection/model` | POST | `{"model":"sam3.1-tracking"}` |
| `/api/detection/pause` | POST | `{"paused":true}` to pause, `false` to resume; idempotent |
| `/api/detection/prompts` | POST | `{"prompts":["hand","cup"]}` |
| `/api/tracking/target` | POST | `{"target":"hand"}` or null |
| `/api/tracking/instance` | POST | `{"revision":1,"frame_sequence":42,"instance_id":7}` |
| `/api/servo/arm`, `/api/servo/disable` | POST | No body required |
| `/api/servo/recalibrate` | POST | `{}`; requires stopped, stationary servos |
| `/api/servo/recover-gains` | POST | `{}`; while stopped, backs up/discards an invalid saved override; no register writes |
| `/api/servo/position` | POST | `{"axis":"x","degrees":20}` |

The legacy single `/api/detection/prompt` and `/api/servo/keepalive` endpoints are retained. Validation errors
use 400 + `{"error":...}`; disarmed/conflicting commands use 409; unavailable
hardware/backend uses 503. Never automatically retry a motor command after a lost
reply. GETs and model/prompt API updates never request Start. Device reconnection
can restore torque only when the backend already holds an explicit Start request;
Stop cancels it even while offline. Browser reconnection does not send Start.

Servo status distinguishes `run_requested` (latched Start intent) from `armed`
(actual enabled torque). `recovering` means Start is still requested while
hardware is unavailable/unverified. `can_start` permits queuing Start for a
commissioned assembly; it does not bypass hardware checks. See [device recovery](recovery.md).

## Common detection fields

The public menu contains exactly `sam3.1` (SAM 3.1 Box), `sam3.1-mask`
(SAM 3.1 Mask), and `sam3.1-tracking` (SAM 3.1 Mem), in that order.
These IDs are unchanged, so saved selections and clients remain compatible.
`implementation_model` retains the actual worker ID. A legacy `sam3.1-v18`
deployment reports public `model: sam3.1-tracking` without swapping its worker.
Direct legacy v18 API requests remain supported when configured.

`paused` is backend-owned inference intent; `pause_revision` orders changes
without resetting the prompt/temporal revision. Pause stops new frame submissions
and CPU prefetch; already-running GPU work or model loading finishes normally.
The worker, weights, compiled graphs, prompts and last completed frame remain
loaded. Resume samples a post-resume camera frame and discards an older in-flight
result. No model/prompt changes or browser reload implicitly resume inference.
A backend/device restart starts unpaused. The combined Pause/Resume button controls
both connected backends; individual pages control their own backend. A mixed
pause state offers Resume to align both. Failed peers are reported, not silently
retried. Model/prompt changes while paused take effect when resumed.

Automatic targeting holds position while inference is paused without clearing
Start intent or the target class. Stop still stops motors independently; Resume
inference never arms stopped motors. Camera capture remains live internally,
but the displayed image stays on the last fully processed frame, including
after a page reload while paused.

`api_version`, `model`, `models` (including per-device `available` capabilities),
`prompts`, `revision`, `state`, `error`, `frame_sequence`, `frame_url`,
`frame_age_ms`, `boxes`, `categories`, `mask_overlay`, `temporal_tracking`,
`active_instance_ids`, `retired_instance_ids`, `propagated_instance_ids`,
`latency_ms`, `fps`, `timing` and `execution` have the same semantics on both.

Boxes use normalized `[x1,y1,x2,y2]`, an applied `prompt`, score and instance ID.
Mask-producing workers also include `mask_centroid: [x,y]`, normalized to the
original camera frame using foreground pixel centers (`index + 0.5`). `null`
means an empty mask and is not a valid control target. Only omission of this
field permits box-center fallback for box-only models. Centroids are computed
from the original instance mask, never the composited/display PNG.
Mask/JPEG/box metadata refer to the same captured image. IDs are local to a device
and session: a clicked instance on Arc must not be sent to Thor.

`execution` has `device_type`, `device`, `precision`, `torch_compile`,
`graph_replay` (`cuda`, `sycl` or null), and `compilation_scope`. The existing
flat worker diagnostics remain for compatibility, but the frontend must not use
their presence to infer a different API.

Timing measurements are milliseconds, unavailable values are null. In temporal
SAM, `tracking_ms` includes CPU association/postprocessing and synchronized GPU
work; it is **not** pure GPU model time. `worker_total_ms` includes preprocessing
and mask display encoding. `frame_age_ms` measures capture-to-current-status age,
not just inference. `model_ms` stays null unless explicitly measured.

The tracking profile preserves IDs through short occlusion. After at least five
seconds **and** 16 missing frames, an invisible identity is retired and its slots
and memories freed. A selected target holds while its ID remains active, then
may reacquire the nearest instance of the same class after retirement. Centering
does not discard a temporal ID. No expiry or reacquisition arms a stopped servo.

## Shared target policy

`tracking.continuity` is `nearest-of-class` for Box/Mask and `temporal-id` for
Tracking, on both Arc and Thor. Box/Mask prefer a clicked instance while visible,
then resume nearest-of-class when it disappears or becomes centered. Their
frame-associated IDs are not persistent appearance identities. Tracking keeps
its selected native ID through centering and temporary loss, then reacquires
that class only after the ID retires. Missing detections hold position rather
than releasing torque or performing a search sweep.

The legacy `hold_reacquire` configuration key is accepted but does not select
the former Arc-only world-bearing policy. See the [policy audit](sam-policy-audit.md)
for the old differences and current shared rules. Model/prompt changes discard
instance IDs but preserve a still-applied target class and Start intent. Removing
the selected class or explicitly commanding manual movement clears that target.
Calibration, motor gains and physical angle limits remain assembly-specific.
