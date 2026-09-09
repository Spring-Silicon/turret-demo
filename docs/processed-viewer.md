# Completed-result viewer

The standalone Arc, standalone Thor and combined mounts share `static/app.js`.
Each panel displays only completed worker results from `/api/detection/events`
(paired JPEG and detection metadata). HTTP frame fetching is a reconnect
fallback for those same processed results, not a raw-camera fallback.

JPEG and optional indexed mask PNG are decoded and composited offscreen together.
The completed raster is drawn onto one persistent visible canvas, with frame
metadata and box targets updated in the same task. Normal frames never clear,
resize or replace the visible canvas; a real resolution change prepares a new
surface before swapping it in. A failed decode or draw, model switch, camera loss
or stalled worker leaves the preceding complete result visible. An offline status
cannot cover an existing result with the black placeholder. Old-revision pictures
are not clickable. No timers interpolate camera frames, and raw `/stream.mjpg` is
never requested by the UI.

Status polls may describe N+1 while the image stream is delivering N. The viewer
accepts that completed N result independently of the newer status, so polling
cannot starve a slower backend's display. Backend process changes also reset
sequence ordering without exposing old-instance click targets to the new process.

Each device advances independently. There is one active decode and only the
latest next result; a slow browser/network can skip completed results, but cannot
queue an increasing video delay or slow the inference/control process. Canvas
`data-frame-key` attributes expose the exact model/revision/sequence pair for
read-only validation.

## Mask selection

`sam3.1-mask` and `sam3.1-tracking` show no bounding rectangles or box labels.
The selected instance's mask is red; another instance under the pointer or
keyboard focus is lighter red. Other masks retain their per-instance colors.
Class selection uses the controller's reported instance when available. Hover
only repaints the same completed JPEG/mask pair: it does not request a new frame,
change tracking, or send a motor command.

Pointer hit testing uses the visible mask pixels, including overlap composition,
not invisible bounding rectangles. Clicks retain the exact displayed frame and
instance ID. Keyboard-accessible instance buttons remain, without box outlines.
Palette colors are matched once per decoded color with a small tolerance for
canvas alpha rounding; ambiguous colors cannot select an unintended instance.
All of this runs in the viewer; model outputs and the control loop are unchanged.

## Arc native tracking cache

The pinned native model remains hash-verified. An application-level cache policy
preserves native stage hooks, exact input signatures and owned temporal outputs.
Fixed stages capture on first use. Variable temporal stages can capture frequently
repeated signatures during their first 64 calls, retaining no more than the
original cache capacity. After 64 completed worker frames all caches are frozen,
including rarely-used stages that have not yet run. Prompt/session resets do not
reopen the capture window.

Cache hits use graph replay. Misses run the same tensor region under
`torch.compiler.set_stance("eager_on_recompile")`: valid compiled operators are
reused; shape changes cannot force synchronous compilation/capture during the
ongoing session. Uncached shapes can be slower than replay. Initial model loading
and graph warmup still take time. Weights, resolution, thresholds, memory policy
and instance association are not changed by this policy; compiled/direct floating
point differences still require GPU replay qualification.
