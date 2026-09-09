# Arc mask-dropout investigation — 2026-09-09

The user reports physical turret jerks coinciding with brief ball-mask flickers.
Both unsuccessful controller/P800 trials are reverted; see
[the moving-response experiment](arc-moving-response.md). This investigation has
not yet produced a qualified live fix.

## Saved-frame observations

A 65-second capture retained processed JPEGs, masks and timestamps from both
hosts. Arc published 921 distinct results (574 with a ball); Thor published 494
(347 with a ball). Arc had 99 internal detection gaps, including 95 shorter than
0.5 seconds; Thor had 49, including 47 shorter than 0.5 seconds. The cameras saw
different images at different rates, so these counts do not isolate a hardware
or numerical-precision cause. Visual inspection confirms the ball remains in
view on some missing-mask frames.

The current controller sends a measured-position hold on the first missing
mask, then a new absolute pointing goal when a mask returns. This is a concrete
stop/reacquire mechanism. Its contribution to the user's perceived jerkiness
still needs synchronized live validation.

## Exact-image replays

440 saved Arc JPEGs were replayed with prompt `ball` and diagnostic confidence
0.05. Filtering replay results at the production threshold (>0.5) reproduced all
327 original visible detections, with exactly equal scores; none of the 113
original missing detections became visible. These sampled dropouts therefore
reproduce in the model without live CPU prefetch. They are not explained by a
prefetch/image mismatch in this replay.

| Model recipe and score cutoff | Frames with a valid mask | Internal missing frames | Internal gap runs | Frames with multiple masks |
| --- | ---: | ---: | ---: | ---: |
| Current skip4/attention8, >0.5 | 327 | 78 | 44 | 0 |
| Current skip4/attention8, >0.4 | 345 | 61 | 38 | 0 |
| Current skip4/attention8, >0.3 | 358 | 48 | 31 | 3 |
| Current skip4/attention8, >0.2 | 372 | 34 | 26 | 27 |
| Earlier 32-block native mask, >0.5 | 337 | 69 | 35 | 0 |
| Earlier 32-block native mask, >0.3 | 366 | 40 | 30 | 0 |

The two independently sampled contiguous runs are evaluated separately; leading
and trailing absence does not count as an internal gap. These counts indicate
mask availability, not labeled detection accuracy. Lower thresholds also expose
additional spatially separate candidates and do not recover every visible ball.
The earlier model recovers 18 original missing detections but loses 8 original
visible detections at 0.5. Its median GPU model time is 70.47 ms versus 61.96 ms
for the current model in these sequential offline runs. This comparison does not
establish full dense-model accuracy or a live motion improvement.

The earlier model was loaded from an isolated copy of the existing verified base
bundle, without its optional attention8 extension. Production model artifacts,
configuration and control source were not modified. After each replay, the
normal model, prompts, ball target and prior Start intent were restored and
running inference/no servo error verified. Arc remains P1200/I0/D1600 on both
axes, velocity/acceleration profiles zero, production confidence 0.5 and the
skip4/attention8 recipe. Thor was not interrupted.

## Next live test

The user requires advance notice and an explicit readiness reply before any new
recording. No capture may start merely because a timer expired. After readiness,
give a clear start cue and collect a timed stationary/moving/stationary baseline
before changing another setting. The planned recording has not started.

Local scripts, original JPEGs, metadata, replay manifests/results and comparison
analysis are in `/home/ubuntu/work/turret-motor-20260909/`. Arc's replay outputs:

- `/home/spring/.local/share/turret-demo/confidence-replay-20260909T063054/`
- `/home/spring/.local/share/turret-demo/base32-confidence-replay-20260909T063423/`

The first attempted confidence replay failed before producing results because
its SSH environment lacked the service's OpenCL library settings. The successful
replays copied only those compute-library variables from the running backend;
the failure provides no model-quality evidence.
