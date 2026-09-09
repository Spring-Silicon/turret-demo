# Arc motor gains validated against Thor — 2026-09-09

Arc now retains **P400/I0/D0 on both axes**, matching Thor's measured motor
gains. After the repeated 60-second live comparison, the user reported Arc was
"clearly smoother" and that very fast motion could still jerk, but Thor jerked
more. This is the accepted setting for this assembly and moving-ball demo.

The earlier P800 trial failed live and was reverted; it retained D1600. The
successful change lowers P from 1200 to 400 and D from 1600 to 0 on both axes.

## Original cued baseline

The user explicitly said ready before this 40-second recording. It retained
564 Arc results and 302 Thor results. Arc had 97 missing-ball frames and 58
transitions from visible to missing; Thor had 10 missing frames and 6 such
transitions. Neither device incurred a new servo retry or fault. The user
reported jerking during slow movement, while holding still at the end, and
some jerking at the beginning even with the mask visible.

Measured feedback and goal traces confirm frequent hold/reacquire transitions
on Arc, plus small stepwise corrections while the mask remains visible. During
the first 10 seconds, Arc's measured pan spans 1.14–2.02 degrees and Thor's
5.89–6.77 degrees. These are different viewpoints; equal overall span does not
mean equal instantaneous smoothness. Later motion phases are identified from
the footage itself: user response to a cue is not an exact timestamp.

An offline background-feature check compares calibrated ceiling-ray rotations
against the recorded encoder timeline. Arc's best fitted image delay is 8 ms
(2–8 ms across alternating-pair splits); Thor's is approximately 30–32 ms.
The median angular residual is 0.277 degrees for Arc versus 0.319 at zero delay;
Thor's is 0.235 versus 0.494. This approximate fit uses interpolated encoder
samples and rotation-only geometry; it is not an exposure-time measurement.
It provides no evidence of a large Arc-only timestamp mismatch. No timestamp or
camera-control correction is selected from it.

## Actual Thor registers

With the user's explicit authorization, Thor's backend was stopped for exclusive
serial access. The probe issued only model pings and register reads, then restored
the unchanged service, SAM3.1 Mask / ball target and prior Start intent. Advancing
inference and no servo error were verified afterward.

Both Thor motors read P400/I0/D0, feedforward 0/0, profile velocity 0 and profile
acceleration 0. Both use operating mode 4 and drive mode 0. Thus the deployed
Arc P1200/I0/D1600 gains are a verified difference, rather than an inference from
Thor's missing configuration overrides. Readback artifact:
`/home/spring/.local/share/turret-demo/thor-register-readback-20260909T065028/`.

The first read-only probe container could import its SDK but lacked the serial
device's group permission with all capabilities dropped. The retry added only
that device group and succeeded. Both attempts restored the unchanged backend.

## Retained settings and validation boundaries

The adapter-bound retained profile is
[`arc-5B3D045331-motion.json`](../config/arc-5B3D045331-motion.json).
Only both axes' `position_gains` change. Applying it must preserve the current
model, confidence, profiles, camera, calibration, saved zeros and angle limits.
The existing `ServoController.arm()` writes and read-verifies all three gains
before enabling torque. Keep a complete original configuration backup.

Earlier Arc motor-only trials already exercised these settings without faults.
Pan speed-ripple RMS fell from about 4.1 to 2.08 degrees/s, while fitted lag rose
from about 103 to 154 ms. Tilt ripple increased from about 1.35 to 1.80 degrees/s
and lag from 119 to 191 ms. Final holding errors were about 0.61/0.70 degrees.
Those tradeoffs were assessed in the live moving/stationary-ball comparison
below. These settings are not claimed to be optimal for every assembly or task.

Before any further recording, tell the user it is ready and wait for a new
explicit readiness reply. Give a START cue and movement/hold cues. Compare the
latest baseline, measured motion, centering and the user's physical observation.
If rejected, restore the complete saved configuration and original Start intent.
Do not label the jitter fixed based on motor-only measurements.

Latest footage, raw telemetry, plots and analysis scripts are retained under
`/home/ubuntu/work/turret-motor-20260909/`, with prefix
`cued-baseline-20260909`. The frontend-only FPS/mirror deployment is separate:
commit `4e71d45` restarted only Arc's frontend and refreshed the existing Firefox
window; Arc backend/viewer/model PIDs were verified unchanged.

## Deployment state

The candidate was applied to Arc from `40d3976`. Its backend was restarted;
P400/I0/D0 was read-verified by normal arming on both axes. SAM3.1 Mask / `ball`
resumed with advancing frames, the prior Start intent and no servo error. All
protected source/calibration/zero hashes and the expected full configuration
matched. Receipt and original baseline backup:
`/home/spring/.local/share/turret-demo/thor-gains-trial-40d3976-20260909T065820/deployment-receipt.json`.

## Accepted 60-second repeat

The user first reported that the deployed settings seemed better, then requested
a fresh 60-second recording. The first take completed (846 Arc / 453 Thor frames),
but the user asked to restart it. The repeat is the acceptance take and is stored
with prefix `thor-gains-cued-redo-20260909`; it followed advance notice and a fresh
START cue, with slow sweeps, faster sweeps and an extended final hold. The earlier
take remains preserved and is not silently combined with this one.

| Recording | Duration | Arc frames / missing masks | Thor frames / missing masks | Arc / Thor dropout events |
| --- | ---: | ---: | ---: | ---: |
| Original P1200/D1600 baseline | 40 s | 564 / 97 | 302 / 10 | 58 / 6 |
| Accepted P400/D0 repeat | 60 s | 844 / 35 | 452 / 29 | 29 / 24 |

A dropout event is a transition from a valid class mask to no valid class mask.
Both devices had **zero new servo read retries and zero servo faults** in the
repeat. The final 15 seconds retained 212 Arc frames (10 missing masks) and 113
Thor frames (7 missing masks). Mask loss is therefore not eliminated.

Arc's missing-mask fraction was 4.1% in the repeat versus 17.2% in the baseline.
The ball motion, camera viewpoint and durations differ, so this is descriptive
of the recordings, not a controlled model-accuracy or percentage-smoothness claim.
The model, confidence and frame scheduling were unchanged by the gain update.

The user's physical assessment is the acceptance result: clearly smoother,
with occasional jerks under very fast motion and more jerking on Thor in that
condition. Keep the current gains; do not add another filter/model/profile
change to chase this accepted result. No further recording is running.

All changes are persisted in Arc's existing config and read-verified by its
normal arm sequence. Both backends remain operational. The source, saved zeros,
geometry, model and limits remain those protected in the deployment receipt.
