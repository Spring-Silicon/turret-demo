# Arc trial using Thor's measured motor gains — 2026-09-09

The earlier P800 trial failed live and was reverted. It retained D1600. This
candidate instead uses the motor gains actually read from the smooth Thor:
P400/I0/D0 on both axes. It is a pending live trial, not a qualified fix.

## Evidence from the latest cued test

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

## Candidate and acceptance

The adapter-bound candidate is
[`arc-5B3D045331-thor-gains-trial.json`](../config/arc-5B3D045331-thor-gains-trial.json).
Only both axes' `position_gains` change. Applying it must preserve the current
model, confidence, profiles, camera, calibration, saved zeros and angle limits.
The existing `ServoController.arm()` writes and read-verifies all three gains
before enabling torque. Keep a complete original configuration backup.

Earlier Arc motor-only trials already exercised these settings without faults.
Pan speed-ripple RMS fell from about 4.1 to 2.08 degrees/s, while fitted lag rose
from about 103 to 154 ms. Tilt ripple increased from about 1.35 to 1.80 degrees/s
and lag from 119 to 191 ms. Final holding errors were about 0.61/0.70 degrees.
Those tradeoffs require an actual moving/stationary-ball comparison; copying
Thor gains does not prove they are optimal for Arc's assembly.

Before recording the candidate, tell the user it is ready and wait for a new
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

Candidate recording has not started. A new explicit readiness reply is required;
plan 60 seconds with a longer final stationary segment. Physical smoothness and
centering remain unvalidated for this deployment.
