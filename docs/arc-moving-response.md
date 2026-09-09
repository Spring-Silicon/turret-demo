# Arc moving-target motor response, 2026-09-09

Arc's physical pan motion was reported as jerky while Thor was smooth. The
missed-detection grace period and optional bearing filter did not resolve the
live complaint and were reverted in `001a9de`. The next experiment isolated motor
response from SAM by commanding known, smooth angle trajectories.

Current retained gains are **P400/I0/D0 on both axes**, accepted after the later
[live comparison with Thor's measured settings](arc-thor-gain-trial.md). The
P800 rollback and synthetic experiments below are historical.

## Live result: P800 did not resolve the complaint

The pan-P800 candidate was deployed from `97ff9de`, the backend was restarted,
and the user repeated the moving-ball comparison. The user reported that Arc's
physical turret was still extremely jerky. The synthetic motor-response
improvement below did not resolve the real complaint. Do not treat this candidate
as a qualified fix or reapply it from the historical commit.

After that failed trial, pan P was restored to 1200; both axes used P1200/I0/D1600 and zero
velocity/acceleration profiles. No production source, model, filter, camera,
calibration, saved-zero or limit changes were made. Normal SAM3.1 Mask / ball
tracking and the previous Start intent were restored. The active P800 profile
file has been removed from this branch to prevent accidental redeployment.

The live comparison captured 60 seconds: 853 distinct Arc results and 455 Thor
results, with substantial physical pan movement on both. Arc had 42 internal
mask gaps (including one long absence) and Thor had 21. No servo faults occurred.
These scenes and views are not identical model inputs; these counts do not prove
quantization causes the observed jerkiness.

A subsequent read inside Thor's running container verified that `tracking.py`,
`servo.py`, `geometry.py`, `pose_history.py` and `policy.py` have exactly the same
hashes as Arc's restored application. Thor has no configured gain override;
later actual register readback confirmed P400/I0/D0 on both axes, as documented
in the accepted comparison linked above.

## Motor-only experiment

The normal Arc backend was stopped for exclusive serial access. Its existing
ServoController performed all model checks, bounds checks, fresh feedback reads,
gain readback and fault handling. An instrumented subclass copied existing
feedback samples and timestamped goal writes; it did not substitute a different
motor-command implementation. No video/inference ran during the sweeps.

Each 10-second trial commanded pan +/-8 degrees with a 2-second sine period and
tilt +/-4 degrees with a 2.5-second period, around the current pose near -25/-25
degrees. Goals were issued every 68 ms. Maximum ideal speeds were about 25 and
10 degrees/s. Each trial returned to the same start, with a 0.7-second hold.
Repeated controls bracketed the candidates. All 13 trials completed with zero
servo faults and zero read retries. All original RAM settings and operator
controls were restored after each batch; full configuration, zeros and geometry
hashes were verified unchanged.

Feedback samples were typically 23.3 ms apart, with extra fresh reads at commands.
Analysis uses seconds 1 through 9, interpolated onto a common 40 ms grid. A
least-squares sinusoid at the commanded frequency estimates amplitude and delay.
"Speed ripple" is RMS of the finite-difference velocity remaining after removing
that fitted sinusoid. It measures sampled speed unevenness; it is not a peak
jerk/acceleration bound or an independent visual measurement. Holding error is
measured after the 0.7-second final hold, not infinite-time static accuracy.

| Pan setting P/D; profile V/A | Pan speed ripple, deg/s RMS | Fitted pan delay, ms | Final pan error magnitude, deg |
| --- | ---: | ---: | ---: |
| 1200/1600; 0/0, repeated controls | 4.07–4.30 | 103.4–103.8 | 0.00 |
| **800/1600; 0/0** | **2.56** | **120.7** | **0.09** |
| 1200/800; 0/0 | 3.88 | 100.9 | 0.00 |
| 1200/1600; 100/20 | 2.37 | 134.9 | 0.09 |
| 1200/1600; 100/10 | 2.05 | 157.8 | 0.09 |
| 600/1600; 0/0 | 2.33 | 142.0 | 0.53 |
| 400/1600; 0/0 | 2.16 | 179.7 | 0.70 |
| 400/0; 0/0 | 2.08 | 153.6 | 0.61 |
| 600/0; 0/0 | 2.14 | 123.1 | 0.17 |

The rejected P800 candidate reduced measured pan speed ripple by 38–40% against the
first batch's bracketing controls, with about 17 ms additional fitted delay.
Its continuous-path pan RMS error increased from 1.82 to 2.12 degrees, primarily
from that delay. The user subsequently rejected its real moving-target behavior. P600/D0
was somewhat smoother, but removes the damping previously needed for step
overshoot and has greater holding error; it was not selected.

Tilt gains remain unchanged: lowering P did not improve its speed ripple and
increased both lag and holding error. Applying one identical gain setting to
both axes is not supported by this experiment.

One control used the original gains with 132 ms goal intervals, approximating
Thor's measured moving-ball cadence. Pan speed ripple rose to 10.59 deg/s and
tilt to 5.08 deg/s. Slowing Arc to Thor's inference cadence is therefore not
supported as a motor-smoothing fix. This test does not reproduce Thor's physical
assembly or establish Thor's actual gains.

## Artifacts and deployment

Local artifacts: `/home/ubuntu/work/turret-motor-20260909/`, including the exact
probe scripts, raw timestamped motion data, and analysis. Arc retained complete
raw data and restored status under:

- `/home/spring/.local/share/turret-demo/moving-response-20260909T055628/`
- `/home/spring/.local/share/turret-demo/moving-response-20260909T055941/`

P800 deployment and rollback receipts on Arc:

- `/home/spring/.local/share/turret-demo/pan-p800-97ff9de-20260909T060432/deployment-receipt.json`
- `/home/spring/.local/share/turret-demo/restore-pan-p1200-97ff9de-20260909T060902/deployment-receipt.json`

The rollback changed only `servo.axes.x.position_gains.p` from 800 to 1200,
restarted the backend, and verified advancing Mask frames, ball selection,
armed state and no servo error. The original recovery-archive motor settings
remain the active baseline. Thor has not been interrupted or changed.
