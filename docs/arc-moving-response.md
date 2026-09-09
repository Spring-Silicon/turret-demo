# Arc moving-target motor response, 2026-09-09

Arc's physical pan motion was reported as jerky while Thor was smooth. The
missed-detection grace period and optional bearing filter did not resolve the
live complaint and were reverted in `001a9de`. The next experiment isolated motor
response from SAM by commanding known, smooth angle trajectories.

## Selected live candidate

`config/arc-5B3D045331-motion.json` records an assembly-specific gain profile;
it is not a complete backend config and must not replace one. Apply only its
`position_gains` to the matching axes of Arc adapter `5B3D045331`, after checking
adapter serial, model and motor IDs. Pan P changes from 1200 to 800. Pan I=0,
D=1600 and tilt P=1200/I=0/D=1600 are retained. The existing backend writes and
read-verifies the gains on Start.

There is no application-code, detection, model, camera, deadband, motor-profile,
calibration, zero, limit or Thor deployment change. Both velocity/acceleration
profiles remain zero. Stop, fault handling and target-selection policy remain
`sam-shared-v1`. The previous experimental filter remains disabled.

This is a measured motor-response improvement and a candidate for the live
moving-target complaint. Successful startup or a smooth synthetic sweep alone
does not establish that all live jerkiness is fixed. Operator acceptance remains
pending until the same moving-ball comparison is repeated.

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

The selected profile reduces measured pan speed ripple by 38–40% against the
first batch's bracketing controls, with about 17 ms additional fitted delay.
Its continuous-path pan RMS error increased from 1.82 to 2.12 degrees, primarily
from that delay. This tradeoff must be checked on a real moving target. P600/D0
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

Deploy by preserving the complete current config, changing only the pan P
register setting to 800, restarting Arc's backend, and restoring its latest
model/prompts/target/Start intent. Confirm profile identity and gain readback,
advancing processed frames and no servo faults. Rollback restores the backed-up
configuration and restarts the backend; no source-code rollback is needed.

The earlier full-image recovery archives deliberately retain their original
configuration. Reproducing this candidate requires applying the profile above
after restoring that archive. Thor remains unchanged.
