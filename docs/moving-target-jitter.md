# Moving-target jitter: Arc and Thor, 2026-09-09

The user reported that Arc was less smooth than Thor in SAM3.1 Mask while
following a moving target. Live diagnosis and changes are on `user/demo-changes`.

## Measured cause

Both devices used the same shared controller, calibrated geometry, ±90° limits,
and unbounded servo profile settings. Arc used its existing P=1200/I=0/D=1600
gains and packed feedback; Thor had no explicit gain override in its config.
The live Arc controller matched the repository byte-for-byte. No servo faults
occurred during the coordinated 35-second `ball` pass.

| Observation | Arc | Thor |
| --- | ---: | ---: |
| Distinct processed frames sampled | 493 | 264 |
| Frames with the selected ball | 486 | 262 |
| Internal detection gaps | 7 | 1 |
| Length of each internal gap | 1 frame | 1 frame |
| Median worker cycle | 69.2 ms | 131.3 ms |

The original controller immediately replaced the target angle with the current
encoder position on **every** missing mask. The next detection commanded pursuit
again. Replaying the same Arc observations through the actual old/new controller
produced seven such braking commands before the change and zero afterward.
This establishes a software cause of stop/start motion, not proof that every
perceived vibration comes from it. Different views and models can still produce
different detections.

## Shared correction

`sam-shared-v2` tolerates a short detection gap by allowing the existing bounded
goal to finish for at most 200 ms after the last accepted result. It sends no
new target or predicted movement during the gap. Persistent absence holds the
measured pose once; repeated misses cannot reset the deadline. The control thread
also expires the deadline when inference stops delivering new frames.

Explicit Stop, manual/target/model/prompt changes and camera/inference faults
retain their immediate handling. No previous box/mask is exposed as a current
detection. Both runtime adapters use the same policy and time boundary, rather
than a device-specific missed-frame count. There is no change to confidence,
model precision, frame resolution, calibration, limits, gains or servo profiles.

## Arc filter qualification

The existing optional world-bearing filter was also evaluated on the same
recording. `min_cutoff_hz=0.5`, `speed_gain=40` reduced pan-command direction
reversals (both adjacent speeds exceeding 1°/s) from 34 to 20. The angular
difference from the unfiltered estimate was 0.213° median, 0.389° p90, and
0.775° maximum. These measure filtering/lag on recorded observations, not
physical tracking accuracy against labeled ground truth. The filter does not
change the model's masks or the displayed aiming centroid.

This profile is specific to the measured Arc assembly. It did not improve pan
reversal count on Thor, so Thor retains its existing filter configuration.
Use this addition to Arc's existing `tracking` config, preserving every other
field and its assembly-specific files:

```json
"bearing_filter": {"min_cutoff_hz": 0.5, "speed_gain": 40}
```

Filtering happens after calibrated camera motion is removed, in world-ray
coordinates. It adapts to target speed and resets on changed identities, large
jumps, long gaps and explicit control changes. Brief tolerated detection gaps
retain its state. This adds no model worker or resident-model cache.

## Verification and rollback

Regression coverage includes a missed mask without a brake or invented box,
reacquisition, sustained loss, expiry without another frame at both observed
inference rates, immediate fault handling, Stop during loss, and new-selection
isolation. Existing geometry, filter, recovery, API and shared-policy tests remain
in force. Recorded-input replay is separate from live acceptance.

The local diagnostic artifacts are in
`/home/ubuntu/work/turret-jitter-20260909`: baseline and coordinated recordings,
geometry/config snapshots, original installed modules, replay results and the
full validation log. Do not overwrite the immutable September 9 recovery archives.

To roll back, restore the backed-up application modules and configuration,
restart only the affected backend, then restore the operator's latest model,
prompts, target and Start intent. Removing only `tracking.bearing_filter` disables
the optional Arc filter while retaining the shared missed-detection fix.

## Deployed Arc result

Application commit `535593b5268839166638d141937f635d5fbb6694` was pushed to
`user/demo-changes`, fetched into `/home/spring/turret-demo`, and installed into
Arc's existing backend environment. Only `tracking.py` and `policy.py` changed
in the installed package. Both hashes match the pushed source. The only config
change is the bearing-filter entry above; gains, motor profiles, limits, model
settings, calibration and saved-zero files are unchanged.

Arc backup and receipt:
`/home/spring/.local/share/turret-demo/jitter-535593b-20260909T052650/`.
The first deployment attempt restored the original code after a target-restoration
request used the wrong API field. The corrected attempt restored SAM3.1 Mask,
`ball`, class following and the prior Start intent, then completed model warmup.
No calibration or motor-register change was involved in that recovery.

The complete CPU/JavaScript validation suite passed, with its two dependency-gated
skips. After adding the final filter-state regression, all 41 tracking tests also
passed. A one-minute live recording after deployment captured 846 distinct Arc
results, zero servo errors/retries, one uninterrupted backend PID, and median
worker time 69.15 ms. The ball was stationary near image center: both encoder
positions stayed constant despite intermittent detection loss. This qualifies
live startup, state restoration and stationary holding; **a repeated live moving-
target comparison and subjective smoothness confirmation remain pending**. The
before/after moving-target claims above are same-input controller replays.

Thor's running backend remains on `sam-shared-v1`, with its configuration and
filter unchanged. The shared source was fetched there and an application-only
image `spring-turret-demo:jitter-535593b` was built from the pinned recovery
runtime, but it has not been deployed. Its image ID is
`sha256:d1993d3607f613de621788c27b6530279cfb4fb3c6babd476e182f95fba5fef4`.
Deploying that image is a separate live-device action. Arc's filter should not be
copied to Thor: the recorded Thor pan reversal count did not improve with it.
