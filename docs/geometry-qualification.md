# Spring Edge 2 pattern-free geometry qualification

2026-09-06, version 0.14.0. Camera USB identity `0c45:0261:UC684`, 1280x720,
commissioned pan ID 2 and tilt ID 1. This calibration is specific to this
camera/lens/mount. Do not install its JSON on another assembly.

## Geometry-only held-out test

Settled, encoder-measured images were collected within ±8° pan / ±6° tilt of the
starting pose. Eight non-reference poses trained an equidistant fisheye model
and camera-to-tilt rotation; six other poses were reserved for testing. Fitting
used reciprocal SIFT matches, spatial caps and a robust loss. Held-out samples
were not used to fit or select parameters. No checkerboard or detection model
outputs were used for calibration.

| Held-out feature prediction | Old linear mapping | Fisheye + joint geometry |
| --- | ---: | ---: |
| Median error | 4.235 px | 1.006 px |
| 90th percentile | 9.879 px | 2.004 px |

There were 6,483 held-out point matches across six poses, occupying 45–47 of 48
image grid cells per pose. Median error improved 4.2x and p90 error 4.9x. Every
held-out pose passed the fixed median/p90 gates. These are feature reprojection
errors under measured rotations, **not** physical centering error after one
motor command and not a guarantee over the full ±90° mechanical range.

Fitted intrinsics in pixels: fx=522.7413, fy=517.5973, cx=641.8349, cy=370.8235;
equidistant radial terms k1=-0.0099951, k2=-0.0092143. Full parameters, mount
rotation and per-pose metrics are in [the device JSON](geometry-spring-edge-2.json).
The runtime checks qualification, ray-domain monotonicity, camera identity,
resolution, motor identities and directions before using the model.

## Live pointing check

Two stationary local feature groups were successfully tested in both modes.
Single-corner attempts were aborted due to ambiguous descriptors; group-based
matching used RANSAC, minimum support, residual and spatial-coverage gates. The
third target lost sufficient matches, so that attempt stopped safely; there is
no claimed three-target pass. No matching thresholds were relaxed to count it.

| Target | Old first-command centering error | New first-command error | New after up to two feedback corrections |
| --- | ---: | ---: | ---: |
| 1 | 12.25 px | 15.53 px | 3.16 px |
| 2 | 11.75 px | 11.11 px | 1.42 px |

The new geometry predicted the observed feature motion to 0.72–1.51 px across
these four moves. However, encoder readback lagged the settled requested goals
by roughly 1° on some axes. The old oversized linear correction sometimes
incidentally offset that servo holding error, so first-command accuracy did
**not** consistently improve. Keep the runtime's existing settled holding-bias
estimation and visual feedback; geometry alone does not remove backlash,
compliance, stiction, motion latency or close-range parallax.
The requested home pose was the same between tests, but actual starting poses
varied with servo holding error; this was not an identical-start first-move benchmark.

The check tool applies two explicit feedback corrections when the first move
misses. This is not a measurement of closed-loop convergence time in the normal
detector/tracker and does not change the normal tracker's centering deadband.

## Deployment and checks

The calibrated runtime solves the two joint angles together against the actual
image-center ray. Instance association also uses calibrated reprojection. Servo
zero changes are accounted for through encoder-origin offsets. Motion limits,
fault checks, Stop/lease handling and model computation remain unchanged. There
is no OpenCV/numpy/scipy dependency in the service's per-frame geometry path.

Synthetic tests cover full-frame ray round trips, 300 random coupled pointing
cases, optical versus image center, reversed directions, zero changes/rollover,
invalid files and device mismatch, plus tracker and instance integration.
Calibration source images remain on the device, outside Git.

Version 0.14.0 was installed with the calibrated mapping enabled and exact
camera/motor binding verified. Original measured start: X=-24.17°, Y=1.49°;
final encoder-corrected return: X=-24.08°, Y=1.58°, then torque off. Saved zero
file hash and all config fields except the new `tracking.geometry_file` were
verified unchanged. Previous release/config remain available for rollback.
