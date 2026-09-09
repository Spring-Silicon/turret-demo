# Non-model overhead qualification — 2026-09-08

> Correction: the initial tracking preprocessing change reused a graph-output
> tensor across frames. The XPU multi-prompt image cache uses object identity,
> so Arc could reuse stale image features. The subsequent fix returns a fresh
> owned tensor for each frame and rejects missing per-frame image passes.
> Earlier whole-tracker FPS/model timings are not valid speedup evidence.
> Pixel-equality checks alone did not catch this identity/ownership regression.

Scope: Arc `spring-edge-turret` and Thor `agxthor-5`, SAM3.1 tracking and mask
profiles. No model weights, quantization, resolutions, confidence thresholds,
temporal policies, servo calibration, or control gains changed.

## Changes

- Tracking: retain exact PIL bilinear resize, move the original float32
  normalization to a compiled GPU LUT, and transfer uint8 instead of float32.
  The LUT preserves rounding/operation order. First use checks exact equality
  against the original torchvision CPU input.
- Tracking: prepare CPU pixels while the previous frame is inferred. A prepared
  result is consumed only when both its token and JPEG bytes match the requested
  frame. A miss takes the original synchronous preparation path; no stale frame
  is substituted and the model still processes the whole selected frame.
- Mask: reuse pinned output buffers and enqueue all readbacks before one stream
  synchronization. All masks, scores, IDs, counts, overflow flags, and boxes are
  retained. Returned arrays are borrowed only until the next worker call.
- Thor mask: use eight CPU threads for image preparation. Same-image tests at
  1/2/4/8 threads produced identical resized uint8 pixels. Arc retains four.

## Exact-input measurements

Median of 30 measured iterations after five warmups, on the actual GPUs. Video
preprocessing tests also covered RGB, grayscale, and CMYK JPEG conversion.

| Stage | Arc before → after | Thor before → after |
| --- | --- | --- |
| Tracking preparation, synchronous | 13.97 → 10.90 ms | 17.06 → 11.93 ms |
| Tracking critical path when CPU pixels are already prepared | 13.97 → 0.51 ms | 17.06 → 0.15 ms |
| Mask tensor readback, ten full-size masks | 5.34 → 2.40 ms | 0.27 → 0.14 ms |

Prepared-path figures exclude CPU work overlapped with the preceding frame.
They are not whole-model latency or an FPS claim. Readback results were
bit-for-bit equal, including changed contents and changed prompt-batch shapes.
An additional Thor readback-plus-centroid/resize test also favored pinned
readback; the optimization was not judged solely on transfer timing.

Thor's CPU-only image preparation test fell from approximately 19–20 ms with
four threads to 12.8 ms with eight. In the live mask worker, median preparation
fell from 24.7 to 20.8 ms; the prepared-frame hit rate was 37% in that sample.

Arc steady tracking showed 0.75 ms median preparation, with a 72% prepared-frame
hit rate. Thor steady tracking showed 0.62 ms mean preparation on prepared
frames (30% hit rate), and 21.24 ms across all frames versus 33.45 ms before.
Its median remained 30.17 ms because most frames used the synchronous fallback.
Live mask samples on both machines had changing object counts, and
tracking samples had different prompts/session histories. Their overall FPS
must not be presented as controlled before/after speedups. Initial tracking
samples also include graph warmup; use steady samples separately.

## Deployment and rollback

Arc: only `tracking_preprocess.py`, `output_transfer.py`,
`sam31_tracking_worker.py`, `sam31_tracking_native_worker.py`, and
`sam31_mask_worker.py` were installed in the existing package/source mirror.
Original files are retained under
`/home/spring/.local/share/turret-demo/overhead-20260908/installed-before` and
`source-before`. The pinned Israel model manifest remains unchanged.

Thor: image `spring-turret-demo:0.16.12-overhead-agxthor-5`, derived from the
actual running 0.16.8 image plus its inspected live source, not the unrelated
pending 0.16.10 launcher. A separate `run-container-overhead.sh` and systemd
`30-overhead.conf` preserve that existing launcher. The original image and
package snapshot remain available for rollback. No device/network/container
privileges were expanded.

Local benchmark evidence and host snapshots are in
`../overhead-20260908.hEv7ae` relative to the repository. Full repository
validation passed, with the two existing dependency-gated tests skipped;
accelerator equality checks ran separately on both actual GPUs.

Initial-pass checks: both services running, both tracking workers advancing frames,
compiled graph flags and exact-input validation true, JPEG/PNG payloads decoded
successfully, and no servo errors. Both restored to SAM3.1 Tracking / `arm`,
motors armed and holding. Clicked instance IDs cannot survive a new temporal
session, so automatic target selection was cleared; the user must click the
desired object again. Arc-to-Thor viewer route remains direct Ethernet
(`192.168.249.2` via `enp7s0`).

## Follow-up: frozen Arc tracking masks and GPU mask postprocessing

The frozen-mask cause was an input-ownership regression introduced by the first
tracking preprocessing optimization above. `VideoPreprocessor` returned the same
graph-owned tensor for successive camera frames. Arc's multi-prompt
`DetectionRegion` intentionally caches image features by Python object identity
to share one immutable frame across prompts. It therefore mistook new pixels
for the previous frame. The preprocessor now returns an owned clone per frame,
and the Arc native worker checks that the image-and-detection call count advances
on every frame. The model bundle and temporal algorithm are unchanged.

The XPU ownership regression test checks equal prepared pixels, distinct tensor
identity/storage, and immutable previous-frame values after the next call. A
40-frame live tracking sample with 1–3 objects verified one image pass per frame:
107.93 ms median model time and 8.59 observed FPS. This supersedes earlier
misleading tracking throughput that skipped image encoding.

For non-tracking masks, a separate compiled/replayed **integer postprocessing**
stage computes original-mask moments and the exact display label image on GPU.
Only that uint8 label image and small original metadata are read back, rather
than all full-resolution masks. Resizing uses PIL-derived nearest-neighbor
indices; overlap ranking, centroids, palette and PNG encoding match the prior
implementation. The model graph, weights, precision, resolution, confidence and
retained detections are unchanged. Exact in-flight image preparation can now be
awaited for at most 25 ms instead of doing duplicate work; that wait is included
in both preprocessing and whole-worker timings.

Matched four-mask, 1280×720 postprocessing benchmarks (30 iterations after five
warmups, alternating order; includes readback, geometry, composition and PNG):

| GPU | Previous stage | GPU stage | Reduction |
| --- | ---: | ---: | ---: |
| Arc | 32.60 ms | 17.27 ms | 47% |
| Thor | 24.76 ms | 13.79 ms | 44% |

These are matched synthetic-mask **stage timings, not whole-demo FPS gains**.
Both actual GPUs passed exact PNG-byte, centroid, metadata and rank checks,
including overlapping/tied/empty masks, two classes and compiled transitions
from zero to capacity. Each deployed worker also checks its first actual model
output against the original CPU implementation.

Live Thor postprocessing (excluding PNG) fell from 15.26 to 1.76 ms, with changing
object counts. CPU preparation remains substantial: new whole-worker median was
132.04 ms versus 131.24 ms before; observed FPS was 7.54 versus 7.26. These live
samples have different preparation phases/counts and are not a controlled
end-to-end speedup. New Arc postprocessing was 2.70 ms, but its new sample had no
objects versus 2–5 before, so its live delta is not a clean comparison either.
A pixel-identical PIL decoder alternative was slower on both hosts and was
rejected.

Follow-up deployment: Arc installed `mask_postprocess.py`, `sam31_mask_worker.py`,
`prefetch.py`, `tracking_preprocess.py`, and `sam31_tracking_native_worker.py` in
the existing package and source mirror. Snapshots are under
`/home/spring/.local/share/turret-demo/mask-overhead-20260908`. Thor runs image
`spring-turret-demo:0.16.13-mask-overhead-agxthor-5`, derived from the previous
0.16.12 image with the mask helper/worker, prefetch and tracking ownership fix.
Its previous launcher is retained at
`/var/tmp/mask-overhead-20260908.2HPK8f/run-container-before.sh`; the separate
overhead launcher/drop-in still leaves the unrelated original launcher intact.
Do not roll back the Arc ownership fix to the initial 0.16.12-era behavior.

Both demos were restored to **SAM3.1 mask / `arm`**, armed and holding with no
selected target. Tracking is available in the dropdown with the Arc fix. Servo
calibration, control gains, device bindings and direct Ethernet are preserved.
Follow-up evidence lives in `../mask-overhead-20260908.szrg4I` relative to this
repository; full local validation passed and accelerator tests ran separately
on each actual GPU.

## 2026-09-09: Arc mask overhead follow-up

Only the non-tracking Mask path is tuned; model weights, its compiled graph,
precision, resolution, thresholds, retained masks, geometry and control policy
are unchanged. No model warm-reuse work is included.

- Integer mask moments use int32 where a mathematical bound proves no overflow.
  A full 1008-square mask has area 1,016,064 and first moments 511,588,224, all
  exactly representable. Larger diagnostic shapes retain int64 accumulation;
  public moment outputs remain int64. Display area is bounded by 8192 squared.
- PNG encoding borrows the already-indexed uint8 label buffer directly instead
  of copying/converting a full grayscale image. Palette, alpha, compression,
  dimensions and output bytes are identical, including the existing size cap.
- Mask workers check for newly captured frames every 1 ms instead of 5 ms while
  awaiting inference. Preparation still requires an exact token AND JPEG match;
  the detector never selects an older frame merely because it is prepared.

Alternating same-input stage measurements on the Arc GPU/CPU:

| Stage | Before | After |
| --- | ---: | ---: |
| GPU mask composition plus readback | 4.12 ms | 3.63 ms |
| PNG creation/encoding | 1.49 ms | 1.09 ms |

These are stage benchmarks while the demo was running, not controlled whole-model
FPS claims. Changing CPU thread counts did not materially improve preparation,
and flattening the moment reductions was slower; neither change was retained.

Staged GPU tests checked exact moments, ranks, metadata and PNG bytes, changed
frames, one/two classes, empty-to-capacity transitions, filled masks, and int64
overflow fallback. The complete CPU/JavaScript suite passed; the new PNG tests
also cover noncontiguous buffers, palette bit depths, payload limits and owned
encoded results. Evidence and original installed files are retained locally in
`../arc-mask-overhead-20260909.Undycg` and on Arc in
`/home/spring/arc-mask-overhead-20260909`. Only `detection.py` and
`mask_postprocess.py` are deployed to Arc; Thor is unchanged.

Live `person` Mask samples, 276 frames before and 496 after:

| Measurement | Before | After |
| --- | ---: | ---: |
| Median worker overhead (worker total minus model) | 6.16 ms | 5.16 ms |
| Median pipeline overhead (cycle minus model) | 8.00 ms | 6.62 ms |
| Median model wall time | 63.72 ms | 63.24 ms |
| Observed processed FPS | 13.77 | 14.16 |

The camera scene changed (mean 3.21 vs 2.11 retained objects), so these are live
observations, not a controlled full-model speedup. Restricting to three-object
frames gave 6.05 vs 5.14 ms median worker overhead; four-object frames gave
6.77 vs 5.90 ms. Neither grouping fixes mask geometry or preparation phase.
After-change p90 worker overhead was 8.44 ms, so this does not eliminate every
8–10 ms overhead spike. The matched stage results above are the isolated
before/after evidence. Exact model-output checks remained enabled in the live
worker. The deployment drained inference before reload, preserved calibration,
prompts, target and Start intent, and produced no new Intel GPU-driver errors.
