# SAM 3.1 Tracking (Arc and Thor)

An optional third model profile, `sam3.1-tracking`, runs Meta's full SAM 3.1
Object Multiplex network: image/text encoding, detection and masks, memory
attention, mask propagation, memory encoding, association and reconditioning.
It is **dense BF16 with compiled tensor-region graph replay on XPU or CUDA**, not
the detector-only compiled graph. The
other profiles are SAM 3.1 boxes and SAM 3.1 Mask; YOLO is removed. Their shared
selection/lifecycle policy is documented in [the backend audit](sam-policy-audit.md).
The optional
Israel native672 build below replaces this dense implementation on explicitly
configured XPU devices only; the dense implementation details in this document
do not describe its precision or resolution.

Enable with `inference.sam31_tracking_bundle` pointing to a pinned SAM3.1 source
bundle and the existing full `inference.checkpoint`. On Thor, set the bundle to
`/opt/sam3` in the container and retain `inference.device_type: cuda`. CUDA uses
the unmodified upstream device paths; only XPU needs import-time adaptation.
The worker uses the same Torch environment;
no system drivers, weights, camera configuration or motor calibration change.

## Live behavior

- One shared model, separate temporal state for each text prompt (up to eight).
  Arc encodes each image once and reuses its features for subsequent prompts;
  prompt-specific detection and temporal state remain separate. Thor's baseline
  still runs an image pass per prompt. Maximum 16 simultaneous tracks per prompt.
- Boxes come from tracked masks and retain SAM's object IDs. They are not
  reassigned by the app's geometric box associator.
- The aiming point is the **mask centroid**: the mean center of all foreground
  pixels in the original full-resolution binary mask. It is computed before
  display overlap composition/downsampling, without another model pass. Class
  acquisition and both calibrated axis corrections use this point. Box-only
  profiles retain box-center aiming; an empty mask has no aiming point.
- The local viewer shows translucent instance masks for this profile. Each
  prompt has one color family, with a stable hue/shade per native tracked ID.
  Hovering reveals the instance label; clicking its box still selects that
  identity. Other model profiles retain their usual bounding boxes.
- Masks reuse the tracker's existing outputs, without another inference pass.
  An indexed PNG travels with its matching camera frame; stale overlays are
  discarded on frame, prompt or model changes. Normally it retains the camera
  resolution; only pathological payloads over 384 KiB are downsampled for
  display. This never changes inference masks, boxes, counts or control.
- Selecting a class acquires its nearest visible instance once. Clicking a box
  selects that identity. During occlusion, motion holds instead of switching to
  a neighboring object. Selecting the class again or clicking another box
  explicitly reacquires; a retired ID is not silently replaced.
- Prompt submission (even unchanged text), model/camera changes, and camera
  gaps reset temporal state. New IDs are not reused within a worker.
- No automatic motor start. Existing angle, geometry and fault protections
  remain in force. The 750 ms detection/click age cutoff is removed; delayed
  completed frames are processed once, not repeatedly. Frames from before Start
  or a target/prompt change remain excluded, and camera-pose pairing is unchanged.

## Causal adapter and memory

The offline predictor expects a complete video. The live adapter instead calls
its one-frame detection/propagation/update path with no future-frame prefetch.
Absolute tracker time advances independently of a single current image slot.
Pointer-time normalization treats the stream as a long video, not a sequence
of progressively longer short clips. The standard tracking thresholds and
masklet confirmation are retained; unconfirmed/suppressed masks are hidden.
The live suppression adapter forwards the GPU hotstart suppression mask into
the CPU hidden-ID table consumed by output postprocessing. The pinned planner
previously discarded that mask, leaving unmatched tracks visible with their
original detection scores. It preserves the model thresholds and ID order.

If object retirement leaves a multiplex state with IDs but no conditioning
frames, the shared session retires those orphaned IDs before propagation. This
avoids the upstream `No points are provided; please add points first` exception;
healthy tracker memories and monotonic IDs are preserved.

The adapter retains 32 recent frames, the last four conditioning frames, and
the first conditioning frame if requested by the model. That covers the stock
16 object pointers and seven mask memories without discarding inputs needed by
future forward-only inference. Reverse tracking and editing historical masks
are not exposed. Source-device adaptation is confined to SAM modules imported
by the dedicated worker; no source files or global Torch APIs are rewritten.

`tracking_frame`, `memory_frames`, `active_instance_ids`,
`propagated_instance_ids`, and truthful compilation flags are exposed in the
detection status. This is an experimental quality profile: test actual
occlusions/crossings in your scene rather than assuming better accuracy from
stable IDs alone.

On Thor, the image encoder, detection encoder/decoder/masks, memory
encoding/attention and tracking masks use `torch.compile(backend="inductor")`
with BF16 inference and `emulate_precision_casts=True`. No quantization or
TensorRT engine is used. Text encoding remains cached `aot_eager`. This common
CUDA tracker serves both the Tracking selection and the Thor side of the shared
v18 selection; Arc's native v18 implementation is not changed.
See [Thor Inductor qualification and deployment](thor-unquantized-inductor.md)
for timing boundaries, regression evidence and rollback details.

The XPU regional baseline keeps its Inductor image encoder and `aot_eager`
heads. The legacy per-stage layout also remains unchanged. The
compiled components execute inside larger explicit CUDA or SYCL replay regions:

| Region | Thor | Arc |
| --- | --- | --- |
| Image encoder, geometry, detection encoder/decoder/heads | One replay | One replay; image features shared across prompts |
| Subsequent text prompt on the same image | Full image/detection replay | Detection-only replay |
| Memory attention and propagation masks | Two existing stage replays | One replay, including demux, resize and object pointers |
| Memory encoding/update | Existing encoder-stage replay | Whole memory update replay |

Text encoding is cached per prompt. The normal tracked frame therefore uses
four main replay launches on Thor and three on Arc, per tracker-state bucket;
initialization/reconditioning and additional prompts can add calls. Association,
session state, CPU mask NMS and conditioning-memory selection remain eager.
This is **not one graph of the entire stateful tracker**, nor a claim that all
operators were fused into one kernel. FA3/perflib remain disabled. No weights,
precision, resolution, memory policy or object limits were reduced.

Thor keeps at most two exact-shape graphs per region; Arc keeps four. New shapes compile/capture on
demand; the first frames and new object/memory shapes can pause during warmup.
Cached kernels survive process restarts, but graph buffers must be recaptured.
Inputs own their GPU storage and every returned tensor is cloned, so later
replay cannot overwrite cached image features or historical mask memories.
Aliased output tensors are cloned only once. NestedTensor storage is explicitly
unpacked at regional boundaries, never hidden from the ownership machinery.
SAM's lazy coordinate caches are primed before tracing, and mutable Python
input containers are rebuilt before each compiled call.

`torch_compile` and the device's graph flag become true only after actual
successful replay. `compilation_scope: tensor-regions` and `graph_stages` expose
per-stage compiler backends, call/capture counts, bounded cached shapes and
observed replay error. Capture rejects nonfinite outputs and tensor-scale RMS
error over `0.001 + 2% * reference RMS` for composed regions (1% for individual
stages); integer/boolean metadata and large
attention-mask sentinels must agree. Minor numerical/mask differences are
accepted, not bitwise equivalence or a broad tracking-accuracy guarantee.

Arc's memory adapter receives current GPU mux/demux matrices on every replay.
It runs upstream CPU memory selection, then defers the single attention call
until the adjacent mask head. A strict identity guard prevents the temporary
placeholder from being consumed as a prediction. A pinned, checked AST adapter
replaces host-list conditioning assignments with a fixed-shape GPU Boolean
mask; changed conditioning membership does not recapture the graph. The native
interactive/reconditioning path remains available. Source contract changes
fail rather than silently falling back or capturing stale Python state.

September 7 recorded-sequence qualification (40 frames, one `table` prompt):
tracking-stage wall time was 323 to 222 ms on Arc and 374 to 230 ms on Thor,
excluding preprocessing and including temporal Python work. Thor retained all
IDs with at most about three pixels of box-coordinate difference; Arc had
brief weak-detection differences while the main track remained stable. Fully
Inductor-fused tracking was rejected for substantial weak-track drift. These
are workload-specific timings, not pure GPU latency or a ground-truth MOT score.

XPU Flash Attention is nondeterministic: independent runs can differ at mask
boundaries and detection/association thresholds. Exact repeated-output parity
is not claimed. State-retention tests verify that pruning preserves the exact
conditioning outputs and recent memories consumed by the next forward pass.

## Regional replay qualification (September 7)

Fresh-process comparisons against the eight-stage implementation, same recorded
20-image sequence repeated for 40 frames; timings are medians of frames 20–39.
Both use full dense BF16 temporal tracking, not the faster detector-only model.

| Device / prompts | Previous tracking time | Regional tracking time | Previous whole worker | Regional whole worker |
| --- | ---: | ---: | ---: | ---: |
| Thor / table | 230.41 ms | 229.09 ms | 264.13 ms | 262.80 ms |
| Arc / table (initial regional qualification) | 221.90 ms | 221.68 ms | 259.11 ms | 259.52 ms |
| Arc / table + floor (final regional implementation) | 442.17 ms | 307.51 ms | 478.47 ms | 345.45 ms |

Combining launches alone gave no meaningful single-prompt speedup. Sharing the
image encoder gives 1.44x tracking throughput for the tested two-prompt workload
(1.39x whole-worker throughput). Arc encoded 40 images rather than 80; each
prompt still received its own detection pass and independent temporal state.
Peak Torch-allocated GPU memory was 7.07 GB for the two-prompt regional test.

Thor's 40-frame IDs, masks, boxes and scores matched exactly. Arc's final
two-prompt run retained all active/visible IDs, unchanged scores, at most one
vertical pixel of box difference, and minimum mask IoU 0.9956. The earlier
single-prompt Arc run retained all active IDs but had two brief visibility
differences on a weak secondary track; minimum mask IoU was 0.9876. One weak
baseline mask included stray ceiling pixels absent in the regional mask,
causing a large bounding-box corner difference despite high mask overlap.
No sustained main-track drift was observed. This short recorded-scene check
is not a ground-truth tracking-accuracy qualification.

New shape capture still causes warmup pauses. Regional replay is enabled by
`install_tracking_graphs(..., layout="regions")`; `layout="stages"` remains
available for controlled baseline comparisons and rollback. The web frontend,
camera/servo configuration, existing YOLO26x, and non-tracking SAM are unchanged.

## Arc football comparison (September 9)

Arc's live Tracking profile returned to **dense BF16 at 1008×1008** after the
native672 build missed the football and retained masks on unrelated objects.
Remove only `inference.sam31_tracking_native_bundle` from the device config and
restart the backend; restore the selected Tracking model and `football` prompt.
The ordinary Mask profile keeps its separate build. Motor settings are unchanged.

A controlled replay used 256 frames from earlier operator-cued Arc footage,
with the original capture timestamps, followed by 64 copies of a scene without
the ball. Both motors were stopped. Of these frames, 214 had a ball detection
from the earlier Mask run, which serves as a reference rather than ground truth.

| Tracking recipe | Aim inside reference ball box | Empty-scene frames with masks | Median tracking time |
| --- | ---: | ---: | ---: |
| native672, previous adapter | 104 / 214 | 25 / 64 | 88 ms |
| dense1008, previous adapter | 213 / 214 | 0 / 64 | 222 ms |
| native672, suppression and orphan-state fixes | 21 / 214 | 0 / 64 | 86 ms |
| dense1008, suppression and orphan-state fixes | 213 / 214 | 0 / 64 | 222 ms |

Restoring suppression removes native's stale masks but also exposes its frequent
failure to confirm the actual ball. Dense retains its ball aiming point with the
fix. Each replay completed 320 frames without a worker crash. Timings exclude
initial graph warmup and use recorded frames 64–255; these are tracking-stage
wall times, not camera-to-display latency. Tiny stray pixels sometimes enlarge
dense bounding boxes while leaving the mask centroid on the ball, so box IoU
alone is a misleading aiming-quality metric here.

The comparison does not isolate quantization, resolution, or temporal-memory
reduction, and it covers one clip/prompt. Physical moving-target confirmation
remains pending. Structured results and artifact hashes are in
[the qualification record](arc-football-tracking-qualification.json).

## Israel full1008 hillclimb candidate (September 9)

The new optional bundle selects **native v15** from Israel's
`tracking_implementation_20260909` work, not the earlier native672 build below.
Use `inference.sam31_tracking_native_bundle` pointing to the private
`israel-tracking-full1008-v15` bundle. Its manifest SHA256 is
`368a39b0abe03f0cf288f3162de40bdb1f5fdf4fa13555e66a5ed683a2c8fd39`.
The worker checks every bundled source/library and the existing full checkpoint
before importing model code. Its source tree is private to this bundle; it does
not overwrite `/home/spring/sam3_1` or depend on Israel staying online.

This recipe retains full 1008×1008 input, native W4A4 image projections,
W8A8 heads, uniform 50% temporal spatial K/V retention with all pointer tokens,
and 75% MLP token retention in blocks 8–31. Image attention remains the original
Torch implementation. Native providers and Inductor tensor regions use SYCL
graph replay. The existing bounded capture/replay cache policy, independent
per-prompt sessions, hotstart suppression publication, orphan-state cleanup,
retirement, mask-centroid output and motor/target policies stay in the demo.
No changes are made to Thor or the Box/Mask profiles.

Israel's audited v15 eager run covers 545 frames in 15 person scenes, with
81.7749% mean mask IoU and 72.5442% mean box IoU. The subsequently completed
compiled 545-frame audit scores 82.0353% masks / 72.9031% boxes with zero matched
adjacent mask-ID switches; this exceeds the compressed QDQ reference's
81.6732% / 70.4823%, not necessarily dense BF16. That audit is
`compiled_confirmation545_assembled_gpu1_v15_v2/report.json`, SHA256
`7910a0b662a2e5160adbb9b825a47aaea5b7bf61ddfb8c33d9672c0ca475a56f`.
The bundle's embedded qualification note records the earlier partial443 state.
Its audited full-engine ordinary/reconditioning medians were 138.74/149.33 ms
for crossing and 133.94/142.06 ms for lucia. These exclude host work/transfers;
they are not live camera FPS, dense-equivalence results, or an 80% roofline claim.
Newer v16/v17 components were still under qualification when v15 was selected.

### Arc qualification: activation deferred

The actual proposed worker completed the same 320-frame football/empty-scene
replay used above, with the current shared suppression and lifetime policies.
Its median model/whole-worker latency was **136 / 153.245 ms**, versus
**222 / 238.845 ms** for the earlier dense run (frames 64–255, excluding cold
compilation). It aimed inside the reference ball on **168 / 214** positive
frames, versus **213 / 214** for dense. Both had zero masks on all 64 appended
empty-scene frames. The candidate completed without a worker crash, and every
frame encoded its new image rather than reusing stale features.

This is a substantial ball-coverage loss, despite the improved person-cohort
scores versus compressed QDQ. It is one recorded clip with earlier detector
boxes as the reference, not general ground-truth accuracy. The candidate is
staged but **not activated**; the original Arc model/config and `person`
target/running intent were restored pending acceptance of that tradeoff.
Thor was not changed. The transient test used a 10 GiB cgroup memory cap and
peaked at about 8 GiB; this does not guarantee every future prompt fits.

Local artifacts on Arc are in
`/home/spring/.local/share/turret-demo/israel-v15-validation`:

- `report.json`: SHA256 `497c016bbbc12b01cdf1805f67bc804c129abc29c7189fb0001720fe4385426f`
- `results.jsonl`: SHA256 `05b1799a50f3ea668bd881bd264f60ceed951a6fceb6723af4739f2382f487a2`
- `worker.log`: SHA256 `cf743b06dc64f4f0b05820094129dbcd96bcdc239b1ca0731ba996a53440fe66`

Set the bundle key and restart Arc's backend to activate; a worker restart resets
temporal identities. Preserve the original config/checkpoint/dense source bundle
and remove only this key to roll back. Each native bundle has a separate compiler
cache. Cold compilation must run without a second GPU model on memory-limited
Arc devices. Local deployment/recorded-frame qualification is separate from
Israel's source results and must be checked before activation.

## Israel native672 tracking opt-in

This build is retained for explicit experiments. It is no longer Arc's live
Tracking choice after the football comparison above.

`inference.sam31_tracking_native_bundle` selects the pinned **v24 native672**
build from `spring@israel` for `sam3.1-tracking` only. Retain the existing
`sam31_tracking_bundle` and full checkpoint for dense rollback. Remove the
native tracking key and restart the worker to return to dense BF16. This key
does not affect `sam31_w4a4_bundle`, detector-only SAM, YOLO, or CUDA/Thor.

The dedicated worker preserves the frozen causal Object Multiplex adapter,
with independent sessions and object IDs per prompt. It uses native W4A4
image projections/MLPs, W8A8 heads, calibrated INT8 convolutions, 50% uniform
temporal spatial K/V retention, and the original pointer memory. The whole
camera frame is bilinearly resized directly to **672 x 672**, then normalized
in FP32. Returned boxes and masks use the original camera dimensions; no crop
or FOV adjustment is made. Native kernels and compiled regions are replayed
using SYCL graphs; session logic remains outside the GPU graphs. New shapes
still need compilation/capture, and a worker restart resets object identities.
This worker keeps the frozen adapter's retirement policy, not newer changes
to the general-purpose dense adapter.

The source bundle manifest SHA256 is
`34958e69ecb21d124669edb2e8518fe73e256eca9fdad01fc890a6c6afd8df48`.
The runtime separately pins `lib/libdnnl.so.3` (oneDNN 3.13) and the native
`engine_graph.py` helper. Libraries are private to this worker, never installed
system-wide. The source's absolute lock keys require the canonical directory
`/home/spring/sam3_1`; `BUNDLE/sam3_1` points to that directory. Do not overwrite
an existing checkout to satisfy this requirement. Checkpoint/source/calibration
and original graph-provider guards remain enabled. The deployed build uses
Torch `2.14.0+xpu`, git `08187d9e0fba026dc8217405802ab5381dc88d90`.

Israel's completed 208-frame, three-clip `person` evaluation reports:

- Mask J: **87.7925%**, down **1.8719 percentage points** from its optimized
  1008 control. This is not a dense-BF16 equivalence result.
- Mask matches: 234/242; zero mask ID switches, unchanged from that control.
- Mean box IoU: 81.4671% versus 75.8303% overall, but judo box IoU fell by
  2.072 points. Improvements are not uniform across scenes.
- Source GPU-only ordinary/reconditioning frames: **59.25 / 67.02 ms**.
  These exclude host work/transfers and are not local demo FPS measurements.

Those clips overlap development/tuning data, so they are not an independent
holdout or a general open-vocabulary accuracy guarantee. Lower resolution can
lose fine mask detail or small objects. The faster build is an explicit quality
tradeoff, not a lossless optimization. Worker output includes the source receipt,
input geometry, precision, and tradeoff under detection metadata.

September 8 `spring-edge-turret` B580 integration: 40 recorded camera frames
measured 55 ms median tracking / 71.60 ms whole worker. A 32-frame judo run
measured 58/71.60 ms; the two annotated foreground people on frames16–31 had
32/32 mask matches at IoU0.5, decoded display-mask mean IoU0.86074 and no ID
switches. These clips are not independent holdout data. A two-prompt four-frame
smoke verified independent IDs/masks, not warmed two-prompt performance.
After deployment, 41 live non-capture samples measured median 55 ms tracking,
72.68 ms worker, 73.43 ms loop; the local viewer later showed 13.9 FPS.
Cold initial live memory-shape capture took minutes and was excluded from those
steady-state medians. Prompts, camera calibration, servo zeros and non-tracking
SAM were preserved; motors remained stopped.
