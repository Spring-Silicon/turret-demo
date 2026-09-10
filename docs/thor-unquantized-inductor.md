# Thor unquantized Inductor tracking

## Implementation

The CUDA regional tracker now uses BF16 Inductor for the image encoder,
detection encoder/decoder/mask head, temporal memory encoder/attention and
tracking mask decoder. Explicit CUDA graphs retain the existing separate
temporal boundaries. Text remains cached AOT-eager; Python association, CPU NMS
and session bookkeeping remain eager. This is not one whole-model graph.

No weights, input resolution, object limits, memory policy or arithmetic
precision were changed. There is no NVFP4, INT8, TensorRT, token reduction or
frame skipping in this path. Inductor uses static full graphs and
`emulate_precision_casts=True`; its nested CUDA-graph option is disabled because
the existing replay wrappers own the graph buffers.

Both **SAM 3.1 Tracking** and the **Thor side of SAM 3.1 v18** select this same
`sam3.1-tracking` worker. Arc's original tracking and native v18 workers are
unchanged. Box and Mask inference are unchanged.

The compiler policy lives in `sam31_tracking_regions.py`.
`install_tracking_regions(..., heads_backend="aot_eager")` explicitly selects
the comparison baseline; the CUDA default is `"inductor"`. This is an internal
qualification override, not a new frontend model option.

## Temporal graph cache

The original two-entry LRU recaptured memory-attention graphs when memory-frame
and object-pointer counts cycled through more than two exact signatures. A live
Thor snapshot recorded 92 captures in 493 calls. Startup also created short-lived
graphs while the memory bank filled.

CUDA memory attention now admits a signature after its third occurrence and
retains up to 16 graphs without eviction. Other signatures execute the same
Inductor operation directly, preserving all original inputs, temporal memory,
object counts and output ownership. A genuinely new shape can compile once;
compiled direct execution does not repeatedly warm/capture/validate a graph.
The initial eager-on-recompile fallback candidate was rejected after regression
differences; production never switches this stage to eager on a cache miss.

Per-stage telemetry now exposes cache policy/capacity, misses, evictions, replay
calls and direct calls as well as captures. Capture validation remains enabled.
The code affects the CUDA Tracking worker (including the shared v18 selection),
not Arc's separately pinned native tracking implementation or the Mask model.

### Cache repair qualification: 2026-09-10

On agxthor-2, alternating three recorded memory-attention signatures for 36 calls
reproduced 36 captures with the old cache and only three with the retained cache;
outputs were bitwise identical. A separate 128-frame causal tracking regression
also produced bitwise-identical published boxes, masks, scores, centroids and
IDs, with memory-attention captures reduced from 16 to one. This verifies parity
with the installed model, not accuracy against ground truth. Nineteen cache
contract tests and four compiler-policy tests passed locally and inside the
deployment image.

After hot-deploying just the two inference modules, 64 distinct live processed
frames passed validation. Memory attention ended with 53 calls, one capture and
zero evictions. The control backend was not restarted, and the previous Mask
model/prompt selection was restored. Persistent image:
`spring-turret-demo:thor2-retained-graphs-20260910`.

The same two qualified modules were ported to `agxthor-5`, paired with
`spring-edge-turret`, without additional benchmark or smoke runs. Installed
package hashes match the agxthor-2 candidate exactly. Its persistent image is
`spring-turret-demo:thor5-retained-graphs-20260910`, layered on the existing
`proteus-menu-20260910` image so the model menu and other optimizations remain.
The effective launcher is `/opt/spring/turret-demo/run-container-overhead.sh`.
Only the inference subprocess is reloaded; the control backend, model/prompt
selection, configuration, calibration and motor intent are preserved. Deployment
scripts and rollback files are in `/home/spring/thor-retained-graphs.MZ6h8S`.

## Qualification: 2026-09-09

Paired tests ran on agxthor-2 with Torch
`2.8.0a0+34c6371d24.nv25.08` / CUDA 13.0, matching the existing agxthor-5
runtime. They used the same checkpoint, 1008 input, person prompt, frames and
tracking policy. Each clip was followed by 64 repeats of its final frame;
latencies below are the last 32 full-tracker iterations. They are not encoder
microbenchmarks, cold-start timings, or proof of continuous moving-scene FPS.

| Test | AOT-heads tracking | Inductor tracking | AOT worker | Inductor worker | Moving mask IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recorded camera, 30 moving frames | 235.12 ms | 187.29 ms | 268.47 ms | 220.34 ms | 99.81% |
| Judo, 17 moving frames | 241.57 ms | 193.88 ms | 266.71 ms | 218.79 ms | 99.77% |

The worker timing includes image preprocessing and annotation, but excludes
network transport and frontend rendering. The two moving sequences had 84 and
105 matched object observations respectively, with no unmatched objects or
matched-ID disagreements. Full held-frame runs also had no count/ID mismatch.
Maximum centroid differences were 0.41 px / 8.41 px on the moving sequences;
the camera held-frame run had a 20.07 px outlier. This is short regression
agreement against the old implementation, not ground-truth accuracy or
bitwise equivalence. Novel shapes still compile/capture and can pause output.

Evidence and runner on agxthor-2:

- `/home/spring/sam31-tracking-thor2.CwUpg3/outputs/torch28-*.json`
- `/home/spring/sam31-tracking-thor2.CwUpg3/outputs/torch28-*.log`
- `/home/spring/thor-inductor-deploy-20260909/test28.py`

Comparison reports and deployment receipt on the development Mac:
`work/thor-inductor-deploy.B8yGlB/` (relative to the parent workspace).

## agxthor-5 deployment

Live verification completed 126 tracking frames, with 16 sampled results after
frame 64: median full tracking 216.04 ms and worker 246.74 ms. This is a live
scene measurement, not a paired speedup comparison. All per-frame neural regions
reported Inductor and CUDA replay, with zero measured replay error. Capture
counts remained unchanged across sampled frames 65–126. The original Mask /
person selection was restored afterward; the control-service PID, motor Start
state, gains, geometry and saved zeros were preserved.

The release changes only two package files over the existing demo image:

| File | SHA-256 |
| --- | --- |
| `sam31_tracking_regions.py` | `b9b69f87b5731fff5b537cd834263b354830d8e3626bf5c7bf26472be1d2e487` |
| `sam31_tracking_graph.py` (documentation only) | `c6294a78633209b40fdde30dafa2b392d0b02c2193590f6e8c2f13419b16dcfe` |

Base image: `spring-turret-demo:0.17.3-servo-pd-20260909`, image ID
`sha256:8f0f350885349a05498bd036ee121d4e67b8dbcadcbd811807fefe5999d6f6bf`.

Persistent release: `spring-turret-demo:0.17.3-thor-inductor-20260909`, image ID
`sha256:b88ee724cdac9d41832fb3a460f53d4389927a6b6c7cf8c03b8685f229fe0754`.
`/opt/spring/turret-demo/run-container-overhead.sh` selects this image on future
starts. No CUDA, Torch, driver, checkpoint or control-policy upgrade is needed.

The currently running container received the same package files atomically,
without restarting the control service. Its original image ID therefore does
not identify the hot-updated package contents; verify the file hashes and
worker telemetry. The new image contains those same files for future starts.

Prepared Dockerfile, guarded apply/rollback scripts, original files and live
verification receipt are preserved on agxthor-5 in
`/home/spring/thor-inductor-deploy.B8yGlB/`.
The apply script stages the files, then makes a verified root-owned copy inside
the container before atomic replacement. Docker copy on this host produced
UID-1000 files even with `-a`; capability-restricted root cannot chmod those.
Both the deployed files and the persistent image were verified as root-owned
and mode 0644.

For a running tracker, `/api/status` must report BF16, `torch_compile: true`,
`cuda_graph: true`, and these `detection.graph_stages` compiler backends:

- `image_and_detection`: `inductor-image+inductor-heads`
- `memory_encoder`, `memory_attention`, `tracking_masks`: `inductor`
- `text_encoder`: `aot_eager`

Rollback is explicitly guarded against overwriting later releases:
`sudo bash /home/spring/thor-inductor-deploy.B8yGlB/release/rollback.sh`.
It restores this patch and the original persistent launcher, but does not
restart control or touch motors. Reload only the inference worker afterward
if Tracking is active. After a subsequent container replacement, re-inspect
the deployment rather than bypassing the rollback container-ID guard.

CPU routing tests are in `tests/test-tracking-compiler.py` and are included in
`scripts/validate.sh`; they complement, not replace, GPU regression tests.
