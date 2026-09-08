# SAM 3.1 W8A8 megakernel handoff

Updated 2026-09-06 in /home/spring/sam3_1, host israel.

## Read this first

The goal is **not achieved**. The user wants at least 80% of an honest roofline for the per-frame SAM 3.1 work needed by https://github.com/Spring-Silicon/turret-demo, using a custom megakernel and both GPUs efficiently during development.

**Latest user scope update:** also optimize the detection heads for one cached
text prompt. The primary target is **engine-only model execution**, from resident
image tensor through the image encoder and grounding heads to device bounding
boxes/scores. The user explicitly excludes host overhead; preprocessing/JPEG,
host transfers/postprocessing, annotation, network and camera acquisition are
outside the acceptance timing. Cache prompt setup outside the per-frame interval.
Earlier instructions to leave head execution unchanged are superseded; preserve
the original checkpoint/reference detector and confidence/box gates. The earlier
1/2/4/8-prompt worker results remain historical diagnostics, not the new primary
acceptance boundary. Freeze a roofline for image plus one-prompt heads, and measure
the combined engine directly rather than claiming a component sum is its latency.

Current image replay with cached MLP reciprocals, vector-sixteen quantization and grouped native QKV/RoPE is **72.724 ms on GPU0 / 73.202 ms on GPU1**, versus approximately **123 ms** for the dense compiled Torch image stage. Five alternating image rounds show that cached reciprocals save **.772/.616 ms** over the preceding block-store MLP, then vector sixteen saves **.718/.621 ms** over cached vector eight. These are separate controlled comparisons. The candidate still fails **five confidence checks and four retained-box checks**. It is a development candidate, not an accuracy-qualified implementation.

A native queued kernel fuses an entire MLP, but the image graph still launches hundreds of kernels and retains upstream attention and convolutions. There is **no single megakernel for the whole image graph**. The 26.63 ms arithmetic subtotal is **not a completed roofline** or a valid final acceptance denominator.

**Recommended next task:** retain the unchanged image libraries/bias and now `runtime/joint_head_attention_packed512.so` through `packed_head_attention.py:packed_heads`. The directly measured one-cached-prompt engine is **85.004 ms on GPU0 / 84.981 ms on GPU1**. Packed X/Y positional bias with explicit retained half-MM/bias rounding saves **.333/.471 ms** over paired previous native-head controls. Both 72-case head screens and both 24-frame combined-engine checks preserve the prior raw outputs bitwise; original five confidence/four box failures remain. See `engine_only_retained_packed_config.json`, `packed_head_summary.json` (71 artifacts), and section19. Next: implement a calibration-supported activation-rotation candidate and resolve the native norm arithmetic prerequisite for the concrete MLP→next norm→QKV segment. The new norm prototypes still miss 2–4 codes and are not promoted. The engine roofline inventory is broader but incomplete; SFU throughput/ISA probe design is now concrete. Preserve original gates and qualification requirements.

The retained engine runs and their audits are terminal. A new continuation is testing packed head bias, native normalization, and Hadamard diagnostics with agents; check live sessions/agent ownership before starting GPU work. Recheck live processes and tool sessions before starting because another chat shares this workspace. A timeout is not terminal evidence.

The detailed chronological experiment log is scripts/turret_megakernel/STATUS.md. Read its latest sections first: early statements that something is “current” are historical. Authoritative retained performance evidence is results/turret_megakernel/cached_recip_vec16_summary.json. Latest rejected experiments are audited in partial_max_summary.json, r32_g128_summary.json, smoothing_summary.json and flash_attention_summary.json; see section14. They audit actual artifacts and matching source revisions. Native image-attention rejections are audited in native_attention_summary.json (41 artifacts); the latest engine/head evidence is engine_head_summary.json (81 artifacts). Earlier summaries and their evidence remain intact.

## 1. Non-negotiable workload contract

The engine includes the shared, once-per-frame detector image graph below plus the cached one-prompt heads and device box/score decoding from checkpoints/sam3.1_multiplex.pt:

~~~python
backbone = model.backbone.forward_image(
    pixels,
    need_interactive_out=False,
    need_propagation_out=False,
)
return (
    backbone["backbone_fpn"][-1].tensors,
    backbone["vision_pos_enc"][-1],
)
~~~

- Batch 1, FP32 input [1, 3, 1008, 1008].
- FP16 autocast with FP32 master weights.
- Return two final FP16 outputs, each [1, 256, 72, 72] (2,654,208 bytes each, verified from stored tensors).
- Image ViT: 32 blocks, embedding dimension 1024, MLP hidden dimension 4736, 16 heads.
- 28 local-attention blocks use nine 24x24 windows, 576 tokens/window.
- Global-attention blocks 7, 15, 23, 31 use 5184 tokens.
- Detector neck computes only the final detector feature level and its positional encoding.
- Preserve upstream explicit BF16 MLP behavior: FC1 matrix result rounds to BF16 **before GELU**. FC2 uses FP16 operands and rounds its matrix result to FP16 before the FP32 residual addition.
- Text encoder/projection remain unchanged; cache prompt embeddings.
- Geometry encoder, fusion encoder, query decoder and box/class/presence heads
  are now authorized optimization targets for one cached prompt. Keep independent
  original model/reference outputs and unchanged confidence/box acceptance gates.
- No masks, interactive segmentation, propagation/video memory, full-video timings, or training losses in the timed image workload.
- Persistent device buffers and replay are part of engine implementation.
  Host preprocessing/upload/annotation are excluded from the user's latest target.

Do not compare the earlier propagation-backbone/full-video workloads to this detector image boundary. CUDA capture support exists in earlier code but was not tested on this Intel-only host; explicit graph measurements here are XPU/SYCL.

## 2. Acceptance rules

Use the unchanged demo validator in /home/spring/turret-demo/src/spring_turret/sam31_worker.py.

~~~python
scores = (
    logits.sigmoid() * presence.sigmoid().unsqueeze(1)
).squeeze(-1).float()
xyxy = torch.cat((center - size / 2, center + size / 2), -1).clamp(0, 1).float()

near = (reference_scores > 0.47) | (actual_scores > 0.47)
retained = (reference_scores > 0.5) | (actual_scores > 0.5)
# Confidence absolute error <= 0.03 on near.
# Normalized xyxy absolute coordinate error <= 0.01 on retained.
# Same query indices, zero relative tolerance, all outputs finite.
~~~

Preserve the exact implementation and dtypes in the demo; the snippet describes the contract.

The validator raises on confidence before testing boxes. Therefore, a failed confidence check does not prove anything about its boxes. scripts/turret_megakernel/failure_diagnostics.py --device D independently checks both gates on saved XPU failure tensors.

Do not use Hungarian/geometric query matching for acceptance. It is diagnostic only. Do not relax thresholds or change the reference to make a candidate pass.

The user's original known Torch confidence failure had no original fixture supplied; it remains explicitly unresolved in reports. Additional dense failures were reproduced locally. In the latest fresh-cache reports there is one dense baseline failure on each GPU; older runs had three on GPU0 and one on GPU1. Keep the baseline results visible for each run. Do not call a newly reproduced failure the user's original missing fixture.

Passing replay_failure.py or failure_diagnostics.py means the saved failure was correctly reproduced/rejected, not that the detector passed.

## 3. Environment and repository state

| Item | Value |
|---|---|
| Workspace | /home/spring/sam3_1 |
| Demo | /home/spring/turret-demo |
| SAM commit | 660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7 |
| Demo commit | 777516b1d28a0ba4ce82afff57760b45ce9b0c6b |
| Python | /home/spring/sam3_1/runtime/turret-megakernel-venv/bin/python |
| Runtime | Python 3.12, Torch 2.14.0+xpu; torchvision 0.29, triton-xpu 3.8 |
| GPUs | Two Intel Arc B580, 12 GB each, xpu:0 and xpu:1 |
| Compiler | /home/spring/springsilicon/graphs/.tools/dpcpp/bin/icpx |
| Headers/libs | Active turret-megakernel-venv SYCL ABI |
| Checkpoint | /home/spring/sam3_1/checkpoints/sam3.1_multiplex.pt |
| Checkpoint SHA256 | 0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6 |

Prior device inspection: 20 Xe cores/160 vector engines, 128 KiB workgroup SLM, 18 MB L3, 256 KiB L1 per Xe core. Prior sustained loaded clock was 2850 MHz; do not assume this without remeasurement. Published dense INT8 peak is 233 TOPS and memory bandwidth 456 GB/s per B580.

Retained image kernels use AOT BMG compilation with 256 GRFs and explicit correctly rounded division options as recorded by build_joint.py. The retained native decoder-attention partial uses 128 GRFs and the same division/FP-contraction contract. Avoid process-wide SYCL compiler flag changes.

The older /home/spring/yolo/venv environment remains untouched. Early detector SmoothQuant results used Torch 2.10.0+xpu and are not directly comparable to current timings.

The worktree is dirty: README.md modified; results/, runtime/, sam3/optimization/, scripts/detector_smoothquant/, scripts/turret_megakernel/, test/test_detector_smoothquant.py and some older benchmark files are untracked. No optimization changes have been committed or made into a PR. Do not reset/clean/discard them.

Do not edit the external springsilicon checkout without reading its applicable AGENTS.md. No applicable AGENTS.md was found in the SAM/demo paths during this work. Current session instructions enable proactive agent delegation. Coordinate explicit GPU ownership and use at most one GPU experiment per device; CPU research/review can run alongside.

## 4. Current measured results

The retained command uses `--native-queued-library runtime/joint_mlp_cached_recip_vec16.so`.
The native adapter prepares each immutable smoothing tensor and its refined
reciprocals before compilation/capture. Quantization uses vector sixteen; all
explicit casts and compensated division operations are preserved. Every saved
feature/position on all 24 frames is loaded and bitwise equal to the preceding
retained bank on both GPUs; all direct/replay checks pass with zero scheduler errors.
All 72 cases are finite. Independent original XPU gates retain five confidence/
four box failures; each dense baseline retains one confidence/box failure.

| Latest full detector run | GPU0 | GPU1 |
|---|---:|---:|
| Cached reciprocals + vector sixteen, compiled direct | 75.024 ms | 75.732 ms |
| Cached reciprocals + vector sixteen, explicit replay | 72.724 ms | 73.202 ms |
| Median image saving versus cached vector eight, five alternating rounds | .718 ms | .621 ms |

GPU1 full worker for 1/2/4/8 prompts is **107.145 / 121.870 / 151.610 / 209.380 ms**;
same-run dense worker is 157.210 / 172.330 / 201.635 / 259.575 ms. Separate worker
runs include preprocessing/annotation variation; the alternating image rounds
quantify the controlled gain. The preceding cached-vector-eight result is
73.349/73.753 ms and saves .772/.616 ms in its own matched block-store comparison.

Retained MLP SHA256: `bbb4c548e66b46dffbafed93f8bf01656d86ad0d5490f67a93e3039ab4aa360e`.
The QKV library and bias correction are unchanged. `cached_recip_vec16_summary.json`
audits both GPUs' standalone/matched-image/full-detector reports, actual loaded
features/failures, independent gates, compiled replay, worker costs and profile.
It also verifies both real embedded ELF images and 256-GRF/no-scratch MLP metadata.
Sources are in `mlp_cached_recip_vec16_sources`; cached-vector-eight sources are in
`mlp_cached_recip_v1_sources`. See section13 and latest STATUS.md for other trials.

The preceding retained command added `--native-qkv-library runtime/joint_qkv_rope_grouped.so`
to the preceding token-major QKV/RoPE configuration. This selects a native SYCL
joint-matrix kernel with 64 rows, four column groups, 256 GRFs, compile-time local
and global RoPE lengths, and traversal of eight neighboring row tiles per weight
panel. The two half-rounding points, FP32 RoPE order, frequency strides and Q/K/V
layouts are unchanged. The native MLP, calibrated weights/scales/biases and all
model masters are unchanged. Native QKV build metadata overrides Triton QKV tiles.

| Latest full detector run | GPU0 | GPU1 |
|---|---:|---:|
| Native QKV/RoPE, compiled direct | 76.209 ms | 77.250 ms |
| Native QKV/RoPE, explicit replay | 74.017 ms | 74.491 ms |
| Median saving in five alternating native/Triton image rounds | 1.390 ms | 1.580 ms |

The alternating comparisons capture both backends from the same model and shared
MLP workspace, verify all 24 changed frames against the retained feature bank,
and include the stage's once-per-frame pixel copy. All five rounds improve on
each GPU. The non-grouped native tile's matched full-image saving is only .078 ms;
it is not retained despite synthetic-block gains. Initial runtime-length variants
are slower; their reports and binaries remain as rejected evidence.

Both full detector runs check exact direct/replay on all 24 frames. All saved
features/positions are actually loaded and bitwise equal to the preceding Triton
candidate, the fresh GPU1 Triton control and the other GPU. All 72 cases are finite,
scheduler errors are zero, and independent XPU gates preserve five confidence/
four box failures. Each dense baseline retains one failure.

GPU1 full worker for 1/2/4/8 prompts is **107.495 / 122.300 / 151.300 / 209.590 ms**,
versus a fresh same-source Triton control at **109.595 / 124.410 / 153.785 / 211.795 ms**.
These separate worker runs include preprocessing/annotation variation; use the
alternating image runs to quantify the kernel's controlled gain. Same-run dense
worker medians are 157.580 / 171.165 / 203.780 / 258.385 ms.

`native_qkv_summary.json` audits nine standalone suites (18 exact checks each),
three earlier synthetic-block suites (15 exact checks each), three matched-image
suites, four complete detector reports, loaded feature/failure artifacts, original
XPU gate audits, build/source hashes and the profile. Source revisions are in
native_qkv_rope_v1_sources, native_qkv_rope_v2_sources,
native_qkv_grouped_v1_sources and native_qkv_sources.

Retained QKV binary SHA256:
`3f5f5d2f2ab49fe0ed5baf98078ace1ecaf17b2d030ea9e747cf0036aa0f848b`.
The retained MLP binary remains `ca5106ece670d91c053c3c1f959a84dac3ff0e40b6fcdfc1a372aec60e6392aa`.

The preceding Triton QKV/RoPE fusion adds `--fuse-qkv-rope --qkv-rope-config
results/turret_megakernel/qkv_rope_tokenmajor_m64_config.json` to the retained
projection/window/residual configuration. It uses BM64/BN128/BK64, eight warps,
workers0, stages2, block pointers, automatic GRFs and token-major Q/K storage.
V retains its upstream QKV backing strides. The native MLP binary and all
calibrated weights/scales/biases are unchanged.

| Latest verified measurement | GPU0 | GPU1 |
|---|---:|---:|
| QKV/RoPE, compiled direct | 77.991 ms | 78.282 ms |
| QKV/RoPE, explicit replay | 75.694 ms | 75.788 ms |
| Fresh retained-candidate control, replay | 79.351 ms | 77.689 ms |
| Matched replay saving | 3.657 ms (4.609%) | 1.901 ms (2.446%) |

Every returned feature/position on all 24 frames is bitwise equal to the retained
candidate, fresh controls, repeated runs and other GPU. Fresh controls and fused
candidates also check bitwise compiled-direct versus changed-frame replay on all
24 frames. All 72 cases are finite, all scheduler errors are zero, and independent
XPU audits retain five confidence/four box failures. Each dense baseline keeps its
one existing failure.

GPU0 full worker for 1/2/4/8 prompts: **107.625 / 122.390 / 151.490 / 209.080 ms**,
versus the matched retained candidate **111.415 / 125.910 / 154.770 / 212.440 ms**.
These full-worker samples come from the initial fused run (75.829 ms image replay);
GPU1's initial run measured 75.612 ms. The table uses the later repeated runs with
explicit per-frame direct/replay checks.

Preceding evidence: `qkv_rope_summary.json`, `demo_qkv_rope_{control,verified}_gpuD.json`,
`demo_qkv_rope_gpuD.json`, matching `features_qkv_rope_*.pt`,
`qkv_rope_*_xpu_gates.json`, and `profile_qkv_rope_direct_summary.json`.
The summary audits hashes and all loaded feature comparisons. Source snapshots
are `qkv_rope_v1_sources`, `qkv_rope_v2_sources`, and `qkv_rope_sources`.

The preceding projection-only tuning used `--projection-config
results/turret_megakernel/projection_residual_config.json` in addition to both
fusion flags. BM=64, BN=128, BK=64, eight warps, workers=0, stages=2, block
pointers and automatic GRFs apply only to the 32 attention projections. QKV,
MLP, native binary and calibration are unchanged.

| Latest measurement | GPU0 | GPU1 |
|---|---:|---:|
| Tuned projection, compiled direct | 82.174 ms | 81.026 ms |
| Tuned projection, explicit replay | 79.055 ms | 77.815 ms |
| Same-source untuned control, explicit replay | 81.242 ms | — |

Matched GPU0 replay gain: 2.186 ms (2.691%). All 24 frames' returned outputs
match the prior candidate, matched control and other GPU bitwise. All 72 cases
are finite and all scheduler errors are zero. Both independent XPU audits retain
five confidence and four box failures. Each fresh dense baseline has one failure.

GPU1 actual worker for 1/2/4/8 prompts: **112.260 / 127.225 / 156.750 / 214.500 ms**,
versus dense **156.170 / 170.940 / 200.290 / 258.115 ms** in the same run.
Historical projection evidence: `projection_tuned_summary.json`, `demo_projection_tuned_gpu{0,1}.json`,
`features_projection_tuned_gpu{0,1}.pt`, `projection_tuned_cross_gpu.json`,
`projection_tuned_vs_matched_control.json`, `projection_tuned_xpu_gates_gpu{0,1}.json`,
`profile_projection_tuned_direct_summary.json`. Source snapshots and all relevant
artifact hashes are in the summary. No accuracy-qualified implementation exists.

The preceding fusion results are retained below as history:

Resumed work adds `--fuse-window --fuse-attention-residual` to the retained
`joint_mlp_blockstore.so` configuration. This remains a development candidate
with five confidence and four retained-box failures.

| Measurement | GPU0 | GPU1 |
|---|---:|---:|
| Window + attention-residual fusion, compiled direct | 84.285 ms | 83.005 ms |
| Window + attention-residual fusion, explicit replay | 81.075 ms | 79.777 ms |
| Window fusion only, compiled direct | 85.274 ms | 84.050 ms |
| Window fusion only, explicit replay | 82.250 ms | 80.855 ms |
| Fresh original block-store control, explicit replay | 84.983 ms | — |

The matched GPU0 window-only gain is 2.734 ms (3.216%). The subsequent attention
residual gain is 1.126 ms (1.370%) versus a same-source GPU0 window-only control
at 82.201 ms replay. Both fusions together improve GPU0 replay by 3.908 ms
(4.598%) versus the fresh original block-store control. Both returned outputs are bitwise identical across
all 24 frames to the previous candidate and across GPUs. All 72 cases are finite,
and every recorded scheduler error is zero. The same five confidence/four box
failures remain. Each same-run dense baseline has one failure; the original
user-reported failure still lacks its original fixture.

GPU1 actual full worker with both new flags:

| Cached prompts | Dense baseline | Candidate |
|---|---:|---:|
| 1 | 156.040 ms | 114.590 ms |
| 2 | 170.840 ms | 128.810 ms |
| 4 | 200.390 ms | 158.475 ms |
| 8 | 258.140 ms | 215.855 ms |

New evidence: `attention_residual_summary.json`, `demo_attention_residual_gpu{0,1}.json`,
`features_attention_residual_gpu{0,1}.pt`, `attention_residual_cross_gpu.json`,
`attention_residual_vs_window.json`, `attention_residual_xpu_gates.json`,
`attention_residual_xpu_gates_gpu0.json`,
`profile_attention_residual_direct_summary.json`. The summary includes the matched
control, source, report, profile and feature hashes. The block-store native binary
is unchanged. `joint_mlp_handoff_control.so`, rebuilt from current generalized
sources, is byte-for-byte identical to the historical retained binary.

Prior block-load full-image runs (historical):

| Measurement | GPU0 | GPU1 |
|---|---:|---:|
| Block-load candidate, compiled direct | 89.319 ms | 88.697 ms |
| Block-load candidate, explicit replay | 84.800 ms | 84.876 ms |
| Same-device block-store control replay | — | 84.970 ms |

The 0.094 ms difference did not establish a useful gain; block-store was retained
at this historical stage. The current default is the cached-reciprocal/vector-sixteen
library in section4. Block loads remain opt-in.

Earlier block-store runs measured 83.108/83.612 ms replay and 86.364/87.114 ms direct. Those are valid historical measurements but do not establish that a later 84.9 ms run regressed because of code; use matched controls under current runtime conditions.

Latest GPU1 actual full worker:

| Cached prompts | Dense baseline | Block-load candidate |
|---|---:|---:|
| 1 | 158.835 ms | 119.275 ms |
| 2 | 173.655 ms | 133.845 ms |
| 4 | 202.615 ms | 163.605 ms |
| 8 | 260.605 ms | 221.195 ms |

Latest dense image direct/replay: 126.073/122.957 ms on GPU1.

Current failure set (same on both GPUs):
- dance-twirl/00000: person, woman.
- dance-twirl/00045: person, woman, dress.
- All five fail confidence; the first four also fail retained boxes.
- Dress confidence error is approximately 0.0315045, just above 0.03; boxes pass.
- All 72 checked cases are finite; every recorded per-block scheduler error is zero.

Both final image outputs are bitwise identical across GPUs on all 24 frames. Block-load outputs are bitwise identical to block-store outputs; block-store outputs are bitwise identical to the preceding compensated native candidate. This demonstrates equivalence between those implementations on this development set, not equivalence to dense Torch or universal accuracy.

Authoritative evidence under results/turret_megakernel/:
- blockload_summary.json
- demo_blockload_fresh_gpu0.json, demo_blockload_fresh_gpu1.json
- demo_blockload_fresh_control_gpu1.json
- blockload_fresh_cross_gpu.json, blockload_fresh_vs_blockstore.json
- blockload_fresh_xpu_gates.json
- blockstore_summary.json, compensated_summary.json
- features_blockload_fresh_gpu0.pt, features_blockload_fresh_gpu1.pt
- features_blockstore_gpu0.pt, features_blockstore_gpu1.pt

Binary hashes:
- joint_mlp_blockstore.so: ca5106ece670d91c053c3c1f959a84dac3ff0e40b6fcdfc1a372aec60e6392aa
- joint_mlp_blockload2.so: 984b56a7d9ce360899c3bb428b069503bf64c35884b3d7061cee59cd81e1b8eb

Each binary has an adjacent .build.json with command and dependency hashes. Current source headers have subsequently gained opt-in experiments; an old binary's dependency hash need not match today's source. Preserve the old binary and metadata. Build new experiments under new names.

## 5. Profiling and roofline

The latest cached-reciprocal/vector-sixteen direct trace still has 306 GPU kernels:
- Queued native MLPs: 30.203 ms across 32 calls.
- Native attention: 19.442 ms across 32 calls.
- Native QKV/RoPE: 10.589 ms across 32 calls (9.222 local + 1.366 global).
- Attention projections: 4.687 ms across 32 calls.
- Fused normalization/quantization: 5.296 ms across 64 calls.
- Remaining quantization: 1.204 ms across 32 calls.

Sum of kernel durations: 72.432 ms; first-to-last span: 73.640 ms. These are
profiler diagnostics, not unprofiled acceptance timings. Use
profile_mlp_cached_recip_vec16_gpu0_direct.trace.json and its _summary.json.
The preceding Triton-QKV trace has a 75.247 ms subtotal / 76.964 ms span, with
MLP 31.676 ms, attention 19.477 ms and QKV/RoPE 11.900 ms. QKV/RoPE fusion removes
32 separate RoPE/conversion kernels and 32 subsequent layout-copy kernels.
Earlier window fusion removed
28 copies; projection-residual fusion removed another 32 residual-add launches.
The latter removed about 4.6 ms of add kernels but raised QKV/projection time
from 12.051 to 15.532 ms before tuning; the new tile reduces that to 13.516 ms.
The native MLP and attention remain much larger costs.

Use `profile_qkv_rope_direct.trace.json`, its `_summary.json`, and
`scripts/turret_megakernel/summarize_trace.py`. Matched earlier traces are
`profile_projection_tuned_direct.trace.json` (370 kernels, 78.666 ms subtotal),
`profile_window_residual_direct.trace.json` (402 kernels, 81.964 ms subtotal)
and `profile_window_residual_control_direct.trace.json` (430 kernels, 84.733 ms).
The original block-load trace remains valid historical evidence.

Only count GPU kernel events once. CPU operator/device attribution duplicates
kernel time. The replay profiler emits no per-kernel events here; that is a
coverage limitation, not zero GPU work. Unprofiled explicit graph timing works.

Arithmetic bounds in arithmetic_bounds.json:
- 4.609573650432 trillion INT8 image operations / 233 TOPS = 19.7836 ms.
- 0.782757789696 TFLOP attention plus 0.01507590144 TFLOP convolutions, using an **assumed** 116.5 TFLOP/s FP16/BF16 matrix rate = 6.8484 ms.
- Image arithmetic subtotal: 26.6319 ms.
- At 80% of that arithmetic ceiling: 33.2899 ms/image.
- Each unchanged prompt's grounding: 0.280918288384 TFLOP, arithmetic subtotal 2.4113 ms.
- Full-frame arithmetic subtotals for 1/2/4/8 prompts: 29.043/31.455/36.277/45.922 ms.

Excluded: LayerNorm, GELU, softmax, RoPE, activation quantization, memory/cache traffic, scheduling, input/preprocessing/upload, output/annotation. Build and freeze a roofline for the actual final algorithm and timed scope before claiming 80%.

Supporting files: workload_inventory_v2.json, resource_inventory.json, arithmetic_bounds.json, inventory.py, resource_inventory.py, roofline.py.
The corrected workload counter includes aten._addmm_activation and attention; dense image matrix total is 5.407407341568 TFLOP.

`roofline_rate_verification.json` now verifies the 233 TOPS assumption against
Intel's B580/B570 media deck, PDF page 55, which explicitly labels the figure
as dense INT8. The same page specifies 456 GB/s. The downloaded primary PDF is
roofline_sources/b580_media_deck.pdf, SHA256
437c236f6f9a2737d112994b4cfef3dbcaf78195fa4eb6eb7944b0f168eb807c.
Source: https://download.intel.com/newsroom/2024/client-computing/Intel-Arc-B580-B570-Media-Deck.pdf.
This rate verification does not complete the roofline; FP16/BF16 rate evidence,
loaded clock and non-matrix/memory/scheduling resource models remain outstanding.

Resource inventory distinguishes logical accesses from compulsory DRAM traffic. Prior warmed copy probes achieved about 390 GB/s; raw native INT8 GEMMs reached about 137–142 TOPS. Do not present these as complete-model roofline efficiency.

Primary hardware source: https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc/desktop/b-series.html

The earlier user asked about FMR results. The current exact-detector evidence does not establish an FMR comparison; locate and verify the original FMR workload/results before making one.

## 6. What was implemented and what was learned

### Initial detector-only SmoothQuant
sam3/optimization/detector_smoothquant.py installs 128 image Linear routes only, retains FP32 masters, handles upstream FC1's BF16 route, and supports final-detector-neck-only execution. Per-output-channel INT8 weight scales and per-token dynamic activation scales are used.

Under old Torch 2.10:
- Original exact image direct: 207.252 ms.
- Final neck only: 199.023 ms, bitwise returned-feature equality.
- Final neck plus W8A8 alpha .7: 193.882 ms; 12 local development checks passed.
- Original attention could not be captured in that runtime.
- Separate Triton attention plus allocation-pool replay worked but was slower and failed W8A8 accuracy.
See scripts/detector_smoothquant/RESULTS.md. Its old gate implementation and workload/runtime results must not replace the current unchanged demo validator.

### New Torch environment and Triton kernels
Moved development into a separate Torch 2.14.0+xpu environment with working native attention capture.
Implemented persistent INT8 GEMMs, exact 65,536-entry BF16 GELU lookup, fused LayerNorm/quantization, persistent scratch and graph replay.
Early custom GEMMs alone were slower; LUT and norm fusion reduced image time into roughly the 105 ms range.
Alpha .65 with calibration-only bias correction became the development choice.
The validator expanded from 12 to 36 to 72 cases; earlier passes did not survive broader validation.

### Triton MLP megakernels
mega_mlp.py fuses FC1, BF16-rounded GELU, dynamic quantization and FC2 using dynamic task claims and per-row release/acquire dependencies.
A separate state reset remains. Scratch is shared sequentially across blocks; state/error buffers are per block.
Task batching and scheduling tuning improved standalone performance from roughly 4 ms to roughly 1.4 ms.
Parallel quantization, row ownership, grouped ordering and several register-heavy variants did not beat the retained path.
Whole-image times around 100–102 ms still had detector failures.

### Numerical consistency
Frozen references exposed cross-GPU feature differences that scheduler changes did not explain.
Initial positional-add/LayerNorm reduction configurations differed between caches; tiny initial differences amplified downstream.
The preferred fix uses upstream native LayerNorm only at image trunk.ln_pre via native_norm.py. ln_post is Identity for this checkpoint.
Per-block norms are fused into quantization.
A fixed Triton initial-norm alternative was numerically different and rejected.
Text encoder norms must remain unchanged.
Fused MLP residual preserves FC2 half rounding before adding the FP32 residual.

### Attention and window layout
Triton attention prototypes were slower than upstream micro_sdpa at the exact 576/5184-token shapes, so native attention remains.
The resumed work implements an explicit combined window/MLP-residual block forward.
It passes synthetic changed-input eager/compiled/replay tests on both GPUs, then
bitwise full-image comparisons on all 24 frames. Matched image replay improves
2.734 ms on GPU0. A further `--fuse-attention-residual` option adds the attention
residual inside the projection GEMM epilogue, preserving FP16 result rounding
before FP32 addition. It uses the same upstream QKV, RoPE, native SDPA and head
layout, requires exact detector shape and identity layer scales, and checks the
supported attention path. Full-image outputs remain bitwise identical; see the
latest measurements above. Both options remain explicit command-line flags.

### Native SYCL matrix kernels
Built exact-shape packed-B joint_matrix/DPAS variants, preserving explicit scaling/bias rounding.
Native raw INT32 GEMMs reached approximately .355 ms FC1 / .368 ms FC2.
Fused epilogues round FC1 to BF16 and use the exact GELU LUT; FC2 rounds to half then adds residual.
A three-kernel native MLP pipeline reached about 1.31 ms; image replay around 93 ms.

### Native queued MLP
Row-owned single-kernel variants were too slow. Dynamic queues, completion batching, subgroup/vectorized quantization and early last-producer quantization improved the native MLP.
Preferred configuration before store experiments:
- rows 64, column groups 8, matrix fragments per subgroup 4 (64 columns).
- task batch 2, subgroup quantization, vector width 8.
- inline-ready mode 1.
- compensated division mode 5.
- AOT BMG, 256 GRFs, no implicit floating-point contraction, correct division compiler option.

Mode 5 refines native reciprocal and uses FMA residual correction for division. Tests use an independent FP64-intermediate oracle that explicitly rounds each FP32 operation.
The compensated candidate's final image outputs match the native correctly rounded divide candidate bitwise on all 24 frames. This is not a proof for every IEEE input.
Standalone MLP improved to about 1.13 ms, image replay about 86.6 ms.
Other reciprocal shortcuts failed exact checks. More aggressive producer-owned FC2/ready-queue variants were slower; bounded polls and scheduler error checks remain.

### Store and load experiments
Local-memory staging of accumulator epilogues was exact but slower: FC1 .556–.564 ms versus .505–.510 controls; FC2 .529–.591 versus .471–.475.
A measured BMG accumulator mapping gives each lane one column and eight rows for an 8x16 accumulator. Both GPUs verified all 128 coordinates.
Scalar subgroup stores made little difference.
Vector subgroup stores combining four fragments reduced standalone FC1 to .429 ms and FC2 to .459 ms.
Integrated MLP replay became 1.025–1.036 ms, and image replay initially 83.1–83.6 ms. This is the retained block-store candidate.
Residual block loads alone did not help; cached scale/bias plus block loads showed a tiny standalone gain but only .094 ms in the latest matched image comparison. Keep opt-in.

### Wider-tile experiments: rejected (both row counts)
joint_tile.hpp now accepts a defaulted CM fragment count. build_joint.py --matrix-columns 8 creates a 128-column subgroup tile.
Queued addressing, workspace metadata and scheduler audits use the actual width.
With rows=64, column groups=4/8:
- Compiler allocated 256 registers and reported about 100/97 spilled registers.
- Both passed all five changed-input direct/replay checks and all four output comparisons.
- MLP replay: 2.237/2.345 ms, far slower than ~1.03 ms.
- No full-image run or promotion.
Reports: mlp_wide_c4.json, mlp_wide_c8.json; build logs build_mlp_wide_c4.log and build_mlp_wide_c8.log.

The requested rows=32 follow-up is also complete. Four/eight column groups pass
all five changed-input direct/replay checks and scheduler audits. Builds emit no
spill warnings, but replay is 1.229/1.177 ms versus matched 1.027/1.041 ms controls,
19.6%/13.1% slower. GPU1 control also passes `torch.compile`. No full-image run or
promotion for either wider tile. Evidence: `wide_r32_summary.json`,
`mlp_wide_r32_c4_gpu0.json`, `mlp_wide_r32_c8_gpu1.json`,
`mlp_handoff_control_gpu{0,1}.json`.

### Native matrix reduction-loop experiments: no promotion

`build_joint.py --matrix-k-unroll {2,4}` and `--unchecked-a` add opt-in experiments.
All variants pass five changed-input/replay exact checks with zero scheduler
errors. Unroll-two/four replay is 1.036/1.049 ms versus controls 1.025/1.051 ms.
Unchecked A loads are 1.026/1.035 ms on GPU0/GPU1, without a consistent gain on
both matched devices. No promotion or full-image run. The current-source default
rebuild `joint_mlp_matrix_loop_control.so` is byte-for-byte identical to the
historical block-store binary.

`matrix_loop_summary.json` and `native_loop_isa/` retain timing, build and actual
embedded-Zebin resource evidence: 256 GRFs, SIMD16, four EU threads, one barrier,
32 bytes SLM for all variants. Static disassemblies are preserved for the retained
and unchecked-load variants; their instruction counts are not dynamic counts.

### Grouped MLP traversal and prefetch: rejected

`joint_mlp_queued.cpp` now has opt-in grouped task mappings for FC1 and FC2.
`build_joint.py --fc1-row-group 8 --fc2-row-group 8` changes only logical task
order; per-row counts, readiness and dynamic producer claims remain intact.
FC1-only, FC2-only and combined variants are slower. In the initial five-round
matched tests, median savings versus the historical binary are approximately
-.069, -.021 and -.084 ms respectively. Later shared-workspace FC2 grouping
versus a named default control loses .014 ms. The combined build passes five
independent changed-input/replay checks including torch.compile; FC2-only also
passes alone, and a 64-row case checks the final partial group.

The new `compare_queued_mlp.py` exposes a runtime issue when two unnamed native
variants coexist. The second library loses GPU0 during warmup, while that same
candidate passes alone. Explicit stream synchronization does not resolve it.
Rebuilding with a distinct named SYCL kernel makes the paired tests pass. This
supports a kernel-ID collision between DSOs with different device argument
layouts. Queued builds now receive a `QueuedMlpKernel` tag derived from all source
bytes and the compile command. Preserve old binaries; do not load two different
old unnamed queued variants together. Use named builds or separate processes.
GPU0 recovered in fresh processes without resetting or interrupting GPU1.

The matched harness also fixes its initial missing repository import path and
explicitly waits for workspace initialization/weight packing before new-stream
launches. Later comparisons share native activation buffers, scheduler state and
packed weights to reduce allocation/cache differences. All eight completed
matched suites have six exact oracle checks and five alternating timing rounds.
Initial import and device-loss logs/reports remain as failed evidence.

`--matrix-prefetch 2 --matrix-prefetch-fc2` adds L1 prefetch to FC2's reduction
loop; distance4 with `--matrix-prefetch-cache l2` tests L2 instead. Both preserve
integer arithmetic and every rounding step. The L2 variant loses .012 ms in a
shared-workspace comparison; L1's later median saving is only .00135 ms and does
not establish useful improvement. No grouping, prefetch or newly named default
binary is promoted into the image graph. The named default itself has measured
runtime/code-generation variation versus the historical binary; do not attribute
every difference to the scheduling option alone.

`mlp_scheduling_summary.json` audits nine builds, three standalone suites, eight
matched suites, failed paired runs and source revisions. Snapshots are
grouped_mlp_v1_sources, grouped_mlp_stream_v1_sources, grouped_mlp_named_sources
and mlp_scheduling_sources. The retained `joint_mlp_blockstore.so` is unchanged.

## 7. Calibration / accuracy experiments

Calibration uses 16 disjoint real frames from eight DAVIS training sequences:
bear, bmx-bumps, boat, boxing-fisheye, breakdance-flare, bus, car-turn, cat-girl, first/middle frames.

Current baseline calibration:
- alpha .65.
- results/turret_megakernel/bias_alpha065.pt (mean error correction only).
- SHA256: 62dfab2c401679d4f1228277928171f43b5f880b9a70d478f6cac04d33b09d69.
- Activation maxima: results/detector_smoothquant/final_direct/activation_maxima.pt.

Layerwise alpha, gain, attention-only/MLP-only correction combinations were tested. None resolved the full detector gates; native-quantizer reevaluation changed some failure sets. Do not infer equivalence from equal failure counts.

Reusable calibration_bank256:
- 128 image layers.
- 4096 rows/layer: 256 deterministic rows from each of 16 frames.
- Inputs retain FC1 BF16 versus other FP16 effective dtype; weights/biases are FP32.
- About 3.828 GB.
- manifest.json and calibration_bank256_audit.json verify layer coverage, source/checkpoint identity, dimensions and hashes.
- Calibration and detector validation frames are disjoint.

Hessian/GPTQ-style rounding:
- calibrate_hessian.py, merge_hessian.py, hessian_all.json.
- Fixed INT8 per-output scale grid, alpha .65, damping .01, block size 128.
- Covariance uses reconstructed quantized activations.
- FP32 inversion checks failed initially; FP64 CPU covariance/Cholesky/inverse fixed numerical stability.
- Sharded across both GPUs, all 128 layers, all improved calibration MSE; median ratio .744675.
- Full detector worsened to six confidence failures, two box failures. Rejected.
- Final-feature RMS improved only about 3% on average.
- See hessian_summary.json and hessian_xpu_gates.json.

Activation-error fitting:
- A regularized weight adjustment was fitted before Hessian rounding to compensate activation quantization.
- Alternating first/middle calibration-frame split, 2048 rows fit and 2048 evaluate per layer.
- Four FC1/FC2 pilot layers in blocks 0/15.
- Damping .01/.1/1.0: all 12 comparisons lowered fitting error but worsened separate-frame evaluation MSE.
- Rejected before another full-detector run.
- See activation_fit_summary.json.

Sequential propagated block-bias correction is now also tested and rejected.
`calibrate_propagated_bias.py` starts from verified dense initial-normalization
outputs and feeds each compiled W8A8 block its own corrected prefix output.
It adjusts only the 32 FC2 effective biases using dense block-output channel
means, preserving FP16-before-residual rounding and FP32 master weights.
First frames fit; middle frames evaluate, all disjoint from the 24 detector
frames. Strengths 1, .25 and .05 worsen separate-frame final-feature MSE by
7.26%, 2.65% and 3.84%, respectively; fitting-frame final features also worsen.
No full detector run or promotion for these corrections. The native scheduler
passes throughout. `propagated_bias_summary.json` verifies all correction files,
unchanged 96 other biases and teacher/report hashes. The calibration controls
have tiny aggregate MSE differences (maximum final-feature relative change
3.18e-6), so do not infer bitwise reproducibility for that harness. Rejections
use each trial's own control. `propagated_teacher.pt`
contains reusable dense initial activations, channel means, 256 sampled output
rows per block/frame and final feature targets. These are calibration data only.

Actual downstream-loss calibration is now implemented, but has not fixed accuracy.
`test_head_gradients.py` confirms finite nonzero gradients through frozen eval-mode
heads (about 4.44 GB peak allocation). Sam31Engine disables gradients globally;
only offline gradient steps enter `torch.enable_grad()`. Head parameters stay
frozen, and production head logic/validator are unchanged.

`downstream_calibration_manifest.json` defines 16 disjoint calibration frames with
three fixed prompts each. `downstream_calibration_refs.pt` contains full eager
teacher detections, dense image features and pixels; `downstream_calibration_features.pt`
contains current W8A8 image features/positions. Both banks have been loaded and
verified. Extraction preserves one dense baseline and three candidate failures
in 48 calibration cases. These are fitting/selection data, not qualification data.

`calibrate_feature_affine.py` trains only 256 bounded feature gains and 256 biases,
with first frames fitting and middle frames selecting. Gradients use unchanged
eager heads; selection evaluates unchanged compiled heads and the original gate.
Two weighted-loss trials (regularization .1 and 1, twelve epochs) slightly lower
selection loss, but one adds an evaluation failure and neither improves gates.
Two `--objective gates` trials use the near-score/retained-box thresholds and a
half-tolerance training margin, selecting first by actual validator failures.
Their best evaluation losses improve 3.14%/.44%, but both keep the existing
boxing-fisheye/00043 person failure and two fitting failures. Reject all four;
no image integration or 72-case detector trial was warranted. Evidence is in
`feature_affine_summary.json`, with source snapshots and hashed finite artifacts.

`calibrate_feature_lowrank.py` adds a nonlinear rank-16 channel bottleneck with
8448 trainable parameters, a fitting-feature RMS normalization and bounded tanh
output (5% or 10% of channel RMS). The first 5% trial can remove fitting failures
but adds selection failures. The 10%/weak-regularization trial stops during epoch
10 with a nonfinite gradient; its report remains incomplete and its log is kept.
Two additional trials add dense compiled feature distillation and a smaller
learning rate. Neither removes the selection failure. Distillation weight .1
reduces selected separate-frame gate loss by 48.72%, retaining one selection
failure and reducing fitting failures from two to one.

That fixed correction was screened through unchanged compiled heads on cached
features for the 24 development frames. `evaluate_feature_lowrank.py` uses the
training correction's eager FP32 arithmetic and the original dtype-preserving
XPU validator. The control reproduces five failures; correction yields seven
confidence/four box failures, adding car-roundabout/00020 person and /00060 person.
Reject all four trials. No feature correction is integrated or included in image
timings. `feature_lowrank_summary.json` actually loads all four saved checkpoints
and failure artifacts, checks their identities, shapes, finiteness, bounds and
fit/selection disjointness, and preserves the incomplete trial explicitly.

`block_surrogate.py` now supplies an exact retained-kernel forward with an
explicitly approximate backward: recomputed fake-quantized blocks, detached token
scales, straight-through INT8 rounding and FP32 dequantized-weight GEMMs. Real
normalization, casts, RoPE and native attention remain differentiable. It is a
calibration surrogate, not the derivative of discrete kernels. `test_block_gradients.py`
checks actual pretrained local block 0 on GPU0 and global block 31 on GPU1, three
changed inputs each. Forwards are exact; gradients are finite and nonzero; masters
are unchanged and have no gradients; heads are frozen. The global probe's initial
bear activation is explicitly not a real block-31 input. Initial bitwise-backward
repeat assertions fail. Revised probes preserve those reports and measure input
gradient differences <=1.99e-7 (roughly .6–1.9 ppm relative RMS), comparable to
direct surrogate repetition. Bias gradients repeat exactly. This diagnostic
gradient tolerance does not change detector gates.

`capture_tail_inputs.py --start-block 24` initially exposes a native-workspace
aliasing bug: a clone inside the full compiled image returns the final block
output as the supposed prefix. The partial reports and unverified diagnostic
capture remain rejected. `calibration_capture.py` fixes this with a mutating
custom snapshot op and distinct preallocated buffers. Only
`tail24_calibration_fixed.pt` is a verified tail input bank: all 16 full-image
feature/position checks match the retained calibration bank bitwise; all 16
split-tail checks and first/last block outputs also match bitwise. The audit loads
all real [1,72,72,1024] FP32 prefixes and checks identity, shape and finiteness.

`calibrate_tail_bias.py` trains only 8192 bounded FC2-bias correction values for
blocks 24–31 through actual native forwards and surrogate backwards, with frozen
heads/master weights. A custom exact-forward neck adapter recomputes the frozen
neck for backward. Epoch zero asserts exact retained features on all 16 frames.
The inference addition order is preserved: rounded master bias plus complete
correction. Exports copy already-combined XPU corrections and replace exactly
the eight selected correction buffers in the existing 128-entry artifact.

Both twelve-epoch trials complete. Bound .05 / gate objective / feature-loss .1
selects epoch 4, fitting failures 2->1 and selection failures unchanged at one;
selection loss improves 5.1704->3.303. Bound .01 / weighted objective / feature-loss
1 selects epoch 1 with the same fitting/selection failures and only a tiny loss
improvement. Both peak near 5.99 GB allocated. The stronger fixed artifact gets
a full 24-frame/72-case detector screen: five confidence/four box failures remain,
but dress now passes and car-roundabout/00020 person newly fails at .0323078.
Reject it. A disjoint-calibration-only strength sweep over 0/.25/.5/.75/1 selects
strength 1; its loaded corrections are bitwise equal to the rejected endpoint.
Do not select a different strength from development failures.

`tail_calibration_summary.json` audits completed probes, rejected initial captures,
verified inputs, finite correction artifacts/scopes, fit/selection disjointness,
the strength sweep, full detector report and independent XPU gates. Sources are
in block_gradients_v1_sources, tail_capture_v1_sources, tail_bias_v1_sources and
tail_calibration_sources. `calibration_image.py` deliberately retains the preceding
Triton-QKV arithmetic path for these calibration experiments; its outputs are
bitwise equal to the new native candidate on the measured frames.

The full encoder now has verified inputs too: `tail0_calibration_fixed.pt` passes
all 16 full-image and split-32-block checks, including first/last block outputs,
bitwise. All real FP32 prefixes are loaded and audited. The same trainer accepts
start_block0 and trains 32768 FC2-bias values while keeping masters/heads frozen.
The eight-epoch bound .05 / LR .003 / gate-loss trial selects epoch0; its loaded
corrections are all bitwise equal to bias_alpha065.pt. Reject it.

`head_surrogate.py` adds an offline exact compiled-worker-head forward with a
frozen eager-head surrogate backward. `test_exact_head_gradients.py` checks four
calibration cases (ordinary bear plus the three existing failed cases). All three
outputs match the actual compiled heads bitwise and all feature gradients are
finite/nonzero. Eager and compiled heads have different raw outputs, though their
gate pass/fail decisions match on these four cases. This wrapper changes only
offline calibration forwards/backwards, not deployed head parameters or gates.

`calibrate_tail_bias.py --compiled-head-forward` then runs a smaller twelve-epoch
all-block trial: bound .01, LR .0003, regularization1, feature-distillation1 and
weighted objective. Epoch3 is selected: fitting failures 2->1, selection failures
stay at one, selection loss 1.747232->.704153. Peak allocation is 6.54 GB (the
first full-block trial peaks at 6.49 GB). Exactly 32 FC2 corrections change, with
maximum delta .0001328401; the other 96 remain bitwise unchanged. Full 24-frame/
72-case screening completes with exact direct/replay and zero scheduler errors
at 74.210 ms replay, but now has six confidence/three box failures. It fixes
dance-twirl/00045 woman and dress while introducing car-roundabout/00060 vehicle,
car-roundabout/00060 person and drift-chicane/00000 person. Reject it. The dense
baseline's existing one confidence/box failure remains visible.

A calibration-only strength sweep over 0/.25/.5/.75/1 again selects strength1;
its loaded corrections are bitwise equal to the rejected endpoint. Do not select
another strength using development failures. `full_bias_summary.json` audits
real input/correction/feature/failure tensors, head probes, original gates,
selected epochs and source versions. Sources are in full_bias_v1_sources and
full_bias_sources. Both all-block trials remain rejected. Weight-scale learning
was subsequently implemented and rejected in section13; smoothing with regenerated
INT8 codes is implemented and rejected in section14. Independent code learning
remains open. Further unchanged 16-frame bias sweeps lack
support from this evidence. Never train on the 24 development frames and call
them untouched holdout data.

## 8. Data and critical harness behavior

- all_manifest.json: 24 frames / 72 frame-prompt cases; file hashes checked.
- DAVIS root: /home/spring/yolo/eval/davis2017-v1/DAVIS/JPEGImages/480p.
- Validation sequences include blackswan, car-roundabout, camel, goat, horsejump-high, dog, cows, drift-chicane, parkour, dance-twirl.
- frozen_refs.pt: about 420 MB; includes unchanged full eager detector references, dense compiled image features and CPU pixels.
- Frozen identity includes validation/checkpoint/repository/runtime metadata. Pixel equality is checked.
- These 24 frames have repeatedly informed development. They are not untouched final holdout.
- Feature snapshots contain cloned CPU outputs, since captured output buffers are borrowed and reused.
- Failure artifacts are referenced by each report's failure_artifact fields.

demo_candidate.py:
- Uses the exact demo ImageEncoder boundary and unchanged prompt/head paths.
- Separately reports baseline_eager, compiled direct, explicit replay, and optional actual full worker.
- --native-queued-library selects the native MLP candidate.
- Native path requires persistent backend, mega MLP, fused norm and fused residual.
- Native in-place alpha sweeps are rejected because packed weights need rebuilding.
- --weight-calibration verifies complete 128-layer manifests and conflicts with bias/layerwise/sweep settings.
- --native-image-norm changes only image initial normalization.
- --profile-image PREFIX captures diagnostic direct/replay traces after validation and normal timing.
- --fuse-qkv-rope with --qkv-rope-config selects the preceding Triton token-major QKV/RoPE kernel; requires attention residual fusion and real RoPE.
- --native-qkv-library runtime/joint_qkv_rope_grouped.so selects the retained native QKV kernel and packs weights once before capture. It requires --fuse-qkv-rope and overrides Triton QKV tile settings. Native QKV and MLP build/hash fields are separate.
- --verify-direct-replay requires exact direct/replay feature equality on every changed frame outside timing. Clone replay outputs before calling direct because native workspaces are shared.
- run_complete and all_checks_pass have different meanings: completion does not mean acceptance.

test_mega_mlp.py:
- Three changed direct inputs and two changed explicit graph replays.
- Checks hidden BF16 activations, quantized INT8 activations, activation scales, final output.
- --precise-quant-reference is required for the preferred native quantizer.
- Audits scheduler counts/readiness/error state, including actual compiled tile width.
- --compile exercises torch.compile integration.
- Timings include separate state reset and sustained warmup.
- --diagnostic-phases 1/2 are partial-kernel diagnostics, not full-MLP timings or additive phase decomposition.
- build_joint.py now gives queued variants unique named SYCL kernel IDs. Old differently compiled unnamed binaries can collide when loaded together; use named builds for compare_queued_mlp.py or isolate them in separate processes.

## 9. Reproducible commands

Run from /home/spring/sam3_1. These snippets use task-specific variables, not HOME/CODEX_HOME.

~~~bash
SAM31_PY=/home/spring/sam3_1/runtime/turret-megakernel-venv/bin/python
SAM31_CXX=/home/spring/springsilicon/graphs/.tools/dpcpp/bin/icpx
~~~

### Build a current-source control without overwriting the historical binary

~~~bash
"$SAM31_PY" scripts/turret_megakernel/build_joint.py \
  --cxx "$SAM31_CXX" --variant queued \
  --rows 64 --column-groups 8 --matrix-columns 4 --task-batch 2 \
  --quant-subgroup --quant-vector 16 --inline-ready 1 --recip-mode 5 --precompute-recip \
  --correct-divide --aot-bmg --block-store --block-load 0 \
  --output runtime/joint_mlp_vec16_next_control.so
~~~

### Build a grouped native QKV control

~~~bash
"$SAM31_PY" scripts/turret_megakernel/build_joint.py \
  --cxx /home/spring/springsilicon/graphs/.tools/dpcpp/bin/icpx \
  --variant qkv --rows 64 --column-groups 4 --qkv-grouped --aot-bmg \
  --output runtime/joint_qkv_handoff_control.so
~~~

Use a new output name; the retained binary is joint_qkv_rope_grouped.so. Local
and global tests use test_native_qkv_rope.py --device D --library LIB --output
UNIQUE.json, with --global-attention for length5184. Native QKV build metadata
records source/library hashes and its exact AOT command separately from the MLP.

### Standalone correctness and timing

~~~bash
OMP_NUM_THREADS=4 "$SAM31_PY" scripts/turret_megakernel/test_mega_mlp.py \
  --device 0 --native-queued --native-owned-rows 64 --bm 64 --workers 160 \
  --native-owned-library runtime/joint_mlp_cached_recip_vec16.so \
  --fuse-residual --precise-quant-reference \
  --output results/turret_megakernel/handoff_control_gpu0.json
~~~

For the second device, use --device 1 and a unique output filename. Add --compile for one checked run after a candidate passes direct/replay.

### Full image and actual downstream detector

Use the fresh caches below; the earlier turret-inductor-gpuD caches encountered corruption. Set SAM31_DEVICE to 0 or 1 and use separate processes/cache directories/output names.

~~~bash
SAM31_DEVICE=0
OMP_NUM_THREADS=4 \
TORCHINDUCTOR_CACHE_DIR=/home/spring/sam3_1/runtime/blockload-fresh-inductor-gpu$SAM31_DEVICE \
TRITON_CACHE_DIR=/home/spring/sam3_1/runtime/blockload-fresh-triton-gpu$SAM31_DEVICE \
"$SAM31_PY" scripts/turret_megakernel/demo_candidate.py \
  --device "$SAM31_DEVICE" --backend persistent --alpha .65 \
  --gelu-lut --fuse-norm --mega-mlp --native-image-norm --fuse-mlp-residual \
  --fuse-window --fuse-attention-residual \
  --projection-config results/turret_megakernel/projection_residual_config.json \
  --fuse-qkv-rope --qkv-rope-config results/turret_megakernel/qkv_rope_tokenmajor_m64_config.json \
  --native-qkv-library runtime/joint_qkv_rope_grouped.so \
  --verify-direct-replay \
  --native-queued-library runtime/joint_mlp_cached_recip_vec16.so \
  --bias-correction results/turret_megakernel/bias_alpha065.pt \
  --validation-manifest results/turret_megakernel/all_manifest.json \
  --reference-cache results/turret_megakernel/frozen_refs.pt \
  --save-features results/turret_megakernel/features_handoff_gpu0.pt \
  --kernel-config results/turret_megakernel/lut_config.json \
  --mlp-config results/turret_megakernel/mlp_batch8_config.json \
  --output results/turret_megakernel/demo_handoff_gpu0.json
~~~

Add --full-frame on one GPU to time 1/2/4/8 prompts. Add --profile-image results/turret_megakernel/profile_handoff on a diagnostic run if the new change needs profiling. Do not overwrite existing evidence.

Important integration detail: the standalone harness supports 32 rows, but demo_candidate.py currently hardcodes native_owned_rows=64 when --native-queued-library is supplied (search for native_owned_rows=64). Before a successful 32-row candidate can be tested in the full image, change that setup to derive/validate SAM_ROWS from the selected binary’s .build.json, and use a copied MLP config with bm=32. Merely changing the config file is insufficient because the hardcoded value overrides it. Preserve the other settings and the adapter’s row/state consistency checks; retain 64-row behavior for existing binaries.

### Feature equivalence and independent gates

~~~bash
OMP_NUM_THREADS=4 "$SAM31_PY" scripts/turret_megakernel/compare_features.py \
  results/turret_megakernel/features_blockstore_gpu0.pt \
  results/turret_megakernel/features_handoff_gpu0.pt \
  --output results/turret_megakernel/handoff_feature_comparison.json

OMP_NUM_THREADS=4 "$SAM31_PY" scripts/turret_megakernel/failure_diagnostics.py \
  --device 0 --report results/turret_megakernel/demo_handoff_gpu0.json \
  --output results/turret_megakernel/handoff_xpu_gates.json

python3 scripts/turret_megakernel/summarize_trace.py \
  results/turret_megakernel/profile_handoff_direct.trace.json \
  --output results/turret_megakernel/profile_handoff_direct_summary.json
~~~

## 10. Immediate task for the next agent

1. Read this file and the last STATUS.md sections. Check GPU/compiler processes;
   both chats share the workspace and devices. Preserve all existing evidence.
2. Retain cached-reciprocal/vector-sixteen native MLPs, fused windows/residuals, selected projection,
   token-major QKV/RoPE and now `--native-qkv-library runtime/joint_qkv_rope_grouped.so`.
   Keep native image normalization, alpha .65 and bias_alpha065.pt. Verified full
   detector replay is 72.724/73.202 ms with 306 kernels; five confidence/four box
   failures remain. No learned tail/feature correction is promoted.
3. Native QKV is complete as a development improvement. Runtime-length kernels
   are slow; compile-time lengths plus eight-row-tile grouping deliver the gain.
   Keep Q/K stride (L*1024,64,1024,1), V stride (L*3072,64,3072,1), offset2048,
   both half-rounding points and explicit FP32 real/imaginary products. Use
   `compare_native_qkv_image.py --device D --library LIB --output UNIQUE.json`
   for same-model alternating full-image comparisons; synthetic-block gains
   overpredicted the non-grouped kernel's full-image saving. Do not repeat that
   rejected non-grouped experiment unchanged.
4. Grouped queued-MLP scheduling and FC2 prefetch are implemented and rejected.
   Read mlp_scheduling_summary.json before changing those paths. The new paired
   harness shares buffers/weights and requires uniquely named SYCL kernels;
   otherwise old native variants can collide at runtime. Keep historical binary
   names intact. Register-resident INT8 and FP16 probes now have independently
   checked outputs and verified real DPAS loops, with no memory operations inside
   the repeat loop. INT8 demonstrates 232.685 TOPS and FP16 103.437 TFLOP/s. These
   are synthetic diagnostics, not image roofline scores. Shared-local-memory
   operand staging and a second verified fragment-layout/block-load design are
   implemented and rejected. Both emit the same normalized many-small-SLM-load
   pattern. Do not repeat either design or a blind stage-K sweep. Complete-MLP
   subgroup clocks now identify quantization as substantial, and cached reciprocal/
   vector-sixteen improvements are retained. FC1 epilogue partial maxima are now
   implemented and exact, but lose .098/.100 ms in complete-MLP timing. Rows32/
   128-GRF also loses .141 ms. Do not repeat those unchanged designs. Six Triton
   attention variants regress with large reported spills. Before a native SYCL
   attention implementation, read section15: FP16 A8x16/B16x16/C8x16 lane mappings
   are now verified on both GPUs. Four native attention designs use explicit
   fragment lifetimes with no scratch, but all regress. Do not repeat them
   unchanged. Native decoder split-key masked attention is now retained as a
   development improvement; see sections16–17. Use real inputs, original gates
   and combined engine-only XPU events for subsequent head experiments.
5. Full-32-block FC2-bias calibration is implemented and rejected, including the
   actual compiled-head-forward variant. It fits into about 6.54 GB and can lower
   calibration loss while adding development failures. tail0_calibration_fixed.pt
   is the verified full-block input bank; the prior tail24 bank remains valid for
   its own scope. Do not repeat the same 16-frame bias/strength sweep unchanged.
   The expanded 64-frame/192-case bank is frozen and verified: 32 fitting frames
   from eight sequences and 32 selection frames from eight other sequences; all
   ten development sequences are excluded. expanded_tail0.pt matches all full
   image and split-32-block outputs bitwise. Use explicit --calibration-manifest,
   --calibration-features, --calibration-references and --calibration-split flags
   with the generalized capture/trainer. Nonlegacy banks cannot silently use
   the old parity split. The six-epoch trial selects epoch4, reduces selection
   failures 15->14, but retains five development confidence failures while adding
   two new car-roundabout failures. Do not promote it or repeat the same expanded
   bias sweep unchanged. Output-channel weight-scale learning is now implemented:
   its six-epoch expanded-bank trial selects unchanged epoch0 and is rejected.
   Smoothing with regenerated INT8 codes is now implemented. Its per-prompt trial
   aborts in epoch3 on a nonfinite gradient and selects unchanged epoch0. Its
   grouped three-prompt/frame trial completes six epochs and selects epoch1,
   selection failures 15->13, but introduces drift-chicane/00026 car failures on
   both GPUs. Reject both; independent code learning remains open. Preserve the
   split and gates. The calibration_image.py helper deliberately still uses the earlier
   MLP library used to verify the expanded captures; do not silently change its
   arithmetic while comparing old calibration/training evidence.
6. Keep original master weights and reference detector gates unchanged. Head
   execution is now an optimization target; frozen calibration-head backwards
   below remain separate offline surrogates. Block and head
   backwards are explicitly approximate surrogates even when their forwards are
   exact. Export already-combined XPU bias corrections with the inference addition
   order. Calibrate only on the defined fitting partition and choose checkpoints
   only on its separate selection partition. Never choose hyperparameters from
   the reused 24-frame development detector screen or call it untouched
   qualification. Exact forward checks do not make the training surrogate exact.
7. Any worthwhile candidate needs matched controls, all 24 frames/72 prompts on
   both GPUs, --verify-direct-replay, loaded feature comparisons, independent XPU
   gate audits, and paired combined engine-only timing with one cached prompt.
   Host-inclusive 1/2/4/8-prompt worker costs are now historical diagnostics, not
   required acceptance timing. Preserve baseline failures.
   Update STATUS.md, this handoff and hashed summaries; use unique output names.
8. Build and freeze a complete workload roofline. The 26.63 ms arithmetic subtotal
   is still incomplete. Attention, norm/softmax/GELU/RoPE, quantization, memory,
   scheduling and the precise resident image-to-device-box boundary need explicit
   treatment, including the one-prompt heads. Native image attention replacement
   and whole-model kernel scheduling remain unresolved; do not call a captured
   graph with hundreds of kernels one megakernel or claim 80% efficiency.

The final deliverable still requires the requested custom megakernel scope,
an honest complete frozen roofline and >=80% measured efficiency, unchanged
passing detector gates followed by untouched qualification data, and direct/replay
plus directly measured combined engine-only costs with one cached prompt. No workload substitution or relaxed gates.

## 11. Interruptions, caches and chat handoff

Some runtime files were found zero-filled/truncated after interruption/environment restart:
- features_blockload_gpu0.pt and features_blockload_gpu1.pt are zero length; do not use them.
- demo_blockload_gpu1.json is incomplete.
- demo_blockload_retry_gpu0/gpu1 failed with EOFError and generated modules lacking call in old compiler caches.
- Valid replacement evidence is named demo_blockload_fresh_*, features_blockload_fresh_*.
- Use runtime/blockload-fresh-inductor-gpuD and runtime/blockload-fresh-triton-gpuD, and verify caches if another restart occurs.
- Never infer a process is stopped from an observation timeout. Poll its existing session/process first; restart only after terminal evidence or a missing handle.
- Keep per-device benchmark isolation. Do not run competing jobs on the same GPU while measuring.
- Never report a .pt artifact as valid merely because its report says run_complete; actually load/compare it.

The user also requested a duplicate chat. It was created and verified through Codex with 31 turns / 2012 visible items:
- Name: SAM 3.1 hillclimb — fork.
- Thread ID: 01a0779a-508a-79d2-ab0c-61aca0302264.
- Open with: codex resume 01a0779a-508a-79d2-ab0c-61aca0302264.
- The source had a zero-filled record and duplicate ordinal numbering; only the copy was repaired. One already-corrupt blank record was omitted.
- A temporary staging thread was archived. The original chat was not edited.
- This chat administration did not advance model performance or complete the active goal.
- Both chats share the same worktree/GPU resources; coordinate before running concurrent work.

No optimization build/test is intentionally left running at this handoff. Start with a process check, retain the tested QKV/RoPE fusion, and advance the upstream accuracy/remaining native-kernel work. Do not redo the rejected tile, loop or feature-correction sweeps unchanged.


## 12. Preceding continuation: verified matrix rates and expanded calibration

The retained native MLP/QKV binaries and bias are unchanged. This continuation
adds hardware diagnostics, a rejected standalone operand-staging kernel, and a
larger verified calibration bank plus a rejected fixed bias trial.

- `matrix_issue_summary.json`: three real native libraries/four completed suites,
  12 exact checks per suite. Actual extracted DPAS loops verify the counted work
  and absence of memory operations inside the repeat loop. INT8 16x64/128-GRF
  tiles demonstrate 230.692/231.254/232.685 TOPS on GPU0 at 2048/4096/8192 repeats;
  the 2048-repeat GPU1 result is 230.109 TOPS. FP16/FP32-accumulator rates on GPU0
  are 102.698/103.246/103.437 TFLOP/s. Longest marked windows sample 2850 MHz.
  These rates are synthetic diagnostics; they do not complete or lower the
  theoretical workload roofline. `measured_matrix_rates.json` preserves that scope.
- `slm_operand_summary.json`: a 24-KiB shared-operand panel is exact on both real
  GEMM shapes, but five alternating same-input rounds reject it. FC1 takes
  2.665844 ms versus .359624 ms control; FC2 takes 3.783564 versus .344560 ms.
  ISA has many small SLM loads, barriers and 256 GRFs with no allocated scratch.
  Do not integrate this implementation or repeat a stage-K sweep unchanged.
- `expanded_calibration_summary.json`: 64 frames/192 cases, split by sequence into
  32 fitting and 32 selection frames. All ten development sequences are excluded.
  All actual references/features/prefixes are loaded and finite, all 64 direct/
  replay pairs and all full/split-32-block outputs are bitwise exact. Candidate
  calibration failures are seven fitting/15 selection; dense failures are three/
  four. Independently, candidate gates fail 21 confidence/10 boxes across 22 cases;
  dense gates fail seven confidence/five boxes. The box-only candidate failure is
  lindy-hop/00048 person. Neither calibration partition is final qualification.
- `expanded_bias_summary.json`: six epochs, bound .01, LR .0003, regularization1,
  feature-distillation1, weighted objective and actual compiled-head forward.
  Epoch4 is selected: fitting failures remain seven, selection failures 15->14,
  selection loss 4.700244->7.660167. Only 32 FC2 corrections change, max delta
  .0003084566; all other 96 are bitwise unchanged. Peak allocation 7,941,592,576
  bytes; training/evaluation loop elapsed 937.563 seconds.

The fixed epoch4 artifact completes all 24 development frames/72 prompts on GPU1
with exact direct/replay, finite outputs and zero scheduler errors. Replay is
74.238249 ms. Independent original gates still reject five confidence/three box
cases: car-roundabout/00000 vehicle, car-roundabout/00020 person, dance-twirl/00000
person and woman, and dance-twirl/00045 person. It removes dance-twirl/00045 woman
and dress but adds the two car-roundabout cases. Do not promote it; retained bias
is still bias_alpha065.pt. Dense baseline retains its one confidence/box failure.
The missing original user fixture remains unresolved. No second-GPU development
screen or full-worker timing is warranted for this rejected correction.

`calibration_bank.py` requires explicit frozen splits for nonlegacy banks. Use
`--calibration-manifest results/turret_megakernel/expanded_calibration_manifest.json`,
`--calibration-features .../expanded_calibration_features.pt`,
`--calibration-references .../expanded_calibration_refs.pt`, and
`--calibration-split .../expanded_calibration_split.json` with the generalized
capture/trainer. The verified full-block bank is `expanded_tail0.pt`. Legacy
16-frame loading is also verified and retains its old eight/eight parity split.

Sources are preserved in matrix_issue_v1_sources, matrix_issue_v2_sources,
matrix_issue_sources, slm_operand_sources, expanded_calibration_sources and
expanded_bias_sources. Actual extracted binaries/disassembly are in matrix_issue_isa
and slm_operand_isa. Auditors verify data and binary/source hashes; raw historical
reports and binaries are preserved. No optimization/test process is left running.


## 13. Preceding continuation: quantization improvements and rejected scale learning

The retained MLP is now `runtime/joint_mlp_cached_recip_vec16.so`, SHA256
bbb4c548e66b46dffbafed93f8bf01656d86ad0d5490f67a93e3039ab4aa360e.
Build with queued rows64/column-groups8/task-batch2/inline-ready1/recip-mode5,
`--precompute-recip --quant-subgroup --quant-vector 16 --block-store`, AOT BMG,
256 GRFs, explicit correct divide and no implicit FP contraction. The adapter
reads build metadata; `prepare_smooth` runs before compilation/capture and the
smoothing tensor must remain immutable. Both native matrix rounding points,
GELU LUT and compensated division ordering remain unchanged. QKV and bias retain
their previous hashes and no trained correction is promoted.

`cached_recip_vec16_summary.json` audits both GPUs' six-check standalone suites,
a five-check compiled/direct/changed-replay suite, two 48-check same-model image
comparisons, both complete 72-case detector reports, actual loaded 24-frame
features and failure tensors, independent original XPU gates, profile and worker
costs. Every matched round improves. Standalone savings over cached vector eight
are .019812/.018639 ms; matched image savings are .718165/.620573 ms. Final direct/
replay is 75.024/72.724 ms on GPU0 and 75.732/73.202 on GPU1. Independent gates still
fail five confidence/four boxes; dense retains one confidence/box failure. All
outputs are finite, scheduler errors zero and direct/replay exact on all frames.

Actual embedded ISA is saved in `mlp_cached_recip_isa` and
`mlp_cached_recip_vec16_isa`. `extract_zebin.py --all-images` handles separate
MLP/preparation images; the auditor checks actual byte slices and SHA256s. Both
MLPs have 256 GRFs, 32 bytes SLM, no scratch allocation and 192 static DPAS
instructions. Static load counts are not dynamic traffic or a causal attribution
of the speedup. The preparation kernels do not appear in the timed image graph.

`cached_recip_summary.json` preserves the earlier vector-eight improvement:
73.349/73.753 ms image replay and .772226/.615964 ms matched savings over the old
block-store MLP. Its old control binary and metadata are preserved, but the
original joint_tile.hpp source bytes are unavailable. This is explicitly recorded
as a historical source-reproduction gap; all new candidate dependencies resolve
to preserved bytes. The old queued C++ source was recovered with a full hash match
and documented in `recovered_blockstore_sources/recovery.json`.

Other completed experiments:

- `mlp_clock_summary.json`: complete-MLP subgroup clocks, five alternating rounds,
  six exact arithmetic checks and .002720 ms median instrumentation overhead.
  Each round has 40 active/120 idle workers. Active leader lifetime fractions are
  FC1 42.072%, quantization 26.043%, FC2 wait .265%, FC2 29.951%, unmeasured 1.669%.
  No cross-subgroup alignment, conversion to time units or image-wall-time share
  is inferred. Raw counts, interval scope and source identities are audited.
- `slm_fragment_summary.json`: actual A/B operand coordinates are independently
  verified in five changed/direct/replay checks. Contiguous staged fragments and
  subgroup vector loads still emit 64 SLM loads per 16 DPAS. FC1 2.734777 ms versus
  .364387 control; FC2 3.983296 versus .367303. Six exact checks and five alternating
  rounds per shape; rejected before image integration. No scratch allocation.
- `scale_learning_summary.json`: exact-forward output-channel scale training is
  implemented with explicitly approximate block/head backwards. Two eight-check
  probes verify changed scales, finite nonzero gradients and frozen masters/heads.
  A six-epoch expanded-bank trial trains 315,392 bounded multipliers at bound .02,
  LR .0003, regularization1 and feature loss1. All trained epochs lose selection
  to epoch0. The actual artifact has all-one multipliers and bitwise unchanged
  biases; no development screen is needed. Peak allocation 7,951,058,432 bytes.
  Do not repeat this unchanged trial. Smoothing was subsequently implemented and
  rejected in section14; independent INT8-code learning remains open.

Final sources are in `mlp_cached_recip_vec16_sources`, `mlp_cached_recip_v1_sources`,
`mlp_clock_v1_sources`, `slm_fragment_sources`, `scale_sources`,
`scale_gradient_v1_sources` and `scale_training_v1_sources`. Auditors preserve
historical reports and locate the corresponding source revision by hash.
All jobs have terminated. The active goal is not complete: detector accuracy,
untouched qualification, whole-image megakernel and honest >=80% roofline remain.


## 14. Latest continuation: rejected MLP, smoothing and attention experiments

The retained configuration and its 72.724/73.202 ms image replay are unchanged.
No new kernel or learned parameters are promoted. Both GPUs were used for
independent experiments, with one GPU job at a time per device.

- `partial_max_summary.json` audits FC1 epilogues that compute 74 partial maxima
  per activation row, reducing the later quantizer's first pass from 4736 values
  to 74. A separate native entry point and workspace preserve the retained ABI.
  Both GPUs pass six output checks and independent checks of all 383,616 partial
  values per changed input. All five alternating complete-MLP rounds lose:
  median regressions .097547/.099737 ms. Actual embedded ISA has 256 GRFs,
  32 bytes SLM and no scratch. Reject before image integration.
- `r32_g128_summary.json` audits rows32, column-groups8 and 128 GRFs. The paired
  harness derives each library's row count and allocates the correct separate
  scheduler state while sharing activations and weights. Six exact checks pass;
  all five rounds lose on GPU0, median regression .140844 ms. Actual ISA confirms
  128 GRFs, 32 bytes SLM and no scratch. No second-GPU or image run is warranted.

`smoothing_surrogate.py` learns 249,856 bounded input-channel multipliers across
all 128 image linears. Smoothing, weight scales and INT8 codes are regenerated
from frozen effective master weights on XPU; native packed weights are copied
in-place to preserve captured addresses. Eight local/global probes verify unit
regeneration, exact actual native forwards, changed codes/output, finite nonzero
gradients and frozen masters/heads. Backwards still use detached maxima and a
straight-through surrogate; exact forwards do not establish exact derivatives.

`calibrate_tail_bias.py --learn-smoothing` exports the actual XPU quantization
buffers with unchanged base bias. `demo_candidate.py --layerwise ARTIFACT`
validates and installs all 128 tensors before native preparation. Omit a separate
`--bias-correction` when loading this artifact, which already includes the base
corrections. The offline adapter intentionally rejects cached reciprocal/native
QKV workspaces; calibration_image.py retains the earlier verified arithmetic.

Two fixed trials use the expanded frozen 32-frame fitting/32-frame selection split:

- `expanded_smoothing`: per-prompt updates, bound .02, LR .0003, regularization1,
  feature loss1. Only epochs0/1/2 finish; epoch3 aborts on a nonfinite parameter
  gradient. Selection failures are 15/22/17, so the actual saved artifact remains
  unchanged epoch0. Its report correctly has run_complete=false. Do not describe
  this as a completed six-epoch trial or restart it unchanged.
- `expanded_smoothing_frame`: `--group-frame-prompts` averages all three head
  losses for one image backward per frame, 32 updates per epoch; LR .0009, other
  settings unchanged. Six epochs complete in 539.874 seconds with peak allocation
  8,158,789,632 bytes. Epoch1 is selected solely from calibration: fitting failures
  7->9, selection 15->13, selection loss 4.698495->3.689177. Maximum multiplier
  change is .0004166961. All 64 initial image outputs match the frozen bank exactly.

The fixed epoch1 candidate completes 24 frames/72 prompts on both GPUs, with all
outputs finite, zero scheduler errors and exact direct/replay. Actual loaded
feature/position tensors are bitwise equal across GPUs; positions are unchanged
from the retained bank. Direct/replay is 75.434/73.210 ms on GPU0 and 75.476/73.307
on GPU1. Original independent gates reject four confidence/three box cases on
GPU0 and five confidence/three boxes on GPU1. Both introduce drift-chicane/00026
car; GPU1 additionally rejects dance-twirl/00045 woman near the confidence
threshold. Identical image features do not imply identical compiled-head decisions.
Both dense controls retain one confidence/box failure. GPU1 actual worker for
1/2/4/8 prompts is 106.885/121.505/151.100/209.295 ms, dense
156.945/171.840/201.005/259.295. These are separate worker measurements, not a
controlled claim of speedup. `smoothing_summary.json` audits both trial statuses,
loaded parameters/features/failures and independent gates. Reject both trials.

`flash_attention.py` is a standalone FP16 attention prototype at exactly
[9,16,576,64] and [1,16,5184,64], using the real token-major Q/K and interleaved V
strides. It uses FP32 dot accumulators and online max/sum rescaling, then FP16
probabilities for the value multiplication. Six suites each pass six numerical/
changed-replay checks against independent FP32 math (relative RMS < .002) and
complete five alternating timing rounds. They are not bitwise oneDNN replacements
and have not passed detector gates. Every round loses in every configuration:

| Shape / tile / warps / GRFs | oneDNN median ms | Prototype median ms |
|---|---:|---:|
| Local / 64x64 / 4 / default, GPU1 | .365390 | 1.115665 |
| Local / 64x64 / 4 / 256, GPU0 | .360115 | 1.110215 |
| Local / 64x64 / 8 / 256, GPU0 | .360650 | .800500 |
| Global / 64x64 / 8 / 256, GPU1 | 2.424006 | 6.454282 |
| Local / 64x64 / 16 / 256, GPU0 | .360585 | 1.408949 |
| Global / 32x64 / 8 / 256, GPU1 | 2.423571 | 11.541578 |

Explicit compiler queries report n_regs=256 and n_spills=11456 for four warps,
6016 for the later variants. The installed Triton driver maps n_spills directly
to Level Zero spillMemSize; it is not a register count. Units are not independently
established here. An invalid grf_mode=large attempt fails compilation and supplies
no timing evidence. `flash_attention_summary.json` audits the six valid runs,
source versions, driver property and downloaded official oneDNN/PyTorch sources.
The oneDNN revision reported by Torch is 80afa71049cd69a3df32adcccb623b12cd7baa22;
downloaded files at that revision do not independently prove installed GPU source
identity. Do not repeat a blind warp/tile sweep. Native FP16 fragment mapping and
explicit register lifetimes are the next proposed attention experiment.

Source snapshots: `mlp_partial_max_sources`, `mlp_r32_g128_sources`,
`mlp_r32_g128_audit_sources`, `smoothing_training_v1_sources`,
`smoothing_frame_sources`, `smoothing_sources`, `flash_attention_v1_sources`,
`flash_attention_v2_sources`, `flash_attention_sources`. Actual MLP ISA is preserved
in `mlp_partial_max_isa` and `mlp_r32_g128_isa`. The four latest summaries verify
14/13/32/16 hashed evidence artifacts respectively. All experiment/audit processes
have terminated. Accuracy qualification, a whole-image megakernel and the complete
frozen roofline remain unfinished; the active goal is not complete.


## 15. Native FP16 image attention: exact fragment probes, four rejected designs

The native FP16 fragment probe verifies actual A8x16, B16x16 row/column/packed,
and C8x16 coordinates: every lane owns its column and element index selects its
row. Both GPUs pass five exact changed-input/direct/replay checks, including
accumulator-to-A half conversion and independent products. An initial uninitialized
conversion produced zeros; explicit initialization fixes it. A later normal-random
CPU FP32 oracle differed by 2.384e-7 while the native result matched FP64 rounded
to FP32. The final exact probe bounds exponents, so products are exactly representable;
it does not loosen the coordinate or conversion checks.

Four standalone native image-attention designs pass eight finite numerical and
changed-replay checks each at both actual lengths. Relative RMS against independent
FP32 math is below .002; these are not detector qualification or bitwise oneDNN
checks. All five alternating timing rounds lose in every suite:

| Design | Local oneDNN / candidate ms (GPU0) | Global oneDNN / candidate ms (GPU1) |
|---|---:|---:|
| 8 queries/subgroup, 128 GRFs | .360741 / .486352 | 2.399447 / 4.186242 |
| 16 queries/subgroup, 256 GRFs | .360626 / .444368 | 2.388101 / 3.769736 |
| 16 queries, vector key loads | .360961 / .647738 | 2.390713 / 5.794259 |
| Transposed scores, lane/query layout | .360185 / .692302 | 2.390513 / 6.161766 |

Actual extracted ISA contains the two length-specialized kernels in each library,
128/256 GRFs as requested, and no allocated scratch or SLM. Removing spills alone
did not beat oneDNN. Reject all four designs before image integration and retain
native oneDNN image attention. Do not repeat the same variants unchanged.
`native_attention_summary.json` audits 41 hashed artifacts and matching build/source
revisions. Source snapshots and real ISA are preserved under native_attention*,
half_layout* and attention_*_isa in results/turret_megakernel. These timing probes
use the preceding CPU-wall-plus-sync method, not the newer engine-only XPU events.

## 16. One cached-prompt engine boundary and decoder optimization

The user explicitly added detection-head optimization, then clarified that only
engine/model latency matters. The new boundary starts with resident normalized
FP32[1,3,1008,1008] pixels and ends with device normalized xyxy, scores and keep mask.
Prompt encoding/setup, host preprocessing, transfers, host decoding, annotation,
camera/network acquisition are excluded. Use XPU events around the combined
captured image+head+device-decode replay; a sum of component measurements is not an
engine latency. The original checkpoint/reference and same-query gates remain fixed.

`current_image.py` upgrades the older calibration context to the current cached
vec16 MLP and grouped native QKV, without changing the old calibration helper.
All 24 feature/position pairs from the real worker path are bitwise equal to the
retained bank on GPU1, with zero scheduler errors. `profile_one_prompt.py` verifies
24 unchanged compiled-head/direct-replay outputs on each GPU for cached person.
The actual direct traces contain 834 kernels, totaling 13.809/13.997 ms on GPU0/1:
30 attention kernels account for 8.826/8.886 ms, 246 GEMMs for 2.429/2.453 ms.
Six image self-attentions account for about 5.8 ms; six masked decoder cross-attentions
for about 2.58 ms. Replay traces expose no individual kernels and are not used to
count them. Old stage timings include input copies; old 102.762 ms frame-to-host-box
measurement is historical and outside the user's latest acceptance boundary.

`engine_graph.py` owns persistent captured buffers and measures XPU event intervals.
`benchmark_engine_only.py` directly captures retained image, original one-prompt
head and native device decoding into one graph. The first compiled-decode attempt
failed score bit equality despite exact raw heads, boxes, keep mask and replay.
Capturing the original eager tensor decoding operations fixes this: the GPU1
`engine_only_baseline_native_decode_gpu1.json` completes all 24 frames with all
three raw outputs, all six direct/replay outputs and all three decoded outputs
bitwise exact, finite, and with zero scheduler errors. Its 30-sample GPU-event
median is **87.3121355 ms**. This is a directly measured engine baseline, still with
the retained W8A8 detector failures and hundreds of kernels; it is not a qualified
whole-model megakernel or completed roofline result.

A frozen cached-prompt compiler trial (`cached_frozen_head_gpu0.json`) saves only
.0180475 ms median over five rounds and changes raw logits by up to .0201788 and
boxes by .00856927. Reject it without promotion. Its initial changed-replay check
could alias a later direct result; independent saved control comparisons still
establish changed outputs. The current test clones replay outputs before direct
execution; this corrected revision was not rerun for the already rejected trial.
Initial and corrected scripts are preserved separately in engine_only_v1_sources
and engine_only_native_decode_sources.

`capture_head_attention.py` freezes six actual eager decoder input cases: frames
0/13/23, layers 0/5, cached person, real retained image features. Queries are
[1,8,201,32], keys/values [1,8,5184,32], token-major strides [256,32,256,1]. The
finite FP16 mask is contiguous [1,8,201,5184]. This is a kernel probe bank, not a
fitting or detector qualification set. Actual data and source hashes are recorded.

The Triton split-key prototype (512 keys/partition, 11 partitions plus merge) passes
12 numerical/changed-replay checks on each GPU but loses .113333/.104897 ms. Its
partial kernel reports spillMemSize=9920; the merge reports zero spills. Reject that
partial kernel. Native `joint_head_attention_split512.so` replaces only the partial
stage with explicit small fragments, preserving the Triton merge. Actual native
ISA requests 128 GRFs with no scratch or SLM. All 12 numerical/changed-replay checks
pass on each GPU. Five alternating XPU-event rounds give oneDNN/native medians
**.447369/.1001045 ms on GPU0** and **.450677/.1032815 ms on GPU1**, with median
paired savings .3472645/.3471355 ms. Outputs are not bitwise oneDNN equivalent.
The roughly 4.4x standalone speedup carries through to full heads and combined
engine timing; the completed gate/audit evidence and development retention decision
are recorded in section17.
Native source/adapter/test/build snapshots are in native_split_head_v1_sources;
Triton-only revisions remain in split_head_triton_v1_sources.


## 17. Native decoder split attention retained for the development engine

`optimized_heads.py` replaces precisely the six one-prompt decoder cross-attention
calls, leaving head weights and all other attention paths unchanged. It rebinds
only model_misc.F inside a context, avoiding a global torch.nn.functional mutation.
An opaque functional custom op enforces actual FP16 Q/K/V token-major strides and
the contiguous FP16 mask before native memory access; its fake output has the
matching token-major layout. Both compiled-head runs observed all six replacements
and identical actual layouts on 450 warmup/capture/direct invocations. The partial
kernel and Triton merge are captured together. This is two kernels per attention,
not a whole-head or whole-model megakernel.

`test_optimized_heads.py` runs dense controls, retained-image/original-head controls
and retained-image/native-head candidates on all 24 frames/72 prompts, preserving
all three raw outputs for each case. Both GPUs complete 72 poisoned changed-input
replay checks with owned outputs cloned before direct calls, bitwise direct/replay
and all-finite outputs. The original validator and independent box gate report:

| Path, on each GPU | Confidence failures | Box failures |
|---|---:|---:|
| Fresh dense compiled control | 3 | 2 |
| Retained W8A8 image + original head | 5 | 4 |
| Retained W8A8 image + native head | 5 | 4 |
| Native head vs retained original head | 0 | 0 |

The candidate and retained failure identities are the same, including the dance
person/woman failures and the confidence-only dress failure. Preserve the actual
fresh dense-control counts; older controls with one failure do not replace them.
The existing bank is reused development data. Passing the incremental head screen
does not make the original-reference detector gates pass or qualify the W8A8 image.

Five alternating device-event head rounds save **1.9699485 ms on GPU0** and
**1.912474 ms on GPU1**, reducing the head from about 14.23/14.35 ms to 12.27/12.44 ms.
`benchmark_optimized_engine.py` then measures complete resident pixel-to-device-box
captured graphs with paired original-head controls and the same image implementation:

| Combined engine-only timing | GPU0 | GPU1 |
|---|---:|---:|
| Original-head control, median of round medians | 87.247370 ms | 87.389115 ms |
| Native-head candidate, median of round medians | 85.210521 ms | 85.427214 ms |
| Median paired saving | 2.033047 ms | 1.984792 ms |

All five rounds improve, with 20 GPU-event samples per path per round. Uploads,
prompt setup and host work are outside the interval. All 24 frames pass exact
control-head/control-decode, candidate-head/candidate-decode, six-output candidate
direct/replay, finiteness, incremental head gates and zero image scheduler errors.
The combined outputs are compared to separately computed standalone heads on the
actual retained feature bank, so this checks the image/head connection as well.
Output tensors are raw logits[1,200,1] FP32, raw boxes[1,200,4] FP32, presence[1,1]
FP16, decoded xyxy[1,200,4] FP32, scores[1,200] FP32, and keep[1,200] bool.

`engine_head_summary.json` completes an 81-artifact audit, verifies matching source
revisions, loaded six-case input tensors, real native ISA, trace kernel counts,
timing rounds, exact graph checks, and re-evaluates all actual saved dense/retained/
candidate outputs from both GPUs on XPU0 using the unchanged original validator
plus independent confidence/box checks. Saved numerical gate maxima and all gate
decisions reproduce exactly. All 41 native-image-attention and 81 engine/head
evidence hashes were independently reverified after completion. Two initial audit
plumbing failures (external source path labeling, TorchVersion allowlist) were
fixed; their logs remain, and neither changed experimental outputs or gates.

Retain `joint_head_attention_split512.so` through `optimized_heads.py` as the
current development head. The image configuration is unchanged. The historical full-mask native head scope and invocation template are in
`results/turret_megakernel/engine_only_retained_config.json`; section19 records
the subsequently retained compact-bias implementation. A new paired run is:

~~~bash
OMP_NUM_THREADS=4 \
TORCHINDUCTOR_CACHE_DIR=/home/spring/sam3_1/runtime/blockload-fresh-inductor-gpu0 \
TRITON_CACHE_DIR=/home/spring/sam3_1/runtime/blockload-fresh-triton-gpu0 \
runtime/turret-megakernel-venv/bin/python \
  scripts/turret_megakernel/benchmark_optimized_engine.py \
  --device 0 --native-library runtime/joint_head_attention_split512.so \
  --output results/turret_megakernel/YOUR_NEW_UNIQUE_RUN.json
~~~

Use the corresponding GPU1 cache/device for a concurrent second-device run. The
harness refuses to overwrite a report. Sources are frozen in optimized_head_v1_sources,
optimized_engine_v1_sources and engine_head_audit_sources; earlier source versions
and failed trials remain intact. All described jobs are terminal. No external demo
files were modified, and nothing was committed or published.

Next work remains substantial: the unchanged head profile identifies about 5.8 ms
of image-token self-attention, plus mask construction and MLP work; evaluate
specific fusion designs with actual inputs before promoting them. Mask bias is
separable into X/Y components, but fusing that construction into native attention
has not been implemented or measured. Continue image accuracy and whole-model
scheduling work, and freeze an honest roofline for the entire resident one-prompt
engine. Five confidence/four box failures, untouched qualification, a complete
custom megakernel and the >=80% roofline objective remain unresolved.


## 18. Engine roofline inventory and independent FC2 code trial

The previous goal turn made verified progress: native decoder attention reduced
the directly measured engine to 85.211/85.427 ms without adding original detector
failures. In this continuation, current artifacts and their hashes were rechecked.
The session now explicitly enables proactive agent delegation; the old handoff
note forbidding unrequested subagents is superseded. GPU jobs remain isolated by
device, with CPU-only source/roofline work in parallel. Do not assume a GPU is free
without checking current process/session ownership.

`engine_roofline_inventory_v1.py` and its JSON/Markdown reports provide a larger,
CPU-reproducible full-engine inventory. The actual compiled one-prompt head traces
agree on 280,865,449,984 logical FP16 matrix FLOPs, 52,838,400 fewer than the eager
inventory. The image has 4,609,573,650,432 INT8 operations and 797,833,691,136 FP16
attention/convolution FLOPs. Their shared-XMX arithmetic service subtotal is
**29.042797994 ms**, explicitly not a complete acceptance denominator or efficiency
score. The report catalogues image, cached heads and native device decoding,
generated device-call sites, 107 fused head-program dtype/expression inventories,
4,408,201,464 logical softmax probability exponentials, native split padding and
source-address traffic. Cache-reusable bytes are not relabeled as compulsory DRAM.

Intel's primary Xe2 throughput table establishes 2,048 FP16/BF16 versus 4,096 INT8
XMX operations/core/clock. Together with Intel's identification of B580 as Xe2-HPG,
this supports the same-clock derived FP16/BF16 peak116.5 TFLOP/s from the published
233 TOPS dense INT8 rate. At exactly2.85 GHz the unrounded rates are233.472/116.736;
choose and record a clock convention for final acceptance. Sources and29 hashes
are bound to the report. Its byte-identical CPU rerun and all29 hashes were checked.
Remaining requirements include exact combined graph/ISA expansion, dynamic vector/
SFU/reduction/conversion/atomic work, defensible throughput/co-issue ceilings, cache
communication and intended-megakernel overlap/dependency bounds. The goal remains
unachieved; this inventory does not turn the arithmetic subtotal into a roofline.

`accuracy_code_fc2_capture_v2.pt` freezes actual native FC2 hidden values, INT8
activations and activation scales on64 expanded calibration frames. All128 retained
feature/position comparisons and64 direct/replay pairs are exact, finite, with zero
scheduler errors. The fixed sampling rule takes64 raster tokens/frame at every
block, recorded before selection. The first capture attempt failed only on the
snapshot shape of activation scales [5184,1]; its source/report remain, and v2
flattens that snapshot correctly. Successful capture took23.59 seconds.

A new independent integer-code trial fits at most32 greedy ±1 coordinate updates
per output channel against exact activation covariance on the fixed32 fitting
frames. It changes1,000,844 of155,189,248 FC2 codes, with maximum accumulated change
13 in any code. It preserves original masters/heads and all scales, smoothing and
biases bitwise. Predicted versus measured fitting improvement agrees within1.57e-6
relative. Mean local reconstruction MSE improves13.3% fitting/3.9% selection; final
feature MSE improves3.9%/1.4%. Original detector failures nevertheless change7→6
on fitting and15→20 on the independent32-frame selection set (confidence14→20,
boxes6→8). There are seven new selection failures and two removed failures.
**Reject the code candidate.** No development fitting/screen or promotion occurred.
Fitting plus full evaluation took58.13 seconds, peak allocation4,994,076,160 bytes.

`accuracy_code_summary.json` loads actual captures/codes/features, checks immutable
parameters and all192 original detector comparisons, and binds26 evidence hashes;
all26 hashes were independently reverified. Local same-input error decomposition
suggests activation rounding matters more than weight rounding for these FC2s:
activation-only and weight-only errors average.856/.273 of retained local MSE when
using the original rounded bias; restoring weights with the calibrated bias gives
.913. These ratios are not additive attribution or end-model sensitivity.
`accuracy_code_next.md` describes fixed block128 Hadamard rotation (4736=37×128)
after half rounding and smoothing, preserving a single row scale and FC2 DPAS.
That next harness is under development separately; the completed code trial does
not establish any Hadamard result. Do not repeat an unchanged greedy-budget sweep.


## 19. Compact positional bias: compiler rounding diagnosed and preserved

The retained decoder initially expands two [1,200,72,8] FP16 positional-bias axes
into a [1,8,201,5184] mask, including a zero presence-query row. The new isolated
`joint_head_attention_packed.cpp` reads compact [1,8,201,144] X/Y axes and reconstructs
dy+dx in registers, rounding to FP16 before adding to the FP32 attention scores.
It preserves the native 512-key split and Triton merge. Actual ISA has128 GRFs,
SIMD16 and no scratch or SLM allocation. The new library is
`runtime/joint_head_attention_packed512.so`, with unique kernel identity and full
build/source/ELF hashes. Image execution and master weights are unchanged.

`capture_packed_head_attention.py` reconstructs all18 actual eager masks exactly
on frames0/13/23 and saves the same six layer0/5 input cases. The standalone packed
partial passes12 finite, changed-replay, FP32-math and bitwise retained-native
comparisons on GPU0. Its five-round median loses .0041665 ms; the purpose is to
remove full mask construction, not accelerate the already-built-mask kernel.

The first full-head implementation saves .2971615 ms on GPU0 but changes outputs
and fails two incremental confidence/box cases against the retained head. Original
reference failures remain5/4, but this is insufficient for an execution-only
replacement. Preserve/reject `packed_heads_split512_gpu0.json`.
`diagnose_packed_heads.py` then captures every cross-attention boundary in two
failing development cases, with no fitting. Eager paths match every Q/K/V, mask,
attention output and final head output exactly. Compiled paths first diverge in
layer0's positional mask:2,199,900 values differ, maximum .25, while Q/K/V remain
exact. The later query differences are downstream of that first mask difference.

Actual generated code explains the difference. The retained expanded-mask path
uses a half MM result, then half bias addition and half dy+dx addition. Compact
transport initially selects oneDNN addmm for the axes, moving bias before the half
MM rounding. `_axis_mlp` in `packed_mask_rpb.py` now explicitly computes the half
matrix result, casts product and half-rounded bias to FP32 for their addition,
and rounds the sum back to half. This prevents the changed addmm lowering while
preserving the retained compiled semantics. The source change is not an accuracy
threshold adjustment. Eager arithmetic was diagnostic, not the retained compiled
reference; source snapshots document both revisions and generated programs.

`packed_mask_mha.py` preserves upstream MHA arithmetic with only packed-mask shape
validation and reshape changes. `packed_head_attention.py` binds that helper and
the compact RPB method in an isolated context. Original model files remain unchanged.
The initial helper copy and all later changes are preserved with original source
hashes; native ABI runtime assertions check actual Q/K/V and compact-mask layouts.

The corrected full-head runs complete all24 frames/72 prompts on both GPUs with
all three raw outputs bitwise equal to the previous retained native full-mask
head. All72 changed-input/direct-replay checks are exact and finite, and all
incremental gates pass. Both original-reference screens retain5 confidence/4 box
failures; dense diagnostics retain3/2. Five paired head rounds save .3016935 ms
on GPU0 and .4097135 ms on GPU1. No previous model failures are waived.

`benchmark_packed_engine.py` directly compares complete resident image-to-device-box
graphs, with the previous native full-mask head as the control:

| Combined engine-only timing | GPU0 | GPU1 |
|---|---:|---:|
| Previous native-head control, median of round medians | 85.340026 ms | 85.438698 ms |
| Corrected packed-bias candidate | 85.004271 ms | 84.980599 ms |
| Median paired saving | .332683 ms | .470756 ms |

All five rounds improve on each GPU. Each path has20 XPU-event samples/round;
all host work, input copies and cached-prompt setup remain outside the interval.
All24 frames preserve previous raw outputs, independent standalone-head results,
all decoded values, direct/replay and zero image scheduler errors. These are
combined engine measurements, not sums of isolated gains.

`packed_head_summary.json` audits71 artifacts, actual six-case masks, loaded
full-head outputs from both GPUs through the unchanged original XPU gates,
first-divergence tensors, exact corrected outputs, source/ISA and timing rounds.
All71 hashes were independently reverified. New source references bind immutable
snapshots. `engine_head_legacy_source_bindings.json` preserves the one old live
builder reference changed after its historical81-artifact audit; the old snapshot
and all old experimental outputs remain intact. Do not pretend edited live source
still has its earlier digest.

Retain the corrected compact-bias head for development. The current invocation
and artifact hashes are in `engine_only_retained_packed_config.json`, superseding
the prior recipe. Use `benchmark_packed_engine.py --device D --native-library
runtime/joint_head_attention_packed512.so --output UNIQUE.json` with the matching
GPU cache paths and OMP_NUM_THREADS=4. The independent original reference, image
configuration, all failure evidence and the full goal remain unchanged. This
fuses positional bias into two-kernel attention; it is not a full-model megakernel.

## 20. Cross-block native segment prerequisite and SFU evidence

`megakernel_design.md` gives a concrete next segment: queued MLP(i) → next norm1
and activation quantization → QKV/RoPE(i+1). Match MLP row order to the next QKV
window/raster order so each QKV row tile depends on exactly two FC2 producers.
The last producer publishes normalized INT8 rows/scales using the retained release/
acquire scheme. Pair two existing128-thread QKV tiles in each256-thread MLP worker;
retain matrix tiles, rounding and row grouping. A successful segment could remove
62 logical launches over31 transitions, while retaining required residual and
Q/K/V buffers. This design is not implemented as an integrated segment or claimed
to reach80%; CPU task/layout proofs do not establish GPU progress or memory order.

The implemented prerequisite recovers the actual retained Triton norm's symbolic
sum tree and emulates it with one SIMD16 subgroup/row, removing eight original
workgroup barrier instructions. It captures eight real input/affine/smoothing sets
from two frames and both norm positions in blocks0/7. Four native256-GRF variants
compile with no scratch/SLM but fail the first real FP16/raster case before timing:
correct-division variant differs in3 codes and scale by1.862645149e-9; default
norm division gives4 code differences with exact scales; explicit native final
division gives2; a register permutation still gives2. They are all rejected before
performance or model integration. This is not a completed eight-set parity suite.

An instrumented native diagnostic matches the instrumented-and-retained Triton
reference exactly at every exposed stage:5.3M normalized FP32 values, half casts,
smoothing quotients, final ratios, codes/scales and mean/variance/rsqrt statistics.
Additional stores change compiler behavior, so the uninstrumented arithmetic
boundary remains an optimization/code-generation investigation. Do not relax code
or scale equality, promote the instrumented probe, or repeat unchanged variants.
`megakernel_design_summary.json` audits120 artifacts including loaded real inputs,
source versions, all four failures, the complete intermediate diagnostic, actual
ISA and the CPU symbolic inventory. All120 hashes were independently reverified.
GPU0 jobs are terminal and no norm implementation was promoted.

The SFU source investigation found no defensible public Xe2 EXP/RSQRT/add/shuffle/
barrier numeric ceiling. Intel IGC latency tables are explicitly heuristics and
retain an old native-width TODO, so their estimates do not enter the denominator.
Actual retained decoder-partial ISA contains32 SIMD16 and8 scalar EXP sites per
key-tile loop, yielding3,234,816 SIMD16 instructions plus808,704 scalar instructions
across six layers. `engine_roofline_sfu_evidence_v1.json` binds the counts to actual
code;19 hashes and a byte-identical rerun are verified. `engine_roofline_sfu_probe_plan_v1.md`
specifies exact register recurrences, independent-chain/occupancy/iteration sweeps,
scalar versus SIMD16 ISA, clocks, numerical/replay checks, co-issue and barrier
controls. These probes are proposed, not executed. A demonstrated rate would be
a lower bound on capability, not automatically an upper bound for acceptance.
Eight earlier mutable inventory-source paths now have immutable matching snapshot
bindings in `engine_roofline_sfu_inventory_bindings_v1.json`.

## 21. Fixed unsigned Hadamard128 local diagnostic

`accuracy_hadamard.py` tests the fixed natural-order unsigned block128 Sylvester
transform on frozen actual FC2 calibration operands, after FP16 rounding and FP32
smoothing. It uses zH and UH/128, regenerates derived weight codes/scales and keeps
one activation scale/row. It performs no fitting, sign/permutation search, model
integration or detector screening. CPU checks establish exact orthogonality and
dyadic inverses; general FP32 inverse RMS is1.01e-7 and FP64 no-quantized product
error2.11e-15. The initial GPU-run invocation terminated before any XPU operation
on relative/absolute artifact-path comparison; v2 resolves paths without numerical
changes. Successful local run took15.44 seconds, peak allocation409,399,808 bytes.

The fixed transform improves28/32 layers in both fitting/selection partitions.
Mean per-layer MSE ratios are .586/.528; ratios of aggregate mean MSE are .520/.454.
Blocks0–3 regress, including block1 at2.79×/2.62×. XPU unquantized transformed
products agree with independent FP64 within2e-6 relative RMS;128 integer-dot probes
match independent CPU INT64 exactly. The baseline uses actual captured native
quantization, whose eager reconstruction differs by15/11 activation codes and
about2.1e-8 relative scale RMS; those reconstruction differences are disclosed,
not silently substituted for the native baseline.

`accuracy_hadamard_summary.json` loads actual derived buffers and verifies16 hashes,
all64 local cases and exact independent reproduction of1,212,416 checked transformed
weight codes. All16 hashes were independently reverified. Derived buffers in
`accuracy_hadamard128_local_gpu1_v2.pt` require matching activation rotation and
must not be installed in the unchanged inference path. No detector improvement or
latency result is established. A single fixed randomized-sign follow-up is running
as a separate diagnostic to test whether unsigned H's DC concentration explains
early-layer regressions; it has not changed the retained image implementation.
