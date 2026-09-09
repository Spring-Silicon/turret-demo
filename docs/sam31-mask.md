# SAM 3.1 mask

Choose **SAM3.1 mask** (`sam3.1-mask`) to detect instances and produce a mask for
each one on every frame. This is independent of `sam3.1-tracking`: it does not
maintain SAM temporal memory. The other profiles are box-only SAM and Tracking;
YOLO is removed. See [shared target/lifecycle policy](sam-policy-audit.md).
Shared controls select the profile on both devices; individual
views can select it separately.

## Implementations

- **Arc B580:** frozen sleepy-joe `masks/w8a8/skip4_attention8-v0` extension
  over the `masks/tiled-qualification-gpu1` base. It removes local image blocks
  24, 26, 28 and 30 and uses native W8A8 QKV/output projections in the other 28
  blocks. All four global-attention blocks remain. The selected W4A4 MLPs,
  token routing, floating attention, grounding, segmentation weights and tiled
  GroupNorm remain as before.
  `torch.compile` regions execute inside one explicit SYCL graph per prompt count.
- **Thor:** original floating-point SAM detector and segmentation weights,
  FP16 autocast, `torch.compile(fullgraph=True, dynamic=False)` and one explicit
  CUDA graph. No Intel native binaries, W4A4 substitutions or tiled GroupNorm.
- Both use checkpoint SHA256
  `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`.
  They share the architecture and output contract, not identical internal
  numerical precision.

Each replay runs the image encoder once, produces all three FPN scales, then
grounding and segmentation for each prompt, followed by fixed-shape output
selection. Text embeddings are cached outside the per-frame graph. A changed
prompt reuses the graph; changing the number of prompts captures a replacement.
CPU JPEG handling, mask PNG encoding, networking and servo control remain outside
the model graph. The viewer displays the paired, processed frame and its masks.

## Output and limits

All 200 object queries are evaluated. For each prompt, retain the highest-scoring
**10** instances above the configured confidence threshold (default strictly
greater than 0.5). Ties use query order. Output masks are bilinearly resized to
1008×1008 before sigmoid/threshold, then nearest-neighbor mapped to the source
frame for display. This matches the source qualification contract.

`mask_capacity_per_prompt` and `mask_overflow` explicitly report the cap and
omitted instances; the UI warns when the cap is exceeded. Instance colors vary
within their category's base color. These per-frame query slots are not stable
temporal identities. Each output includes the original mask centroid for the
control API. This profile does not change servo calibration, angle limits or
arming.

## Evidence and tradeoff

The newest source recipe's paired B580 measurement was **62.807 ms** GPU
image-to-mask replay versus **70.755 ms** for the previous native mask model.
It won 60/60 timing pairs. On the same 512 COCO person confirmation images, mask
AP was **48.5453 versus 48.6869**: an additional **0.1416 percentage-point loss**.
Those confirmation images were also used during candidate selection; this is not
independent-data accuracy qualification or a guarantee for other prompts. The
measurements exclude CPU preparation, transfers and display.

The earlier native model already had a 0.3789-point AP loss versus its
floating-image control (49.0658 AP). That control retains native grounding, so
neither comparison is an unmodified full dense-model baseline.

The Arc bundle validates its original and relocated manifests, checkpoint and
measured Torch build before loading. Startup compares compiled direct execution
against replay (exact discrete output; 0.001 absolute/relative floating tolerance).
Host qualification also alternates images and changes prompt values/counts.
See the [deployment checks and measured timings](sam31-mask-qualification.md).

## Configuration

Arc needs `inference.sam31_mask_bundle` pointing at the frozen bundle. Thor uses
`inference.device_type: "cuda"` and the existing checkpoint/SAM installation;
an Intel bundle is deliberately rejected on CUDA. Select `inference.model:
"sam3.1-mask"` for the startup default, or switch through the API/UI.

Compilation caches are separate under `sam31-mask-20260908`, leaving existing
profiles' caches and weights untouched. A service restart never re-arms motors.

The collector `tools/collect-sam31-mask.py` runs on sleepy-joe and freezes the
recorded result into a new temporary directory without modifying its campaign.
`tools/relocate-sam31-mask.py BUNDLE` builds a relocated copy with prebuilt,
hash-checked libraries in place of build-on-import operations. Source and runtime
manifest hashes must match `sam31_mask_native.py`; a changed upstream result
requires deliberate requalification, not silently accepting a new manifest.

`tools/collect-sam31-mask-attention8.py` collects the newer selected extension
from sleepy-joe, preserving executed sources, selection/qualification receipts,
calibration and the exact compiled DSO. Only import paths and build-on-import
are relocated. `sam31_mask_attention8.py` pins its complete manifest and checks
28 retained blocks / 56 converted projections. The quantization helper was not
included in upstream's run snapshot: its current committed source is separately
frozen and identified as supplementary provenance.

The extension is installed as `attention8/` inside the original base bundle;
original base manifest contents are not overwritten. Without that extension,
the same loader retains the previous 32-block path. This update does not change
the separate box-only or temporal Tracking profiles, or the Thor CUDA model.
The worker exposes `image_optimization` and the exact recipe in status responses.
