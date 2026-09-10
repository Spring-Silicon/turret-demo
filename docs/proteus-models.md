# Proteus experimental profiles

These are additional options, not replacements for the established SAM models.
They reproduce four experimental neural paths from `spring@proteus` and do not
claim that the source basketball results establish usable tracking accuracy.

| Menu option on Arc | Proteus source in `basketball-demo-6070826` | Thor pairing | Neural memory |
| --- | --- | --- | --- |
| Efficient NoMem | `efficientsam3_v04_nomux_e2e.py` | SAM NoMem | Off |
| Hybrid NoMem | `basketball_tracker_es31vision_meta_text.py` | SAM NoMem | Off |
| Efficient Tracking | `efficient_sam31_oob_tracker.py` | SAM 3.1 Tracking | On |
| Efficient Memory | `efficientsam3_v04_memory_e2e.py` | SAM 3.1 Tracking | On, seven slots, no Multiplex |

The shared dropdown applies the pairing; standalone Arc and Thor pages retain
their own hardware-compatible choices. NoMem Arc profiles accept one text
prompt and one seeded object, as does Efficient Memory. Efficient Tracking retains the shared multi-prompt
session interface. Selecting a model does not arm motors.

## Execution and limitations

- Efficient NoMem: official v0.4 TinyViT/MobileCLIP text initializer, then the
  local nonmultiplex singleton tracker. TinyViT uses 40 native W8A8 linear
  kernels, six fused attention preludes, five deconvolution replacements, and
  XPU graph replay. The decoder remains eager. Source test: mask lost after seed.
- Hybrid NoMem: EfficientSAM3.1 TinyViT with Meta SAM3.1 neck/tracker/text,
  `hybrid_w8a8_nomem`. Same backbone optimization recipe. The source initializer
  deliberately accepts scores down to zero; its poor seed/mask behavior is
  disclosed, not disguised with a fabricated confidence score.
- Efficient Tracking: original v0.3 TinyViT-M/MobileCLIP-S0 context-16 video
  checkpoint with seven memory slots and the original eager neural execution.
  Input normalization alone is compiled. Source test: no basketball detections.
- Efficient Memory: v0.4 text acquisition followed by the standard memory-on
  tracker, without Object Multiplex. Uses seven memory slots, 309 tracker tensors,
  and the corrected 22 tracker-neck tensors from the donor checkpoint. Same
  W8A8 TinyViT/backbone graph recipe as Efficient NoMem; memory attention,
  memory encoder and decoder remain eager, not one compiled end-to-end graph.
  Proteus reports 58.6 ms median tracking-only and masks on 219 basketball frames;
  acquisition/rendering are excluded. This is not live demo FPS or broad accuracy
  validation, and the saved receipt does not pin the entire subsequent dirty source.
- Thor SAM NoMem is an alias for the existing compiled image-mask worker:
  image backbone, text grounding and mask head, with no temporal tracker or
  memory bank. It retains CUDA/Inductor and CUDA graphs. This is a memory-policy
  comparison, not an identical TinyViT architecture or a new tracker ablation.

Live adaptations: current-frame-only input, bounded history, prompt/camera/shape
reset handling, native IDs, and shared missing-track retirement. NoMem retries
text acquisition after retirement; it does not propagate an offline seed forever.
The two NoMem initializers live in separate child processes because their SAM
packages differ from the propagation packages. Empty masks are never reused on
a newer camera frame. Initialization/compilation time is not steady latency.
The source resize kernel only supports upscaling. Camera inputs larger than
1008 in either dimension use exact Pillow bilinear resize and the same FP32
normalization before XPU upload. They are not silently approximated or cropped.
Memory sessions retain bounded history and keep pointer-time normalization stable
from the first live frame. Encoded memory is checked before publishing results.

## Arc runtime layout

Set `inference.proteus_bundle` to an absolute directory. The deployed directory
is `/home/spring/.local/share/turret-demo/proteus-20260910` on
`spring-edge-turret`. Copy these paths from `/home/spring` on Proteus, preserving
relative paths and symlinks (`rsync -aH --relative`; exclude `.git` and
`__pycache__`; no deletion of existing runtime trees):

```text
proteus-composed-b580-20260907/private-host
proteus-composed-b580-20260907/root/env/venv
proteus-composed-b580-20260907/root/native-matched-2025.3.2/extracted/opt/intel/oneapi/compiler/2025.3
sam31-b580-lab/env/site-packages
sam31-b580-lab/env/include
sam31-speed-memoff-20260910/sam31-source
sam31-speed-memoff-20260910/adapters
sam31-speed-memoff-20260910/efficientsam3
sam31-speed-memoff-20260910/assets/efficient_sam3_tinyvit_m.pt
sam31-speed-memoff-20260910/assets/frame.jpg
sam31-speed-memoff-20260910/cache/native/onednn-3.11.2-build/src
efficient-sam3-nomux-25ms-20260910/sam3
basketball-demo-6070826/efficient-sam31-oob-20260910
basketball-demo-6070826/agent-efficient-release/efficientsam3_tinyvit.pt
basketball-demo-6070826/basketball_tracker_es31vision_meta_text.py
basketball-demo-6070826/es31vision_meta_text_image_seed.py
```

Within the bundle, link `sam31-speed-memoff-20260910/sam3` to `sam31-source`.
Link its `assets/sam3.1_multiplex.pt` to the existing configured Meta checkpoint.
Copy `basketball-demo-6070826/frames/000001.jpg` as
`efficient-nomem-calibration.jpg` at the bundle root. This retains the original
quantization calibration instead of calibrating on whichever object is live.

`proteus_runtime.py` uses the copied private ELF loader/Python, Torch 2.12.1+XPU,
oneAPI 2025.3.2, and explicit matching OpenCL ICD. Existing demo Python, graphics
drivers and optimized SAM bundles are not upgraded or replaced. The bundle is
about 16.8 GB logical; provision additional room for native compilation caches.

Verified checkpoint SHA-256:

```text
3a52c42f975a9562cb656a57aebb60314d7871c3956e4955c91364a8fe3875dc  efficientsam3_tinyvit.pt
f4a044d87545a114f875e62929a88414966d931ec02377e8d9e65ba57a7f94c3  efficient_sam3_tinyvit_m.pt
4420078e557d7e707072b73da75a9ed28e8650d4cd1578d6ea21b9fb1781e428  efficient_sam3p1_tinyvit_m_mobileclip_s0_ctx16.pt
0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6  sam3.1_multiplex.pt
d3288da5aec8044ec30f184923214746480d2cab779e386ad9ebdcbee82d943b  efficient-nomem-calibration.jpg
```

The source NoMem builder changed during transfer. The refreshed, loading-tested
`sam3/model_builder.py` digest is
`b7955e0ffda2ed061a4a0e5345666944c820750f48fdb16e9d4b9d5e87cffc56`.
It obtains tracker-neck tensors from the tracker donor; the earlier snapshot
incorrectly requested them from the image-only checkpoint and could not load.
Hybrid construction/initializer statements are extracted from hash-pinned
scripts by `proteus_models.py`; their admission, offline CLI and render code are
never executed on the demo host.

Efficient Memory additionally uses `inference.proteus_memory_bundle`, currently
`/home/spring/.local/share/turret-demo/proteus-memory-v04-20260910.MZWTyo`.
This separate snapshot contains `efficient-sam3-nomux-25ms-20260910/sam3`, the
memory benchmark sources and its saved receipt. Its builder has the SHA-256 above,
which the worker checks before import. It reuses the base bundle's runtime,
front/tracker checkpoints and calibration image; there is no duplicate 16 GB
runtime. The admitted runner is retained as provenance only and is never launched
by the demo. For a smoke test, pass `--memory-bundle <snapshot>` alongside
`--profile efficient-memory`.

## Deployment and short checks

Install the package's model registry, hardware adapter, detection controller and
new worker modules. Update the on-device frontend assets as well as backend
assets: the frontend caches assets at startup and needs a restart. Keep existing
camera configuration, servo bindings, calibration and model bundles intact.

Thor's model-menu layer is `spring-turret-demo:proteus-menu-20260910`, based
on its actual running `0.17.3-thor-inductor-20260909` image. The current startup
image adds the two-module graph-cache fix as
`spring-turret-demo:thor5-retained-graphs-20260910`. Its service uses
`/opt/spring/turret-demo/run-container-overhead.sh`, not the older unused
`run-container.sh`. The previous launcher/image remain available for rollback.
Future builds from this repository use the normal installation/Docker workflow;
do not replace newer Thor optimizations with an older base image.

For a short on-device check, `tools/proteus-smoke.py --config <config.json>
--profile <model-id> --bundle <bundle> --frames 3 --pause-live` tests startup,
current-frame output and session reset. It restores unchanged original prompts
and never opens the servo device. Supply `--image <jpeg> --prompt basketball`
to exercise known source initialization. Smoke success means the integration
runs and returns valid output, not that the experimental model is accurate.
