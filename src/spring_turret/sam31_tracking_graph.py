"""Owned-buffer graph replay of SAM's tensor stages, not its mutable sessions.

Each exact input signature has a bounded graph cache. Outputs are cloned before
returning: SAM keeps features and mask memories across frames, so returning a
graph's borrowed buffers would silently corrupt temporal state on the next call.
"""
from collections import OrderedDict
import sys


class TrackingGraphStage:
    def __init__(self, torch, fn, name, device_type, *, max_variants=2, progress=None, backend="inductor", composed=False,
                 cache_policy="lru", capture_repetitions=1):
        from torch.utils import _pytree
        self.torch, self.tree = torch, _pytree
        self.fn, self.name, self.device_type = fn, name, device_type
        self.backend = backend
        self.composed = composed
        # A composed detector includes several BF16 attention stages. Its
        # accumulated replay error is still bounded; temporal output/ID parity
        # is qualified separately before release, not inferred from this gate.
        self.replay_rtol = .02 if composed else .01
        self.runtime = getattr(torch, device_type)
        options = {"emulate_precision_casts": True}
        if device_type == "cuda":
            options["triton.cudagraphs"] = False  # Explicit capture owns buffers.
        # A region already contains compiled modules. Capture their launches
        # together without a second compiler changing their numerical kernels.
        self.compiled = fn if composed else torch.compile(fn, backend=backend, fullgraph=True,
                                      dynamic=False, **({"options":options} if backend == "inductor" else {}))
        self.variants = OrderedDict()
        if cache_policy not in ("lru", "retain") or max_variants < 1 or capture_repetitions < 1:
            raise ValueError("Invalid tracking graph cache policy")
        self.max_variants = max_variants
        self.cache_policy, self.capture_repetitions = cache_policy, capture_repetitions
        self.admission_counts = OrderedDict()
        self.calls = self.captures = 0
        self.cache_misses = self.evictions = self.replay_calls = self.direct_calls = 0
        self.progress = progress
        self.max_replay_nrmse = 0.

    def signature(self, value):
        torch = self.torch
        leaves, spec = self.tree.tree_flatten(value)
        signature = []
        for leaf in leaves:
            if isinstance(leaf, torch.Tensor):
                if leaf.device.type != self.device_type:
                    raise ValueError(f"{self.name}: graph inputs must be GPU tensors")
                signature.append(("tensor", tuple(leaf.shape), tuple(leaf.stride()),
                                  leaf.dtype, leaf.device))
            elif leaf is None or type(leaf) in (bool, int, float, str):
                signature.append((type(leaf), leaf))
            else:
                raise TypeError(f"{self.name}: unsupported graph input {type(leaf)}")
        return (spec, tuple(signature))

    def clone(self, value):
        owned = {}
        def copy(t):
            if id(t) not in owned:
                owned[id(t)] = t.clone()
            return owned[id(t)]
        return self.tree.tree_map_only(self.torch.Tensor, copy, value)

    def invoke(self, inputs, fn=None):
        # SAM's fusion encoder replaces entries in src/src_pos lists with
        # reshaped views. Rebuild containers for every warmup/capture so the
        # next invocation still sees the original shapes and owned buffers.
        args, kwargs = self.tree.tree_map_only(self.torch.Tensor, lambda t: t, inputs)
        return (self.compiled if fn is None else fn)(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        value = (args, kwargs)
        key = self.signature(value)
        state = self.variants.pop(key, None)
        if state is None:
            self.cache_misses += 1
            if self.cache_policy == "retain":
                # Growing or rarely-used temporal shapes do not deserve a
                # capture. Never evict a resident graph to recapture it later.
                count = self.admission_counts.pop(key, 0) + 1
                if len(self.variants) < self.max_variants:
                    self.admission_counts[key] = count
                    if len(self.admission_counts) > 128:
                        self.admission_counts.popitem(last=False)
                if len(self.variants) >= self.max_variants or count < self.capture_repetitions:
                    # A cache miss is still the identical full model operation.
                    # Keep the same Inductor numerical path, including on
                    # cold shapes. Dynamo may compile a genuinely new shape
                    # once; an evicted graph must not add repeated captures.
                    outputs = self.invoke(value)
                    self.calls += 1
                    self.direct_calls += 1
                    return self.clone(outputs)
            # Evict BEFORE capture, bounding GPU ownership even at peak.
            if len(self.variants) >= self.max_variants:
                self.runtime.synchronize()
                self.variants.popitem(last=False)
                self.evictions += 1
            print(f"Tracking compile/capture: {self.name}", file=sys.stderr, flush=True)
            if self.progress is not None:
                self.progress("compiling_tracker_" + self.name)
            inputs = self.clone(value)
            stream = self.runtime.Stream()
            self.runtime.synchronize()
            with self.runtime.stream(stream):
                # Populate SAM's lazy coordinate/position caches before Dynamo
                # tracing (notably decoder boxRPB on the very first frame).
                self.invoke(self.clone(inputs), self.fn)
                for _ in range(2):
                    expected = self.invoke(inputs)
                expected = self.clone(expected)
            self.runtime.synchronize()
            graph = (self.runtime.CUDAGraph() if self.device_type == "cuda"
                     else self.runtime.XPUGraph())
            with self.runtime.graph(graph, stream=stream):
                outputs = self.invoke(inputs)
            graph.replay()
            self.runtime.synchronize()
            # Capture is not considered successful until actual replay agrees
            # with a direct call. Eager/compiled accuracy is qualified separately.
            actual_leaves, actual_spec = self.tree.tree_flatten(outputs)
            expected_leaves, expected_spec = self.tree.tree_flatten(expected)
            if actual_spec != expected_spec:
                raise RuntimeError(f"{self.name}: graph output structure changed")
            for actual, reference in zip(actual_leaves, expected_leaves, strict=True):
                if isinstance(actual, self.torch.Tensor) and actual.is_floating_point():
                    # BF16 attention can differ between direct launches and
                    # replay. Judge tensor-scale error, not relative error at
                    # near-zero elements; final boxes/IDs are qualified too.
                    if not self.torch.isfinite(actual).all() or not self.torch.isfinite(reference).all():
                        raise RuntimeError(f"{self.name}: nonfinite graph output")
                    if actual.numel() and max(actual.abs().max().item(), reference.abs().max().item()) > 1e20:
                        # Additive attention masks use finfo.min. Squaring that
                        # would overflow and invalidate the relative-error test.
                        self.torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                        continue
                    # Normalize before squaring to avoid float32 overflow for
                    # large (but finite) text attention-mask values.
                    divisor = max(reference.abs().max().item(), actual.abs().max().item(), 1.) if actual.numel() else 1.
                    a, r = actual.float() / divisor, reference.float() / divisor
                    error = (a-r).square().mean().sqrt().item() * divisor if actual.numel() else 0.
                    scale = r.square().mean().sqrt().item() * divisor if reference.numel() else 0.
                    normalized = error / max(scale, .1)
                    self.max_replay_nrmse = max(self.max_replay_nrmse, normalized)
                    print(f"Replay check {self.name}: rmse={error:.6g} rms={scale:.6g} nrmse={normalized:.6g}", file=sys.stderr, flush=True)
                    if error > .001 + self.replay_rtol * scale:
                        raise RuntimeError(f"{self.name}: excessive replay error ({normalized:.4%} normalized RMS)")
                else:
                    self.torch.testing.assert_close(actual, reference, rtol=.001, atol=.001)
            state = (inputs, graph, outputs, stream)
            self.captures += 1
            self.admission_counts.pop(key, None)
            if len(self.variants) + 1 >= self.max_variants:
                self.admission_counts.clear()
        else:
            inputs, graph, outputs, stream = state
            sources, _ = self.tree.tree_flatten(value)
            targets, _ = self.tree.tree_flatten(inputs)
            for source, target in zip(sources, targets, strict=True):
                if isinstance(source, self.torch.Tensor):
                    target.copy_(source)
            graph.replay()
            self.replay_calls += 1
        self.variants[key] = state
        self.calls += 1
        return self.clone(state[2])


def install_tracking_graphs(torch, model, device_type, *, progress=None, backend="hybrid", layout="stages"):
    """Same tensor boundaries as SAM3.1's upstream _compile_model.

No quantization, memory-policy changes, frame skipping, mask omission or
whole-session capture. CPU association and variable Python bookkeeping stay
eager. Exact shape specializations do not pad/drop objects or truncate memory.
The legacy stage layout's hybrid backend retains native ATen heads. The current
regional CUDA path uses the separately qualified unquantized Inductor heads;
the XPU regional policy is unchanged. Text remains cached outside frame replay.
"""
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.accumulated_cache_size_limit = 2048
    if layout == "regions":
        try:
            from .sam31_tracking_regions import install_tracking_regions
        except ImportError:
            from sam31_tracking_regions import install_tracking_regions
        return install_tracking_regions(torch, model, device_type, progress=progress)
    if layout != "stages":
        raise ValueError("Unknown tracking graph layout")
    stages = {}
    modules = {
        "text_encoder": model.detector.backbone.language_backbone.encoder,
        "image_encoder": model.detector.backbone.vision_backbone,
        "detection_encoder": model.detector.transformer.encoder,
        "detection_decoder": model.detector.transformer.decoder,
        "detection_masks": model.detector.segmentation_head,
        "memory_encoder": model.tracker.maskmem_backbone,
        "memory_attention": model.tracker.transformer.encoder,
        "tracking_masks": model.tracker.sam_mask_decoder,
    }
    for name, module in modules.items():
        selected_backend = ("inductor" if name == "image_encoder" else "aot_eager") if backend == "hybrid" else backend
        cache = {"max_variants":16, "cache_policy":"retain", "capture_repetitions":3} if device_type == "cuda" and name == "memory_attention" else {}
        stage = TrackingGraphStage(torch, module.forward, name, device_type, progress=progress, backend=selected_backend, **cache)
        module.forward = stage
        stages[name] = stage
    return stages


def tracking_graph_status(stages, device_type):
    active = any(s.calls for s in stages.values())
    return {"torch_compile": active, "cuda_graph": active and device_type == "cuda",
            "sycl_graph": active and device_type == "xpu",
            "compilation_scope":"tensor-regions" if any(getattr(s, "composed", False) for s in stages.values()) else "tensor-stages",
            "graph_stages": {name:{"calls":stage.calls, "captures":stage.captures,
                                   "cached_shapes":len(stage.variants), "compiler_backend":stage.backend,
                                   "cache_policy":getattr(stage, "cache_policy", "lru"),
                                   "cache_capacity":stage.max_variants,
                                   "cache_misses":getattr(stage, "cache_misses", 0),
                                   "evictions":getattr(stage, "evictions", 0),
                                   "replay_calls":getattr(stage, "replay_calls", 0),
                                   "direct_calls":getattr(stage, "direct_calls", 0),
                                   "replay_rtol":stage.replay_rtol,
                                   "max_replay_nrmse":stage.max_replay_nrmse}
                             for name,stage in stages.items()}}
