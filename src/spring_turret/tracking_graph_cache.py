"""Bounded, non-recompiling misses for the pinned native tracking stages.

Warm a bounded cache, then replay exact cache hits. New object/memory shapes
execute the same region directly, using valid compiled kernels but not blocking
the live session on recompilation/capture. Outputs remain owned by SAM.
This policy does not alter the frozen model, weights, memory or object selection.
"""


class _NonblockingCache:
    def __call__(self, *args, **kwargs):
        value = (args, kwargs)
        key = self.signature(value)
        state = self.variants.get(key)
        if state is None:
            if self.calls < self.capture_window:
                self.warmup_signatures[key] = self.warmup_signatures.get(key, 0) + 1
                if (len(self.variants) < self.capture_capacity and
                        self.warmup_signatures[key] >= self.capture_repetitions):
                    # Initial warmup retains pinned capture/numerical checks.
                    return super().__call__(*args, **kwargs)
            else:
                self.warmup_signatures.clear()
            with self.torch.compiler.set_stance("eager_on_recompile"):
                outputs = self.invoke(value)
            self.direct_calls += 1
        else:
            inputs, graph, outputs, _ = state
            sources, _ = self.tree.tree_flatten(value)
            targets, _ = self.tree.tree_flatten(inputs)
            for source, target in zip(sources, targets, strict=True):
                if isinstance(source, self.torch.Tensor):
                    target.copy_(source)
            graph.replay()
            self.replay_calls += 1
        self.calls += 1
        return self.clone(outputs)


def install_nonblocking_cache(stages):
    """Wrap existing instances, preserving the native observer/invoke hooks."""
    classes = {}
    for name, stage in stages.items():
        if isinstance(stage, _NonblockingCache):
            continue
        base = type(stage)
        if base not in classes:
            classes[base] = type("Nonblocking" + base.__name__, (_NonblockingCache, base), {})
        stage.direct_calls = stage.replay_calls = 0
        # Short, growing memory signatures are not worth a graph. During a
        # bounded startup window retain only repeatedly-used temporal shapes.
        # Freeze afterward: unseen objects/memories cannot recapture mid-demo.
        temporal = name in ("memory_attention_and_mask", "memory_update", "mask_input_encoding")
        stage.capture_window = 64 if temporal else 1
        stage.capture_repetitions = 3 if temporal else 1
        stage.capture_capacity = stage.max_variants if temporal else 1
        stage.warmup_signatures = {}
        stage.__class__ = classes[base]


def finish_cache_warmup(stages):
    """Close even rarely-used stages after the worker's initial frame window."""
    for stage in stages.values():
        stage.capture_window = 0
        stage.warmup_signatures.clear()
