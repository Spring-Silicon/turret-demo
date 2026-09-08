"""Fixed-shape compiled XPU stages with explicit, validated SYCL replay."""

from typing import Any


class CompiledStage:
    def __init__(self, torch: Any, module: Any, name: str, progress: Any):
        self.torch, self.name, self.progress = torch, name, progress
        self.compiled = torch.compile(
            module,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            options={"emulate_precision_casts": True},
        )
        self.graph = None
        self.calls = 0

    def __call__(self, *inputs):
        torch = self.torch
        if self.graph is None:
            self.progress("compiling", self.name)
            # Warm the exact inference tensors/stream captured below; different
            # tensor guards would cause a forbidden Dynamo retrace in capture.
            self.inputs = tuple(value.clone() for value in inputs)
            self.stream = torch.xpu.Stream()
            torch.xpu.synchronize()
            with torch.xpu.stream(self.stream):
                for _ in range(2):
                    outputs = self.compiled(*self.inputs)
            torch.xpu.synchronize()
            expected = tuple(value.clone() for value in outputs)
            self.progress("capturing", self.name)
            self.graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(self.graph, stream=self.stream):
                self.outputs = self.compiled(*self.inputs)
            self.graph.replay()
            torch.xpu.synchronize()
            for reference, actual in zip(expected, self.outputs, strict=True):
                torch.testing.assert_close(actual, reference, rtol=0.001, atol=0.001)
        else:
            if len(inputs) != len(self.inputs):
                raise ValueError(f"{self.name}: wrong number of graph inputs")
            for source, target in zip(inputs, self.inputs, strict=True):
                if source.shape != target.shape or source.dtype != target.dtype:
                    raise ValueError(f"{self.name}: graph input shape/dtype changed")
                target.copy_(source)
            self.graph.replay()
        self.calls += 1
        # Borrowed output buffers: clone before caching across later replays.
        return self.outputs
