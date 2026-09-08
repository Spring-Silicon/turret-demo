"""Fixed-shape compiled GPU stages with explicit, validated graph replay."""

from typing import Any


class CompiledStage:
    def __init__(self, torch: Any, module: Any, name: str, progress: Any):
        self.torch, self.name, self.progress = torch, name, progress
        # Infer the accelerator from the first call, including modules with no
        # parameters. CUDA and XPU graphs have the same owned-buffer contract.
        self.module = module
        self.compiled = None
        self.graph = None
        self.calls = 0

    def _compile(self, inputs):
        torch = self.torch
        kinds = {value.device.type for value in inputs if value.device.type != "cpu"}
        if len(kinds) != 1 or not kinds <= {"xpu", "cuda"}:
            raise ValueError(f"{self.name}: expected inputs on one CUDA or XPU backend")
        self.device_type = kinds.pop()
        self.runtime = getattr(torch, self.device_type)
        options = {"emulate_precision_casts": True}
        if self.device_type == "cuda":
            # We own capture and replay; do not nest Inductor's graph trees.
            options["triton.cudagraphs"] = False
        self.compiled = torch.compile(
            self.module,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            options=options,
        )

    def __call__(self, *inputs):
        torch = self.torch
        if self.graph is None:
            if self.compiled is None:
                self._compile(inputs)
            runtime = self.runtime
            self.progress("compiling", self.name)
            # Warm the exact inference tensors/stream captured below; different
            # tensor guards would cause a forbidden Dynamo retrace in capture.
            self.inputs = tuple(value.clone() for value in inputs)
            self.stream = runtime.Stream()
            runtime.synchronize()
            with runtime.stream(self.stream):
                for _ in range(2):
                    outputs = self.compiled(*self.inputs)
            runtime.synchronize()
            expected = tuple(value.clone() for value in outputs)
            self.progress("capturing", self.name)
            graph = runtime.CUDAGraph() if self.device_type == "cuda" else runtime.XPUGraph()
            with runtime.graph(graph, stream=self.stream):
                self.outputs = self.compiled(*self.inputs)
            graph.replay()
            runtime.synchronize()
            for reference, actual in zip(expected, self.outputs, strict=True):
                torch.testing.assert_close(actual, reference, rtol=0.001, atol=0.001)
            # Never mark a failed/partially captured graph usable.
            self.graph = graph
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
