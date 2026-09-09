"""Reusable pinned readback, preserving every output bit and tensor shape."""


class OutputTransfer:
    def __init__(self, torch, runtime):
        self.torch, self.runtime = torch, runtime
        self.buffers = None
        self.signature = None

    def __call__(self, outputs):
        signature = tuple((tuple(v.shape), v.dtype, v.device) for v in outputs)
        if signature != self.signature:
            self.buffers = [self.torch.empty(v.shape, dtype=v.dtype, device='cpu', pin_memory=True)
                            for v in outputs]
            self.signature = signature
        # Enqueue every copy before waiting once. Buffers are borrowed until the
        # next call; workers finish CPU postprocessing before another inference.
        for target, source in zip(self.buffers, outputs, strict=True):
            target.copy_(source, non_blocking=True)
        self.runtime.current_stream().synchronize()
        return [v.numpy() for v in self.buffers]
