#!/usr/bin/env python3
"""CPU-only CUDA routing and graph buffer ownership regression tests."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spring_turret.detection import WorkerClient, validate_config
from spring_turret.sam31_graph import CompiledStage
from spring_turret.sam31_worker import Sam31Engine


class Tensor:
    def __init__(self, value=1, device="cuda"):
        self.value, self.device = value, SimpleNamespace(type=device)
        self.shape, self.dtype = (1,), "float32"

    def clone(self):
        return Tensor(self.value, self.device.type)

    def copy_(self, other):
        self.value = other.value


class BackendTests(unittest.TestCase):
    def test_explicit_cuda_config_and_worker_arguments(self):
        with tempfile.TemporaryDirectory() as cache:
            config = dict(enabled=True, python="/usr/bin/python", checkpoint="/weights",
                          cache_dir=cache, device_type="cuda")
            validate_config(config)
            for extra in ({"device_type":"cpu"}, {"sam31_native_bundle":"/intel"},
                          {"sam31_w8a8_development_bundle":"/intel"},
                          {"yolo26x_checkpoint":"/weights/yolo"}):
                with self.assertRaises(ValueError):
                    validate_config({**config, **extra})
            with patch("spring_turret.detection.subprocess.Popen") as launch:
                WorkerClient(config).launch()
                argv = launch.call_args.args[0]
                self.assertEqual(argv[argv.index("--device-type")+1], "cuda")
                self.assertIn("sam31-cuda", launch.call_args.kwargs["env"]["TORCHINDUCTOR_CACHE_DIR"])
                self.assertNotIn("--allow-unqualified-w8a8", argv)
            for kind in ("native_bundle", "w8a8_development_bundle"):
                with self.assertRaisesRegex(ValueError, "require XPU"):
                    Sam31Engine(Path("/weights"), 0, "float16", .5, True,
                                device_type="cuda", **{kind:Path("/intel")})

    def test_cuda_and_xpu_capture_copy_and_replay(self):
        for kind in ("cuda", "xpu"):
            with self.subTest(kind=kind):
                events, captures = [], []
                class Graph:
                    def replay(self):
                        stage.outputs[0].value = stage.inputs[0].value * 2
                        events.append("replay")
                def compiled(value):
                    return (Tensor(value.value * 2, kind),)
                runtime = SimpleNamespace(Stream=object, synchronize=lambda: None,
                    stream=lambda stream:nullcontext(), graph=lambda graph, stream:nullcontext(),
                    **{("CUDAGraph" if kind == "cuda" else "XPUGraph"):Graph})
                def compile(module, **kwargs):
                    captures.append(kwargs)
                    return compiled
                torch = SimpleNamespace(compile=compile, **{kind:runtime},
                    testing=SimpleNamespace(assert_close=lambda actual, ref, **kwargs:
                        self.assertEqual(actual.value, ref.value)))
                stage = CompiledStage(torch, object(), "test", lambda *args:None)
                source = Tensor(3, kind)
                output, = stage(source)
                self.assertEqual(output.value, 6)
                self.assertIsNot(stage.inputs[0], source)
                again, = stage(Tensor(7, "cpu"))
                self.assertIs(again, output)
                self.assertEqual(output.value, 14)
                self.assertEqual(stage.calls, 2)
                self.assertEqual(len(captures), 1)
                self.assertTrue(captures[0]["fullgraph"])
                self.assertTrue(captures[0]["options"]["emulate_precision_casts"])
                if kind == "cuda":
                    self.assertFalse(captures[0]["options"]["triton.cudagraphs"])
                bad = Tensor(1, kind)
                bad.shape = (2,)
                with self.assertRaises(ValueError):
                    stage(bad)


if __name__ == "__main__":
    unittest.main()
