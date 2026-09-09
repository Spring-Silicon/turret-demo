#!/usr/bin/env python3
"""CPU-only temporal profile contracts; GPU parity is a separate qualification."""
import ast
import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spring_turret.sam31_tracking import CurrentImage, XpuSource, prune_tracker_state
from spring_turret.detection import validate_config, validate_tracking_result, DetectionController, WorkerClient
from spring_turret.models import MODELS, model_available, model_prompts


class TemporalTests(unittest.TestCase):
    def test_tracking_worker_routes_device_and_never_loads_intel_libraries_on_cuda(self):
        with tempfile.TemporaryDirectory() as cache:
            cfg = {"enabled":True, "model":"sam3.1-tracking", "device_type":"cuda",
                   "python":"/usr/bin/python", "checkpoint":"/models/full.pt",
                   "cache_dir":cache, "sam31_tracking_bundle":"/opt/sam3"}
            with patch("spring_turret.detection.subprocess.Popen") as popen:
                WorkerClient(cfg).launch()
                args = popen.call_args.args[0]
                self.assertEqual(args[args.index("--device-type")+1], "cuda")
                self.assertEqual(args[args.index("--precision")+1], "bfloat16")
                self.assertEqual(args[args.index("--source-bundle")+1], "/opt/sam3")
                self.assertEqual(popen.call_args.kwargs["env"].get("LD_LIBRARY_PATH"), os.environ.get("LD_LIBRARY_PATH"))

    def test_cuda_source_keeps_cuda_and_preserves_annotation_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sam3/model").mkdir(parents=True)
            (root / "sam3/model/sam3_multiplex_tracking.py").touch()
            (root / "sam3/__init__.py").write_text('''
from dataclasses import dataclass
from typing import ClassVar
DEVICE = "cuda:0"
@dataclass
class Item:
    shared: ClassVar[int] = 1
    count: int = 2
''')
            script = '''
import sys
from pathlib import Path
from dataclasses import fields
from spring_turret.sam31_tracking import configure_source
fingerprints = configure_source(Path(sys.argv[1]), sys.argv[2])
import sam3
assert sam3.DEVICE == sys.argv[2] + ':0'
assert [field.name for field in fields(sam3.Item)] == ['count']
assert sam3.Item.__annotations__['count'] is int
assert len(fingerprints['sam3/__init__.py']) == 64
'''
            for device in ("xpu", "cuda"):
                subprocess.run([sys.executable, "-c", script, str(root), device], check=True,
                    env={**os.environ, "PYTHONPATH":str(Path(__file__).resolve().parents[1]/"src")})

    def test_sam_cancellation_with_blocked_prefetch_reader_exits_cleanly(self):
        script = '''
import sys
from types import SimpleNamespace
from spring_turret import sam31_worker as worker
worker.Sam31Engine = lambda **kwargs: SimpleNamespace(
    preprocessor=SimpleNamespace(prepare_cpu=lambda jpeg: None), image_stage=None)
sys.argv = ['sam31_worker', '--checkpoint', sys.argv[1]]
worker.main()
'''
        proc = subprocess.Popen([sys.executable, '-c', script, __file__],
            env={**os.environ, 'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            import json
            self.assertEqual(json.loads(proc.stdout.readline())['type'], 'ready')
            proc.terminate()
            self.assertEqual(proc.wait(timeout=3), 0)
            self.assertNotIn(b'Fatal Python error', proc.stderr.read())
        finally:
            if proc.poll() is None: proc.kill(); proc.wait()
            for stream in (proc.stdin,proc.stdout,proc.stderr): stream.close()

    def test_profile_is_explicit_optional_xpu_or_cuda_and_free_text(self):
        cfg = {"enabled":True, "sam31_tracking_bundle":"/sam", "python":"/bin/python",
               "checkpoint":"/weights", "cache_dir":"/cache", "model":"sam3.1-tracking"}
        validate_config(cfg)
        self.assertTrue(model_available("sam3.1-tracking", cfg))
        validate_config({**cfg, "device_type":"cuda"})
        self.assertTrue(model_available("sam3.1-tracking", {**cfg, "device_type":"cuda"}))
        for patch in ({"sam31_tracking_bundle":"relative"}, {"device_type":"cpu"}):
            with self.assertRaises(ValueError): validate_config({**cfg, **patch})
        self.assertFalse(model_available("sam3.1-tracking", {"enabled":True}))
        self.assertEqual(model_prompts("sam3.1-tracking", ["red cups", "bottles"]), ["red cups", "bottles"])
        self.assertEqual(MODELS["sam3.1-tracking"]["worker"], "sam31_tracking_worker.py")

    def test_session_prompts_are_independent(self):
        c = DetectionController({"enabled":True, "sam31_tracking_bundle":"/sam"}, None)
        c.set_prompt("cup")
        c.set_model("sam3.1-tracking")
        self.assertEqual(c.prompts, [])
        c.set_prompts(["table", "chair"])
        c.set_model("sam3.1")
        self.assertEqual(c.prompts, ["cup"])
        c.set_model("sam3.1-tracking")
        self.assertEqual(c.prompts, ["table", "chair"])
        revision = c.revision
        c.set_prompts(["table", "chair"])
        self.assertGreater(c.revision, revision)  # Same-text Update also resets memory.

    def test_native_id_contract_does_not_claim_compilation(self):
        box = {"instance_id":103, "prompt":"cup", "xyxy":[.1,.2,.3,.4], "score":.9}
        result = {"boxes":[box], "active_instance_ids":[103], "temporal_tracking":True,
                  "tracking_backend":"sam31-object-multiplex", "tracking_frame":2,
                  "memory_frames":2, "torch_compile":False, "sycl_graph":False}
        validate_tracking_result(result, ["cup"])
        self.assertEqual(result["boxes"][0]["instance_id"], 103)
        for patch in ({"temporal_tracking":False}, {"active_instance_ids":[True]},
                      {"boxes":[box, box]}, {"tracking_frame":0},
                      {"boxes":[{**box, "xyxy":[0,0,float('nan'),1]}]}):
            with self.assertRaises(RuntimeError): validate_tracking_result({**result, **patch}, ["cup"])

    def test_image_slot_does_not_cache_old_or_future_frames(self):
        first, second = object(), object()
        image = CurrentImage(first, 17)
        self.assertIs(image[0], first)
        self.assertIs(image[17], first)
        self.assertEqual(len(image), 1)
        with self.assertRaises(IndexError): image[16]
        image = CurrentImage(second, 18)
        self.assertIs(image[18], second)

    def test_xpu_transform_leaves_cuda_backend_namespace_intact(self):
        source = 'a.cuda(); a.is_cuda; torch.cuda.synchronize(); torch.device("cuda:1"); torch.backends.cuda.matmul.allow_tf32 = False'
        changed = ast.unparse(XpuSource().visit(ast.parse(source)))
        self.assertIn('a.xpu()', changed)
        self.assertIn('a.is_xpu', changed)
        self.assertIn('torch.xpu.synchronize()', changed)
        self.assertIn("'xpu:1'", changed)
        self.assertIn('torch.backends.cuda.matmul', changed)

    def test_pruning_preserves_reachable_attention_and_first_condition(self):
        store = {"cond_frame_outputs":{i:object() for i in range(0,200,16)},
                 "non_cond_frame_outputs":{i:object() for i in range(200)}}
        state = {"output_dict":store, "output_dict_per_obj":{0:copy.copy(store)},
                 "temp_output_dict_per_obj":{}, "frames_already_tracked":dict.fromkeys(range(200)),
                 "consolidated_frame_inds":{"cond_frame_outputs":set(range(0,200,16)),
                                             "non_cond_frame_outputs":set(range(200))}}
        prune_tracker_state(state, 199, keep_first=True)
        self.assertEqual(set(store["cond_frame_outputs"]), {0,144,160,176,192})
        self.assertEqual(set(store["non_cond_frame_outputs"]), set(range(168,200)))
        self.assertIn(0, state["consolidated_frame_inds"]["cond_frame_outputs"])


if __name__ == "__main__":
    unittest.main()
