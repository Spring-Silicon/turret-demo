"""The same public contract for CUDA, XPU, temporal and detector-only workers."""
import json
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spring_turret.api_contract import API_VERSION, detection_result, detection_status, public_status
from spring_turret.detection import DetectionController, WorkerClient
from spring_turret.models import MODELS, model_available, model_prompts


class ContractTests(unittest.TestCase):
    def test_yolo_removed_from_registry_legacy_snapshots_and_command_boundary(self):
        from spring_turret.viewer import ViewerApplication
        self.assertEqual(set(MODELS), {"sam3.1", "sam3.1-mask", "sam3.1-tracking", "sam3.1-v18",
                                      "efficient-nomem", "hybrid-nomem", "efficient-tracking", "efficient-memory", "sam3.1-nomem"})
        self.assertFalse(model_available("yolo26x", {"enabled":True, "yolo26x_checkpoint":"/old"}))
        with self.assertRaises(ValueError): model_prompts("yolo26x", ["person"])
        for model in MODELS:
            self.assertEqual(model_prompts(model, ["red shoe"]), ["red shoe"])
        result = detection_status({"models":[{"id":key} for key in [*MODELS, "yolo26x"]]})
        self.assertEqual({item["id"] for item in result["models"]}, set(MODELS))
        app = ViewerApplication.__new__(ViewerApplication)
        with patch("spring_turret.viewer.command_request") as command:
            with self.assertRaises(ValueError): app.command("/api/detection/model", {"model":"yolo26x"})
            command.assert_not_called()  # No hardware/worker side effects.
        with patch("spring_turret.detection.subprocess.Popen") as launch:
            with self.assertRaises(ValueError): WorkerClient({"model":"yolo26x"}).launch()
            launch.assert_not_called()
    def test_device_progress_uses_common_phases_without_changing_raw_state(self):
        for raw, phase in (("loading", "loading"), ("loading_tracker", "loading"),
                           ("compiling", "preparing"), ("compiling_tracker_memory_update", "preparing"),
                           ("capturing", "capturing"), ("validating", "validating"),
                           ("waiting_for_camera", "waiting_for_camera"), ("running", "running")):
            for device in (None, "xpu", "cuda"):
                source = {"state":raw, "device_type":device}
                result = detection_status(source)
                self.assertEqual(result["progress"], {"phase":phase, "stage":raw})
                self.assertEqual(result["state"], raw)
                self.assertIsNone(result["timing"]["model_ms"])
                self.assertIsNone(result["execution"]["torch_compile"])
                self.assertIsNone(result["pipeline_timing"]["cycle_ms"])
                self.assertEqual(source, {"state":raw, "device_type":device})
                self.assertEqual(result, detection_status(result))

    def test_temporal_warmup_keeps_frame_and_does_not_invent_model_timing(self):
        raw = {"state":"running", "model":"sam3.1-tracking", "revision":3,
               "frame_sequence":12, "frame_url":"/3-12.jpg", "mask_overlay":{"png":"pixels"},
               "progress_stage":"compiling_tracker_memory_update", "latency_ms":210,
               "timing":{"tracking_ms":210}, "sycl_graph":True}
        result = detection_status(raw)
        self.assertEqual(result["progress"]["phase"], "preparing")
        for field in ("state", "revision", "frame_sequence", "frame_url", "mask_overlay", "latency_ms"):
            self.assertEqual(result[field], raw[field])
        self.assertIsNone(result["timing"]["model_ms"])
        self.assertEqual(result["execution"]["device_type"], "xpu")
        for state in ("error", "idle", "disabled", "waiting_for_camera"):
            self.assertEqual(detection_status({**raw, "state":state})["progress"]["phase"], state)

    def test_public_boundary_is_common_and_preserves_device_policy(self):
        source = {"servo":{"armed":True}, "detection":{"state":"loading"}, "tracking":{"state":"off"}}
        original = copy.deepcopy(source)
        result = public_status(source)
        self.assertEqual(source, original)
        self.assertIsNone(result["tracking"]["continuity"])
        self.assertTrue(result["servo"]["armed"])
        arc = public_status({**source, "tracking":{"state":"off", "continuity":"hold-reacquire"}})
        self.assertEqual(set(result["tracking"]), set(arc["tracking"]))
        self.assertEqual(arc["tracking"]["continuity"], "hold-reacquire")
        self.assertEqual(result["api_version"], API_VERSION)
        self.assertEqual(result, public_status(result))

    def test_all_models_share_fields_and_preserve_backend_diagnostics(self):
        for model in MODELS:
            states = []
            for device in ("xpu", "cuda"):
                config = {"enabled":True, "device_type":device, "model":model,
                          "sam31_tracking_bundle":"/sam"}
                controller = DetectionController(config, None)
                controller.result = {"torch_compile":True, device == "cuda" and "cuda_graph" or "sycl_graph":True,
                                     "custom_kernel_diagnostic":7, "latency_ms":123}
                status = controller.status()
                self.assertEqual(status["api_version"], API_VERSION)
                self.assertEqual(status["execution"]["device_type"], device)
                self.assertEqual(status["execution"]["graph_replay"], "cuda" if device == "cuda" else "sycl")
                self.assertIsNone(status["timing"]["model_ms"])
                self.assertEqual(status["custom_kernel_diagnostic"], 7)
                self.assertEqual(status["boxes"], [])
                self.assertIsNone(status["frame_sequence"])
                json.dumps(status, allow_nan=False)
                states.append(status)
            for section in ("execution", "timing"):
                self.assertEqual(set(states[0][section]), set(states[1][section]))
            self.assertEqual(set(states[0])-{"sycl_graph"}, set(states[1])-{"cuda_graph"})

    def test_worker_routes_are_implementation_details_not_different_http_apis(self):
        with tempfile.TemporaryDirectory() as cache:
            for device in ("xpu", "cuda"):
                for model in ("sam3.1", "sam3.1-tracking"):
                    cfg = {"enabled":True, "model":model, "device_type":device,
                           "python":"/bin/python", "checkpoint":"/weights", "cache_dir":cache,
                           "sam31_tracking_bundle":"/sam"}
                    with patch("spring_turret.detection.subprocess.Popen") as popen:
                        WorkerClient(cfg).launch()
                        argv = popen.call_args.args[0]
                        self.assertEqual(argv[argv.index("--device-type")+1], device)
                        self.assertTrue(argv[2].endswith(MODELS[model]["worker"]))

    def test_defaults_are_owned_and_actual_measurements_not_overwritten(self):
        first = detection_result({}, "xpu")
        first["boxes"].append({})
        self.assertEqual(detection_result({}, "cuda")["boxes"], [])
        self.assertEqual(detection_result({"timing":{"preprocess_ms":4}}, "xpu")["timing"]["preprocess_ms"], 4)


if __name__ == "__main__": unittest.main()
