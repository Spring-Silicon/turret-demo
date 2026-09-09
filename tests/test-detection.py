#!/usr/bin/env python3
"""No GPU needed: prompt cancellation, latest-frame routing, errors, and API."""

import base64
import json
import sys
import threading
import time
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spring_turret.detection import DetectionController, validate_config
from spring_turret.server import TurretApplication, make_handler
from spring_turret.sam31_worker import Sam31Engine
from spring_turret.instances import InstanceAssociator


def eventually(check, timeout=3):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if check():
            return
        time.sleep(0.01)
    raise AssertionError("timed out")


class Camera:
    sequence = 0

    def wait_for_frame(self, previous, timeout):
        self.sequence += 1
        return self.sequence, b"camera-jpeg"

    def wait_for_sample(self, previous, timeout, after=0.0):
        sequence, jpeg = self.wait_for_frame(previous, timeout)
        return sequence, jpeg, time.monotonic()

    def status(self):
        return {"online": True, "frame_age_ms": 1}


class Worker:
    def __init__(self, config):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.requests = []
        self.stopped = False
        self.fail = False

    def launch(self):
        pass

    def receive(self, timeout):
        return {"type": "ready"}

    def detect(self, request_id, jpeg, prompts, progress):
        self.requests.append((request_id, jpeg, prompts))
        self.entered.set()
        self.release.wait(2)
        if self.fail:
            raise RuntimeError("test inference error")
        return {
            "type": "result",
            "id": request_id,
            "boxes": [],
            "latency_ms": 1,
            "torch_compile": True,
            "sycl_graph": True,
            "jpeg": base64.b64encode("|".join(prompts).encode()).decode(),
        }

    def stop(self):
        self.stopped = True
        self.release.set()

    def cancel(self):
        self.release.set()


class DetectionTests(unittest.TestCase):
    def test_temporal_worker_preserves_ids_and_receives_session_boundary(self):
        c, worker = self.controller()
        c.model = "sam3.1-tracking"
        c.config["sam31_tracking_bundle"] = "/source"
        def forbidden(*args):
            raise AssertionError("Native SAM IDs must not use geometric association")
        c.instances.update = forbidden
        original = worker.detect
        def detect(*args, **kwargs):
            self.assertEqual(kwargs["session_revision"], c.revision)
            self.assertIsInstance(kwargs["captured_at"], float)
            result = original(*args)
            return {**result, "torch_compile":False, "sycl_graph":False,
                    "temporal_tracking":True, "tracking_backend":"sam31-object-multiplex",
                    "tracking_frame":2, "memory_frames":2, "active_instance_ids":[777],
                    "boxes":[{"instance_id":777, "prompt":"cup", "score":.9, "xyxy":[.1,.2,.3,.4]}]}
        worker.detect = detect
        worker.release.set()
        c.set_prompt("cup")
        eventually(lambda: c.status()["state"] == "running")
        self.assertEqual(c.status()["boxes"][0]["instance_id"], 777)
        self.assertFalse(c.status()["torch_compile"])

    def test_temporal_marker_cannot_bypass_original_detector_graph_gate(self):
        c, worker = self.controller()
        original = worker.detect
        worker.detect = lambda *args: {**original(*args), "torch_compile":False,
                                      "temporal_tracking":True}
        worker.release.set()
        c.set_prompt("cup")
        eventually(lambda: c.status()["state"] == "error")
        self.assertIn("compiled", c.status()["error"])

    def test_click_history_has_no_age_expiry_but_remains_bounded(self):
        c = DetectionController({"enabled": True}, Camera())
        c.state = "running"
        box = {"prompt": "cup", "instance_id": 42, "xyxy": [.1, .2, .3, .4]}
        with patch("spring_turret.detection.time.monotonic") as clock:
            for i in range(21):
                clock.return_value = 10 + i / 30
                c._cache_frame(f"{c.revision}-{i}", b"jpeg", [box], clock.return_value)
            self.assertEqual(len(c.frames), 8)
            self.assertIsNone(c.frame(f"{c.revision}-0"))
            self.assertEqual(c.selection(c.revision, 0, 42), box)
            clock.return_value = 70.0
            c._cache_frame(f"{c.revision}-24", b"jpeg", [box], clock.return_value)
            self.assertEqual(c.selection(c.revision, 0, 42), box)  # 60 seconds old.
            with self.assertRaisesRegex(ValueError, "not in"):
                c.selection(c.revision, 0, 99)
            with self.assertRaisesRegex(ValueError, "no longer available"):
                c.selection(c.revision + 1, 0, 42)
            for i in range(300):
                c._cache_frame(f"{c.revision}-{100+i}", b"jpeg", [box], clock.return_value)
            self.assertEqual(len(c.frame_selections), 128)
            with self.assertRaisesRegex(ValueError, "no longer available"):
                c.selection(c.revision, 0, 42)  # Evicted by count, not elapsed time.

    def test_model_switch_serializes_workers_preserves_lists_and_discards_old_frames(self):
        workers = []
        def factory(config):
            self.assertTrue(all(worker.stopped for worker in workers))
            worker = Worker(config)
            worker.model = config["model"]
            workers.append(worker)
            return worker
        c = DetectionController({"enabled": True, "max_fps": 30, "sam31_mask_bundle": "/mask"}, Camera(), factory)
        c.start()
        self.addCleanup(c.stop)
        c.set_prompts(["face"])
        eventually(lambda: workers and workers[0].entered.is_set())
        c.set_model("sam3.1-mask")
        self.assertEqual(c.status()["prompts"], [])
        c.set_prompts(["person"])
        self.assertIsNone(c.status()["frame_url"])
        eventually(lambda: len(workers) == 2 and workers[1].entered.is_set())
        workers[1].release.set()
        eventually(lambda: c.status()["state"] == "running")
        self.assertEqual(c.status()["model"], "sam3.1-mask")
        self.assertTrue(all(key.startswith("3-") for key in c.frames))
        self.assertIsNone(c.status()["classes"])
        c.set_prompts(["Cup", "person"])
        self.assertEqual(c.status()["prompts"], ["Cup", "person"])
        c.set_model("sam3.1")
        self.assertEqual(c.status()["prompts"], ["face"])
        eventually(lambda: len(workers) == 3)
        c.set_model("sam3.1-mask")
        self.assertEqual(c.status()["prompts"], ["Cup", "person"])
        with self.assertRaises(ValueError): c.selection(2, 1, 1)

    def test_model_validation_and_switch_to_empty_unloads_previous_worker(self):
        c, worker = self.controller()
        for value in (None, True, [], "yolo26n", "unknown"):
            with self.assertRaises(ValueError): c.set_model(value)
        with self.assertRaises(ValueError): c.set_model("sam3.1-mask")
        c.config["sam31_mask_bundle"] = "/mask"
        c.set_model("sam3.1-mask")
        c.set_prompts(["face"])
        self.assertTrue(worker.entered.wait(1))
        c.set_model("sam3.1")
        eventually(lambda: worker.stopped)
        self.assertEqual(c.status()["state"], "idle")
        self.assertEqual(c.status()["prompts"], [])
        with self.assertRaises(ValueError):
            validate_config({"enabled": True, "python": "/p", "checkpoint": "/c", "cache_dir": "/cache", "model": "sam3.1-mask"})

    def test_engine_preserves_every_instance_and_category_on_one_frame(self):
        engine = Sam31Engine.__new__(Sam31Engine)
        engine.device = 0
        engine.dtype = "fake-fp16"
        engine.graph_active = True
        engine.graph_error = None
        engine.validation = {}
        engine.torch = SimpleNamespace(
            __version__="test",
            inference_mode=nullcontext,
            autocast=lambda *args, **kwargs: nullcontext(),
            xpu=SimpleNamespace(
                synchronize=lambda: None, get_device_name=lambda _: "test"
            ),
        )
        engine.device_type = "xpu"
        engine.runtime = engine.torch.xpu
        pixels = object()
        engine._pixels = lambda jpeg: pixels
        engine._prepare = lambda pixels, prompts: prompts
        engine.grounding_batch_size = 2
        instances = {
            "person": [
                {"xyxy": [0.1, 0.2, 0.3, 0.4], "score": 0.9},
                {"xyxy": [0.5, 0.2, 0.8, 0.9], "score": 0.8},
            ],
            "cup": [{"xyxy": [0.2, 0.6, 0.3, 0.8], "score": 0.85}],
            "chair": [],
        }
        engine.text_cache = dict.fromkeys(instances)
        image_calls = []
        head_calls = []

        def image_stage(image):
            image_calls.append(image)
            return "shared-features"

        def run_heads(features, prompts):
            head_calls.append((features, prompts))
            return [instances[prompt] for prompt in prompts]

        engine.image_stage = image_stage
        engine._run_heads = run_heads
        engine._decode_outputs = lambda outputs: outputs
        with patch("spring_turret.sam31_worker.annotate", return_value=b"jpeg") as draw:
            result = engine.detect_many(b"original", ["person", "cup", "chair"])
        self.assertEqual(image_calls, [pixels])
        self.assertEqual(head_calls, [("shared-features", list(instances))])
        self.assertTrue(result["text_cache_hit"])
        self.assertTrue(result["shared_image_features"])
        self.assertEqual([c["count"] for c in result["categories"]], [2, 1, 0])
        self.assertEqual(
            [b["prompt"] for b in result["boxes"]], ["person", "person", "cup"]
        )
        self.assertEqual([b["prompt_index"] for b in result["boxes"]], [0, 0, 1])
        self.assertEqual(result["boxes"][0]["color"], result["boxes"][1]["color"])
        self.assertNotEqual(result["boxes"][0]["color"], result["boxes"][2]["color"])
        draw.assert_called_once_with(b"original", result["boxes"])
        with patch("spring_turret.sam31_worker.annotate") as draw:
            overlay = engine.detect_many(b"original", ["person", "cup", "chair"], client_overlay=True)
        draw.assert_not_called()
        self.assertTrue(overlay["client_overlay"])
        self.assertNotIn("jpeg", overlay)
        self.assertEqual(overlay["boxes"], result["boxes"])
        self.assertTrue(result["sycl_graph"])
        self.assertFalse(result["cuda_graph"])
        engine.device_type = "cuda"
        cuda = engine.detect_many(b"original", ["person"], client_overlay=True)
        self.assertTrue(cuda["cuda_graph"])
        self.assertFalse(cuda["sycl_graph"])
        self.assertEqual(cuda["device_type"], "cuda")

    def test_text_cache_is_owned_bounded_and_reused_across_reordering(self):
        class Tensor:
            value = ""

            def clone(self):
                copy = Tensor()
                copy.value = self.value
                return copy

        engine = Sam31Engine.__new__(Sam31Engine)
        engine.text_cache = {}
        engine._tokens = lambda prompt: prompt
        static = Tensor()
        calls = []

        def text_stage(prompt):
            calls.append(prompt)
            static.value = prompt
            return (static,)

        engine.text_stage = text_stage
        embeddings = engine._encode_prompts(["person", "cup"])
        self.assertEqual([e[0].value for e in embeddings], ["person", "cup"])
        self.assertEqual(calls, ["person", "cup"])
        engine._encode_prompts(["cup", "person"])
        self.assertEqual(calls, ["person", "cup"])
        engine._encode_prompts(["person", "chair"])
        self.assertEqual(calls, ["person", "cup", "chair"])
        self.assertEqual(set(engine.text_cache), {"person", "chair"})
        engine._encode_prompts(["cup"])
        self.assertEqual(calls, ["person", "cup", "chair", "cup"])
        self.assertEqual(set(engine.text_cache), {"cup"})

    def test_fixed_batch_shapes_and_single_prompt_tail(self):
        engine = Sam31Engine.__new__(Sam31Engine)
        engine.grounding_batch_size = 2
        self.assertEqual(engine._batch_sizes(1), {1})
        self.assertEqual(engine._batch_sizes(3), {1, 2})
        self.assertEqual(engine._batch_sizes(8), {2})
        engine.grounding_batch_size = 4
        self.assertEqual(engine._batch_sizes(3), {4})
        self.assertEqual(engine._batch_sizes(5), {1, 4})

    def controller(self):
        worker = Worker({})
        controller = DetectionController(
            {"enabled": True, "max_fps": 30}, Camera(), lambda _: worker
        )
        controller.start()
        self.addCleanup(controller.stop)
        return controller, worker

    def test_prompt_replacement_discards_inflight_result(self):
        c, worker = self.controller()
        c.set_prompt("person")
        self.assertTrue(worker.entered.wait(1))
        c.set_prompt("chair")
        self.assertEqual(c.status()["revision"], 2)
        worker.release.set()
        eventually(lambda: c.status()["state"] == "running")
        status = c.status()
        self.assertEqual(status["prompt"], "chair")
        self.assertEqual(c.frame(status["frame_url"].split("/")[-1][:-4]), b"chair")
        self.assertTrue(all(key.startswith("2-") for key in c.frames))

    def test_clear_discards_result_and_stops_sampling(self):
        c, worker = self.controller()
        c.set_prompt("person")
        self.assertTrue(worker.entered.wait(1))
        c.set_prompt("")
        worker.release.set()
        time.sleep(0.1)
        self.assertEqual(c.status()["state"], "idle")
        self.assertIsNone(c.status()["frame_url"])
        self.assertEqual(len(worker.requests), 1)
        self.assertFalse(c.frames)

    def test_failure_is_reported_without_retry_loop(self):
        c, worker = self.controller()
        worker.fail = True
        worker.release.set()
        c.set_prompt("person")
        eventually(lambda: c.status()["state"] == "error")
        self.assertEqual(c.status()["error"], "test inference error")
        time.sleep(0.1)
        self.assertEqual(len(worker.requests), 1)
        self.assertTrue(worker.stopped)

    def test_cuda_controller_requires_cuda_graph_and_compilation_proof(self):
        for compiled, cuda in ((True, True), (True, False), (False, True)):
            with self.subTest(compiled=compiled, cuda=cuda):
                c, worker = self.controller()
                c.config["device_type"] = "cuda"
                original = worker.detect
                def detect(*args, **kwargs):
                    return {**original(*args, **kwargs), "torch_compile":compiled,
                            "cuda_graph":cuda, "sycl_graph":not cuda}
                worker.detect = detect
                worker.release.set()
                c.set_prompt("person")
                expected = "running" if compiled and cuda else "error"
                eventually(lambda: c.status()["state"] == expected)
                if expected == "error":
                    self.assertIn("compiled cuda_graph", c.status()["error"])
                    self.assertIsNone(c.status()["frame_url"])
                c.stop()

    def test_bad_prompts(self):
        c, _ = self.controller()
        for prompt in (True, None, ["person"], "x" * 257, "a\nb"):
            with self.assertRaises(ValueError):
                c.set_prompt(prompt)
        c.set_prompt("  chair  ")
        self.assertEqual(c.status()["prompt"], "chair")

    def test_disabled_is_optional(self):
        c = DetectionController({}, Camera())
        c.start()
        c.stop()
        self.assertEqual(c.status()["state"], "disabled")
        with self.assertRaises(ValueError):
            c.set_prompt("person")

    def test_encoder_pose_is_resolved_for_capture_time_after_inference(self):
        c, worker = self.controller()
        pose = {"axes": {
            "x": {"degrees": 10, "goal_degrees": 40}, "y": {"degrees": 5, "goal_degrees": 5}}}
        requested = []
        def provider(at):
            requested.append(at)
            self.assertTrue(worker.release.is_set())
            # These are interpolated historical angles, not current angles.
            return {**pose, "sampled_at": at, "read_completed_at": at + .04,
                    "interpolated": True}
        c.pose_provider = provider
        c.set_prompt("cup")
        self.assertTrue(worker.entered.wait(1))
        self.assertEqual(requested, [])
        worker.release.set()
        eventually(lambda: c.status()["state"] == "running")
        self.assertEqual(c.status()["frame_pose"]["axes"]["x"]["degrees"], 10)
        self.assertEqual(c.status()["captured_at"], c.status()["frame_pose"]["sampled_at"])

    def test_stop_during_inference_does_not_publish_old_pose(self):
        c, worker = self.controller()
        available = True
        c.pose_provider = lambda at: {"sampled_at": at} if available else None
        c.set_prompt("cup")
        self.assertTrue(worker.entered.wait(1))
        available = False
        worker.release.set()
        eventually(lambda: c.status()["state"] == "running")
        self.assertIsNone(c.status()["frame_pose"])

    def test_old_pose_is_not_paired_with_a_later_camera_frame(self):
        c, worker = self.controller()
        c.pose_provider = lambda captured_at: {"sampled_at": time.monotonic() - 1}
        c.set_prompt("cup")
        worker.release.set()
        eventually(lambda: c.status()["state"] == "running")
        self.assertIsNone(c.status()["frame_pose"])

    def test_latest_camera_frame_does_not_wait_for_a_newer_pose(self):
        c, worker = self.controller()
        frame_time = time.monotonic() - .01
        def sample(previous, timeout, after=0):
            self.assertEqual(after, 0)
            return 1, b"latest-jpeg", frame_time
        c.camera.wait_for_sample = sample
        calls = []
        def pose(at):
            calls.append(at)
            return {"sampled_at": at-.03, "read_completed_at": at-.01, "axes": {}}
        c.pose_provider = pose
        c.set_prompt("cup")
        self.assertTrue(worker.entered.wait(1))
        worker.release.set()
        eventually(lambda: c.status()["state"] == "running")
        self.assertEqual(calls, [frame_time])
        self.assertEqual(worker.requests[0][1], b"latest-jpeg")
        self.assertEqual(c.status()["frame_pose"]["read_completed_at"], frame_time-.01)

    def test_pose_read_that_completed_after_frame_is_rejected(self):
        c, worker = self.controller()
        c.pose_provider = lambda at: {"sampled_at": at-.03, "read_completed_at": at+.001}
        c.set_prompt("cup")
        worker.release.set()
        eventually(lambda: c.status()["state"] == "running")
        self.assertIsNone(c.status()["frame_pose"])

    def test_client_overlay_serves_exact_input_jpeg_and_pipeline_timings(self):
        c, worker = self.controller()
        original_detect = worker.detect
        def detect(*args):
            result = original_detect(*args)
            result.pop("jpeg")
            return {**result, "client_overlay": True}
        worker.detect = detect
        c.set_prompt("cup")
        worker.release.set()
        eventually(lambda: c.status()["state"] == "running")
        result = c.status()
        self.assertTrue(result["client_overlay"])
        self.assertEqual(c.frame(result["frame_url"].split("/")[-1][:-4]), b"camera-jpeg")
        self.assertEqual(set(result["pipeline_timing"]),
                         {"pose_ms", "capture_wait_ms", "worker_roundtrip_ms", "cycle_ms"})
        self.assertTrue(all(value >= 0 for value in result["pipeline_timing"].values()))

    def test_config(self):
        validate_config({})
        base = {
            "enabled": True,
            "python": "/venv/bin/python",
            "checkpoint": "/models/sam.pt",
            "cache_dir": "/cache",
        }
        validate_config(base)
        validate_config({**base, "max_fps": 0})
        for key, value in (
            ("python", "relative"),
            ("confidence", 0),
            ("max_fps", 99),
            ("max_fps", -1),
            ("max_fps", float("nan")),
            ("max_fps", True),
            ("precision", "int8"),
        ):
            with self.assertRaises(ValueError):
                validate_config({**base, key: value})

    def test_uncapped_sampling_has_no_artificial_wait(self):
        for config in ({}, {"max_fps": 0}):
            controller = DetectionController(config, Camera())
            for elapsed in (0, .001, .05, .18, 2):
                self.assertEqual(controller._sampling_delay(elapsed), 0)
        controller = DetectionController({"max_fps": 5}, Camera())
        self.assertAlmostEqual(controller._sampling_delay(.05), .15)
        self.assertEqual(controller._sampling_delay(.25), 0)

    def test_new_results_wake_tracking_without_a_poll_timer(self):
        controller, worker = self.controller()
        controller.set_prompt("cup")
        self.assertTrue(worker.entered.wait(1))
        previous = controller.frame_version()
        woke = threading.Event()
        waiter = threading.Thread(target=lambda: (controller.wait_for_update(previous, 3), woke.set()))
        waiter.start()
        self.assertFalse(woke.wait(.01))
        worker.release.set()
        self.assertTrue(woke.wait(1))
        waiter.join(1)
        self.assertNotEqual(controller.frame_version(), previous)

    def test_multiple_categories_share_one_frame_and_clear_atomically(self):
        c, worker = self.controller()
        c.set_prompts(["person", "cup", "keyboard"])
        self.assertTrue(worker.entered.wait(1))
        self.assertEqual(
            worker.requests[0][1:], (b"camera-jpeg", ["person", "cup", "keyboard"])
        )
        c.set_prompts(["chair", "bottle"])
        worker.release.set()
        eventually(lambda: c.status()["state"] == "running")
        result = c.status()
        self.assertEqual(result["prompts"], ["chair", "bottle"])
        self.assertEqual(
            c.frame(result["frame_url"].split("/")[-1][:-4]), b"chair|bottle"
        )
        c.set_prompts([])
        self.assertEqual(c.status()["state"], "idle")
        self.assertEqual(c.status()["prompts"], [])
        self.assertFalse(c.frames)

    def test_prompt_list_validation_and_normalization(self):
        c, worker = self.controller()
        for prompts in (
            None,
            "person",
            {},
            [True],
            [None],
            ["x"] * 9,
            ["x" * 257],
            ["a\nb"],
        ):
            with self.assertRaises(ValueError):
                c.set_prompts(prompts)
        c.set_prompts(["  Person  ", "", "person", "cup", "  "])
        self.assertEqual(c.status()["prompts"], ["Person", "cup"])
        c.set_prompts(["", " "])
        self.assertEqual(c.status()["state"], "idle")

    def test_http_prompt_and_exact_frame(self):
        c, worker = self.controller()
        worker.release.set()

        class NoServo:
            def status(self):
                return {"armed": False}

        app = TurretApplication(
            {"camera": {}, "servo": {}}, camera=Camera(), servo=NoServo(), detection=c
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        connection = HTTPConnection(*server.server_address)
        self.addCleanup(connection.close)
        connection.request(
            "POST",
            "/api/detection/prompt",
            json.dumps({"prompt": "chair"}),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertFalse(json.loads(response.read())["servo"]["armed"])
        eventually(lambda: c.status()["state"] == "running")
        frame_url = c.status()["frame_url"]
        connection.request("GET", frame_url)
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.read(), b"chair")
        with patch.object(app, "status", side_effect=AssertionError("hardware status lock")):
            connection.request("GET", "/api/detection/status?revision=-1&sequence=-1")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            metadata = json.loads(response.read())
            self.assertEqual(metadata["prompts"], ["chair"])
            self.assertNotIn("servo", metadata)
            events = HTTPConnection(*server.server_address)
            self.addCleanup(events.close)
            events.request("GET", "/api/detection/events")
            stream = events.getresponse()
            self.assertEqual(stream.status, 200)
            self.assertEqual(stream.getheader("Content-Type"), "text/event-stream")
            event = json.loads(stream.readline().removeprefix(b"data: "))
            self.assertEqual(event["prompts"], ["chair"])
            self.assertEqual(base64.b64decode(event["jpeg"]), b"chair")
            stream.close()
            events.close()
        for query in ("revision=bad", "sequence=-2"):
            connection.request("GET", "/api/detection/status?" + query)
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            response.read()
        c.set_prompt("")
        connection.request("GET", frame_url)
        response = connection.getresponse()
        self.assertEqual(response.status, 404)
        response.read()
        connection.request("POST", "/api/detection/prompt", '{"prompt":true}')
        response = connection.getresponse()
        self.assertEqual(response.status, 400)
        response.read()
        connection.request(
            "POST", "/api/detection/prompts", '{"prompts":["person","cup"]}'
        )
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(
            json.loads(response.read())["detection"]["prompts"], ["person", "cup"]
        )
        for payload in (
            '{"prompts":"person"}',
            '{"prompts":[false]}',
            '{"prompts":[],"extra":1}',
        ):
            connection.request("POST", "/api/detection/prompts", payload)
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            response.read()
        c.config["sam31_mask_bundle"] = "/mask"
        connection.request("POST", "/api/detection/model", '{"model":"sam3.1-mask"}')
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        body = json.loads(response.read())
        self.assertEqual(body["detection"]["model"], "sam3.1-mask")
        self.assertFalse(body["servo"]["armed"])
        self.assertIsNone(body["tracking"]["target"])
        for payload in ('{"model":"yolo26x"}', '{"model":"nano"}', '{"model":[]}', '{"model":"sam3.1","extra":1}'):
            connection.request("POST", "/api/detection/model", payload)
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            response.read()
        for target in ("person", "cup", None):
            c.set_prompts(["person", "cup"])
            connection.request("POST", "/api/tracking/target", json.dumps({"target": target}))
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            body = json.loads(response.read())
            self.assertEqual(body["tracking"]["target"], target)
            self.assertFalse(body["servo"]["armed"])
        for payload in ('{"target":"unapplied"}', '{"target":true}', '{"target":[]}', '{"target":"person","extra":1}'):
            connection.request("POST", "/api/tracking/target", payload)
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            response.read()

        c.stop()
        c.state = "running"
        c.frame_selections[f"{c.revision}-777"] = {"captured_at": time.monotonic(),
            "boxes": [{"prompt": "cup", "instance_id": 42, "xyxy": [.1, .2, .3, .4], "score": .9}]}
        selection = {"revision": c.revision, "frame_sequence": 777, "instance_id": 42}
        connection.request("POST", "/api/tracking/instance", json.dumps(selection))
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        body = json.loads(response.read())
        self.assertEqual(body["tracking"]["instance_id"], 42)
        self.assertEqual(body["tracking"]["target"], "cup")
        self.assertFalse(body["servo"]["armed"])
        for invalid in ({**selection, "instance_id": 100}, {**selection, "revision": -1},
                        {**selection, "frame_sequence": 778}, {**selection, "instance_id": True},
                        {**selection, "xyxy": [0, 0, 1, 1]}):
            connection.request("POST", "/api/tracking/instance", json.dumps(invalid))
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            response.read()
            self.assertEqual(app.tracking.instance_id, 42)
        c.frame_selections[f"{c.revision}-777"]["captured_at"] -= 60
        connection.request("POST", "/api/tracking/instance", json.dumps(selection))
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertFalse(json.loads(response.read())["servo"]["armed"])


class InstanceTests(unittest.TestCase):
    @staticmethod
    def box(cx, prompt="cup"):
        return {"xyxy": [cx-.03, .4, cx+.03, .5], "prompt": prompt, "score": .9}

    def test_ids_survive_reordering_not_nearest_frame_center(self):
        tracker = InstanceAssociator({})
        a, b = tracker.update([self.box(.2), self.box(.5)], None, 1)
        b2, a2 = tracker.update([self.box(.49), self.box(.21)], None, 1.2)
        self.assertEqual((a["instance_id"], b["instance_id"]), (a2["instance_id"], b2["instance_id"]))

    def test_camera_pan_is_compensated_for_large_image_shift(self):
        tracker = InstanceAssociator({})
        pose = lambda x: {"axes": {"x": {"degrees": x}, "y": {"degrees": 0}}}
        a = tracker.update([self.box(.7)], pose(0), 1)[0]
        b = tracker.update([self.box(.5)], pose(32), 1.2)[0]
        self.assertEqual(a["instance_id"], b["instance_id"])

    def test_ambiguous_crossing_does_not_arbitrarily_reassign_clicked_id(self):
        tracker = InstanceAssociator({})
        first = tracker.update([self.box(.48), self.box(.52)], None, 1)
        next_frame = tracker.update([self.box(.495), self.box(.505)], None, 1.2)
        self.assertTrue({b["instance_id"] for b in first}.isdisjoint(b["instance_id"] for b in next_frame))

    def test_ambiguity_recovers_instead_of_poisoning_every_future_frame(self):
        tracker = InstanceAssociator({})
        first = tracker.update([self.box(.48), self.box(.52)], None, 1)
        crossing = tracker.update([self.box(.495), self.box(.505)], None, 1.03)
        survivor = tracker.update([self.box(.5)], None, 1.06)[0]
        self.assertNotIn(survivor["instance_id"], [b["instance_id"] for b in first + crossing])
        for i in range(1, 61):
            current = tracker.update([self.box(.5)], None, 1.06+i/30)[0]
            self.assertEqual(current["instance_id"], survivor["instance_id"])
            self.assertEqual(len(tracker.tracks), 1)

    def test_nearby_objects_recover_stable_ids_after_separating(self):
        tracker = InstanceAssociator({})
        tracker.update([self.box(.48), self.box(.52)], None, 1)
        tracker.update([self.box(.495), self.box(.505)], None, 1.03)
        separated = tracker.update([self.box(.47), self.box(.53)], None, 1.06)
        for i in range(1, 31):
            current = tracker.update([self.box(.47), self.box(.53)], None, 1.06+i/30)
            self.assertEqual([b["instance_id"] for b in current], [b["instance_id"] for b in separated])

    def test_expired_lost_or_different_class_does_not_reuse_identity(self):
        tracker = InstanceAssociator({})
        first = tracker.update([self.box(.5)], None, 1)[0]
        tracker.update([], None, 1.1)
        brief = tracker.update([self.box(.5)], None, 1.2)[0]
        self.assertEqual(first["instance_id"], brief["instance_id"])
        other = tracker.update([self.box(.5, "bottle")], None, 1.3)[0]
        self.assertNotEqual(first["instance_id"], other["instance_id"])
        expired = tracker.update([self.box(.5)], None, 2.5)[0]
        self.assertNotEqual(first["instance_id"], expired["instance_id"])
        tracker.clear()
        reset = tracker.update([self.box(.5)], None, 2.6)[0]
        self.assertNotEqual(expired["instance_id"], reset["instance_id"])


if __name__ == "__main__":
    unittest.main()
