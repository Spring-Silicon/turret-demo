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


class DetectionTests(unittest.TestCase):
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
        self.assertNotIn("frame_url", c.status())
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
        for target in ("person", "cup", None):
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


if __name__ == "__main__":
    unittest.main()
