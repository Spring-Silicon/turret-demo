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

    def test_encoder_pose_survives_inference_delay_and_is_not_resampled_at_completion(self):
        c, worker = self.controller()
        pose = {"sampled_at": time.monotonic(), "axes": {
            "x": {"degrees": 10, "goal_degrees": 40}, "y": {"degrees": 5, "goal_degrees": 5}}}
        c.pose_provider = lambda: pose
        c.set_prompt("cup")
        self.assertTrue(worker.entered.wait(1))
        # The capture pose is a snapshot. Inference must not replace it with a
        # later pose after the motors have moved during model execution.
        pose = {**pose, "axes": {**pose["axes"], "x": {"degrees": 40, "goal_degrees": 40}}}
        worker.release.set()
        eventually(lambda: c.status()["state"] == "running")
        self.assertEqual(c.status()["frame_pose"]["axes"]["x"]["degrees"], 10)
        self.assertGreaterEqual(c.status()["captured_at"], c.status()["frame_pose"]["sampled_at"])

    def test_old_pose_is_not_paired_with_a_later_camera_frame(self):
        c, worker = self.controller()
        c.pose_provider = lambda: {"sampled_at": time.monotonic() - 1}
        c.set_prompt("cup")
        worker.release.set()
        eventually(lambda: c.status()["state"] == "running")
        self.assertIsNone(c.status()["frame_pose"])

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
        c.frame_selections[f"{c.revision}-777"]["captured_at"] -= 1
        with self.assertRaises(ValueError): c.selection(c.revision, 777, 42)


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
