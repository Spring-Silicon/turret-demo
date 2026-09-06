#!/usr/bin/env python3
"""CPU-only native bundle, protocol and model-routing contract tests."""

import copy
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spring_turret.sam31_native import (
    ARTIFACT, CHECKPOINT, PINNED_FILES, INPUT_BYTES, OUTPUT_BYTES,
    INPUT_SHAPE, OUTPUT_SHAPE, NativeImageStage, validate_handshake,
    validate_outputs, verify_bundle,
)
from spring_turret.detection import validate_config, WorkerClient


class NativeTests(unittest.TestCase):
    def handshake(self):
        return {"schema": "spring.graphs-runner-shared-memory.v2",
                "transport": "shared_memory", "artifact": ARTIFACT,
                "command_graph_count": 1,
                "input_bytes": INPUT_BYTES, "output_bytes": 2 * OUTPUT_BYTES,
                "inputs": [{"slot": "input.459", "shape": INPUT_SHAPE, "dtype": "f32", "byte_len": INPUT_BYTES}],
                "outputs": [{"slot": f"output.{i}", "dtype": "f16", "shape": OUTPUT_SHAPE,
                             "byte_len": OUTPUT_BYTES} for i in range(2)]}

    def test_handshake_and_response_contract(self):
        message = self.handshake()
        validate_handshake(message)
        outputs = [{k: v for k, v in value.items() if k != "dtype"} for value in message["outputs"]]
        validate_outputs({"outputs": outputs, "command_graph_count": 1})
        for key in ("schema", "artifact", "input_bytes", "output_bytes", "inputs", "outputs", "command_graph_count"):
            bad = copy.deepcopy(message)
            bad.pop(key)
            with self.assertRaises(ValueError):
                validate_handshake(bad)
        for key, value in (("shape", [1]), ("dtype", "f32"), ("slot", "output.1"), ("byte_len", 1)):
            bad = copy.deepcopy(message)
            bad["outputs"][0][key] = value
            with self.assertRaises(ValueError):
                validate_handshake(bad)
        with self.assertRaises(ValueError):
            validate_outputs({"outputs": outputs[::-1], "command_graph_count": 1})
        with self.assertRaises(ValueError):
            validate_outputs({"outputs": outputs, "command_graph_count": 0})

    def test_pinned_bundle_rejects_changed_runtime_and_checkpoint(self):
        bundle, checkpoint = Path("/bundle"), Path("/checkpoint")
        def checksum(path):
            return CHECKPOINT if path == checkpoint else PINNED_FILES[str(path.relative_to(bundle))]
        with patch("spring_turret.sam31_native.digest", side_effect=checksum):
            verify_bundle(bundle, checkpoint)
        with patch("spring_turret.sam31_native.digest", return_value="wrong"):
            with self.assertRaises(ValueError):
                verify_bundle(bundle, checkpoint)
        for name in PINNED_FILES:
            def corrupt(path):
                return "wrong" if path == bundle / name else checksum(path)
            with patch("spring_turret.sam31_native.digest", side_effect=corrupt):
                with self.assertRaises(ValueError):
                    verify_bundle(bundle, checkpoint)

    def test_reader_handles_short_reads_eof_and_timeout(self):
        read, write = os.pipe()
        stage = NativeImageStage.__new__(NativeImageStage)
        with os.fdopen(read, "rb", buffering=0) as stream:
            stage.process = SimpleNamespace(stdout=stream)
            os.write(write, b"hello")
            self.assertEqual(stage._read(5, time.monotonic() + 1), b"hello")
            with self.assertRaises(TimeoutError):
                stage._read(1, time.monotonic() + .01)
            os.close(write)
            with self.assertRaises(RuntimeError):
                stage._read(1, time.monotonic() + 1)

    def test_only_sam_receives_native_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"enabled": True, "python": "/python", "checkpoint": "/weights",
                      "cache_dir": directory, "sam31_native_bundle": "/bundle"}
            validate_config(config)
            with self.assertRaises(ValueError):
                validate_config({**config, "precision": "bfloat16"})
            with self.assertRaises(ValueError):
                validate_config({**config, "sam31_native_bundle": "relative"})
            with patch("spring_turret.detection.subprocess.Popen") as launch, patch(
                "spring_turret.detection.tempfile.TemporaryDirectory"
            ) as temp:
                temp.return_value.name = directory
                WorkerClient(config).launch()
                self.assertIn("--native-bundle", launch.call_args.args[0])
                self.assertTrue(launch.call_args.kwargs["start_new_session"])
                WorkerClient({**config, "model": "yolo26x"}).launch()
                self.assertNotIn("--native-bundle", launch.call_args.args[0])

    def test_cancel_signals_entire_worker_group(self):
        worker = WorkerClient({})
        worker.process = MagicMock(pid=1234)
        worker.process.poll.return_value = None
        with patch("spring_turret.detection.os.killpg") as kill:
            worker.cancel()
            kill.assert_called_once_with(1234, 15)


if __name__ == "__main__":
    unittest.main()
