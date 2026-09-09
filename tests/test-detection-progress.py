"""Progress is not a replacement for a completed, paired tracking result."""
import os
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, os.environ.get("TURRET_TEST_SRC", str(Path(__file__).resolve().parents[1] / "src")))
from spring_turret.detection import DetectionController


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.c = DetectionController({"enabled": True, "model": "sam3.1-tracking"}, None)
        self.c.set_prompts(["light"])

    def result(self):
        self.c.state = "running"
        self.c.completed_at = time.monotonic() - 20
        self.c.result = {"frame_url": "/api/detection/frame/1-8.jpg", "frame_sequence": 8,
                         "boxes": [], "mask_overlay": {"png": "paired-mask"}}

    def test_initial_capture_has_progress_but_no_completed_result(self):
        self.c._progress(self.c.revision, "compiling_tracker_text_encoder")
        self.assertEqual(self.c.status()["state"], "compiling_tracker_text_encoder")
        self.assertFalse(self.c.result)

    def test_recapture_preserves_exact_result_even_when_slow_or_zero_objects(self):
        self.result()
        result = self.c.result
        for stage in ("compiling_tracker_memory_update", "compiling_tracker_memory_attention_and_mask"):
            self.c._progress(self.c.revision, stage)
            status = self.c.status()
            self.assertEqual(status["state"], "running")
            self.assertEqual(status["progress_stage"], stage)
            self.assertEqual(status["frame_sequence"], 8)
            self.assertGreater(status["frame_age_ms"], 19000)
            self.assertIs(self.c.result, result)

    def test_prompt_change_drops_old_result_and_ignores_obsolete_progress(self):
        self.result()
        old = self.c.revision
        self.c._progress(old, "compiling_tracker_memory_update")
        self.c.set_prompts(["cup"])
        self.c._progress(old, "compiling_tracker_memory_update")
        self.assertEqual(self.c.state, "loading")
        self.assertIsNone(self.c.progress_stage)
        self.assertFalse(self.c.result)

    def test_camera_loss_is_not_hidden_as_graph_compilation(self):
        self.result()
        self.c._progress(self.c.revision, "waiting_for_camera")
        self.assertEqual(self.c.state, "waiting_for_camera")


if __name__ == "__main__":
    unittest.main()
