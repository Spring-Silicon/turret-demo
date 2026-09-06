#!/usr/bin/env python3
"""No GPU needed: W8A8 source identity, safe helper loading and isolated routing."""
import ast
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spring_turret import sam31_w8a8 as w8
from spring_turret.detection import validate_config, WorkerClient
from spring_turret.sam31_worker import Sam31Engine, check_detection_candidate


class W8A8Tests(unittest.TestCase):
    def test_only_explicit_development_mode_can_report_accuracy_failures(self):
        values = [SimpleNamespace(shape=(1, 3), dtype="float32")] * 3
        with patch("spring_turret.sam31_worker.validate_detections", side_effect=AssertionError("confidence exceeded .03")):
            with self.assertRaises(AssertionError):
                check_detection_candidate(None, values, values, .5)
            result = check_detection_candidate(None, values, values, .5, report_only=True)
            self.assertFalse(result["passed"])
            self.assertEqual(result["policy"], "report-only-development")
            self.assertIn(".03", result["error"])
        for error in (RuntimeError("non-finite values"), ValueError("invalid outputs")):
            with patch("spring_turret.sam31_worker.validate_detections", side_effect=error):
                with self.assertRaises(type(error)):
                    check_detection_candidate(None, values, values, .5, report_only=True)
        bad = [SimpleNamespace(shape=(1, 4), dtype="float32")] * 3
        with self.assertRaisesRegex(RuntimeError, "shape/dtype"):
            check_detection_candidate(None, values, bad, .5, report_only=True)
        with self.assertRaises(ValueError):
            Sam31Engine(Path("/weights"), 0, "float16", .5, True, allow_unqualified_w8a8=True)

    def test_every_source_and_tensor_is_pinned(self):
        root, checkpoint = Path("/bundle"), Path("/checkpoint")
        def digest(path):
            return w8.CHECKPOINT if path == checkpoint else w8.PINNED_FILES[str(path.relative_to(root))]
        with patch.object(w8, "digest", side_effect=digest):
            w8.verify_bundle(root, checkpoint)
        for name in w8.PINNED_FILES:
            with self.subTest(name=name), patch.object(w8, "digest", side_effect=lambda p: "bad" if p == root/name else digest(p)):
                with self.assertRaises(ValueError):
                    w8.verify_bundle(root, checkpoint)
        with patch.object(w8, "digest", return_value="bad"), self.assertRaises(ValueError):
            w8.verify_bundle(root, checkpoint)

    def test_helper_extraction_does_not_execute_harness_or_import_paths(self):
        source = 'raise RuntimeError("top level must not run")\n'
        source += '\n'.join(f'def {name}(): return 1\n' for name in w8.HELPERS)
        source += '\ndef main(): raise RuntimeError("CLI must not run")\nmain()\n'
        tree = w8.helper_definitions(source)
        self.assertTrue(all(isinstance(n, ast.FunctionDef) for n in tree.body))
        scope = {}
        exec(compile(tree, "test", "exec"), scope)
        self.assertNotIn("main", scope)
        for name in w8.HELPERS:
            self.assertEqual(scope[name](), 1)
        with self.assertRaises(ValueError):
            w8.helper_definitions("def persistent_forward(): pass")

    def test_rejects_previously_imported_unpinned_sam(self):
        with patch.object(w8, "verify_bundle"), patch.dict(sys.modules, {"sam3": SimpleNamespace(__file__="/different/sam3/__init__.py")}):
            with self.assertRaisesRegex(ValueError, "pinned SAM source"):
                w8.configure_source(Path("/bundle"), Path("/checkpoint"))

    def test_w8a8_is_explicit_sam_only_and_uses_separate_cache_and_torch_libraries(self):
        key = "sam31_w8a8_development_bundle"
        with tempfile.TemporaryDirectory() as cache:
            config = {"enabled": True, "python": "/inference/venv/bin/python", "checkpoint": "/weights",
                      "cache_dir": cache, key: "/bundle"}
            validate_config(config)
            validate_config({**config, "sam31_allow_unqualified_w8a8": True})
            for bad in ({key: "relative"}, {"precision": "bfloat16"}, {"sam31_native_bundle": "/dense"}):
                with self.assertRaises(ValueError):
                    validate_config({**config, **bad})
            with self.assertRaises(ValueError):
                validate_config({**config, "sam31_allow_unqualified_w8a8": "true"})
            with self.assertRaises(ValueError):
                validate_config({k:v for k,v in {**config, "sam31_allow_unqualified_w8a8": True}.items() if k != key})
            with patch("spring_turret.detection.subprocess.Popen") as launch:
                WorkerClient(config).launch()
                self.assertIn("--w8a8-development-bundle", launch.call_args.args[0])
                env = launch.call_args.kwargs["env"]
                self.assertIn("sam31-israel-w8a8", env["TORCHINDUCTOR_CACHE_DIR"])
                self.assertTrue(env["LD_LIBRARY_PATH"].startswith("/inference/venv/lib:"))
                self.assertNotIn("--allow-unqualified-w8a8", launch.call_args.args[0])
                WorkerClient({**config, "sam31_allow_unqualified_w8a8": True}).launch()
                self.assertIn("--allow-unqualified-w8a8", launch.call_args.args[0])
                WorkerClient({**config, "model": "yolo26x", "sam31_allow_unqualified_w8a8": True}).launch()
                self.assertNotIn("--w8a8-development-bundle", launch.call_args.args[0])
                self.assertNotIn("--allow-unqualified-w8a8", launch.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
