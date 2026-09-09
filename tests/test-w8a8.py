#!/usr/bin/env python3
"""No GPU needed: W8A8 source identity, safe helper loading and isolated routing."""
import ast
from contextlib import contextmanager
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
    def test_packed_patches_end_before_reference_and_are_not_reentered_for_replay(self):
        events = []
        @contextmanager
        def context(library):
            events.append("patched")
            try:
                yield {"invocations": [None] * 18}
            finally:
                events.append("restored")
        value = SimpleNamespace(clone=lambda: value)
        stage = w8.PackedHeadStage.__new__(w8.PackedHeadStage)
        stage.cast_cache_enabled = False
        stage.library, stage.graph, stage.inputs = Path("/kernel"), None, (value,)
        stage.torch = SimpleNamespace(equal=lambda a, b: True,
                                     isfinite=lambda x: SimpleNamespace(all=lambda: True))
        stage.compiled = lambda *args: (value,)
        def run(*args):
            stage.graph = object()
            return (value,)
        with patch.dict(sys.modules, {"packed_head_attention": SimpleNamespace(packed_heads=context)}), \
                patch.object(w8.CompiledStage, "__call__", side_effect=run):
            stage(value)
            self.assertTrue(stage.direct_replay_passed)
            self.assertEqual(events, ["patched", "restored"])
            stage(value)
            self.assertEqual(events, ["patched", "restored"])
        stage.graph = None
        with patch.dict(sys.modules, {"packed_head_attention": SimpleNamespace(packed_heads=context)}), \
                patch.object(w8.CompiledStage, "__call__", side_effect=RuntimeError("compile failed")):
            with self.assertRaises(RuntimeError):
                stage(value)
        self.assertEqual(events[-2:], ["patched", "restored"])

    def test_packed_extension_is_pinned_without_breaking_original_bundle(self):
        root, checkpoint = Path("/bundle"), Path("/checkpoint")
        files = {**w8.PINNED_FILES, **w8.PACKED_FILES}
        files.update({f"runtime/graphics/{name}": value for name, value in w8.GRAPHICS_FILES.items()})
        def digest(path):
            return w8.CHECKPOINT if path == checkpoint else files[str(path.relative_to(root))]
        with patch.object(w8, "packed_profile", return_value=True), patch.object(w8, "digest", side_effect=digest):
            w8.verify_bundle(root, checkpoint)
        for name in {*w8.PACKED_FILES, *(f"runtime/graphics/{n}" for n in w8.GRAPHICS_FILES)}:
            with self.subTest(name=name), patch.object(w8, "packed_profile", return_value=True), \
                    patch.object(w8, "digest", side_effect=lambda p: "bad" if p == root/name else digest(p)):
                with self.assertRaises(ValueError):
                    w8.verify_bundle(root, checkpoint)

    def test_packed_profile_rejects_unsupported_batch_before_loading_gpu(self):
        with patch("spring_turret.sam31_worker.packed_profile", return_value=True):
            for batch in (2, 4, 8):
                with self.assertRaisesRegex(ValueError, "require batch size 1"):
                    Sam31Engine(Path("/checkpoint"), 0, "float16", .5, True, batch,
                                w8a8_development_bundle=Path("/bundle"))

    def test_cast_cache_requires_packed_base_and_pins_every_new_artifact(self):
        root, checkpoint = Path("/bundle"), Path("/checkpoint")
        files = {**w8.PINNED_FILES, **w8.PACKED_FILES, **w8.CAST_CACHE_FILES}
        files.update({f"runtime/graphics/{n}": v for n, v in w8.GRAPHICS_FILES.items()})
        def digest(path):
            return w8.CHECKPOINT if path == checkpoint else files[str(path.relative_to(root))]
        with patch.object(w8, "cast_cache_profile", return_value=True), \
                patch.object(w8, "packed_profile", return_value=False), \
                patch.object(w8, "digest", side_effect=digest):
            with self.assertRaisesRegex(ValueError, "require the retained packed bundle"):
                w8.verify_bundle(root, checkpoint)
        with patch.object(w8, "cast_cache_profile", return_value=True), \
                patch.object(w8, "packed_profile", return_value=True):
            with patch.object(w8, "digest", side_effect=digest):
                w8.verify_bundle(root, checkpoint)
            for name in w8.CAST_CACHE_FILES:
                with self.subTest(name=name), patch.object(w8, "digest", side_effect=
                        lambda p: "bad" if p == root/name else digest(p)):
                    with self.assertRaises(ValueError):
                        w8.verify_bundle(root, checkpoint)

    def test_cast_cache_setup_errors_propagate_and_restore_patches(self):
        events = []
        @contextmanager
        def context(library):
            events.append("patched")
            try:
                yield {"invocations": [None] * 18}
            finally:
                events.append("restored")
        stage = w8.PackedHeadStage.__new__(w8.PackedHeadStage)
        stage.library, stage.graph, stage.cast_cache_enabled = Path('/kernel'), None, True
        with patch.dict(sys.modules, {'packed_head_attention': SimpleNamespace(packed_heads=context)}), \
                patch.object(stage, '_capture_cached', side_effect=RuntimeError('unknown program')), \
                patch.object(w8.CompiledStage, '__call__') as fallback:
            with self.assertRaisesRegex(RuntimeError, 'unknown program'):
                stage(object())
            fallback.assert_not_called()
            self.assertIsNone(stage.graph)
        self.assertEqual(events, ['patched', 'restored'])
        stage.graph = object()
        with patch.object(stage, '_capture_cached') as prepare, \
                patch.object(w8.CompiledStage, '__call__') as replay:
            stage(object())
            prepare.assert_not_called()
            replay.assert_called_once()

    def test_packed_graphics_are_pinned_only_for_sam_and_use_a_separate_cache(self):
        with tempfile.TemporaryDirectory() as cache, \
                patch.object(w8, "packed_profile", return_value=True), \
                patch.object(w8, "verify_graphics", return_value=Path("/bundle/runtime/graphics")) as verify, \
                patch("spring_turret.detection.subprocess.Popen") as launch:
            config = {"enabled":True, "python":"/inference/venv/bin/python", "checkpoint":"/weights",
                      "cache_dir":cache, "sam31_w8a8_development_bundle":"/bundle"}
            WorkerClient(config).launch()
            env = launch.call_args.kwargs["env"]
            self.assertTrue(env["LD_LIBRARY_PATH"].startswith("/bundle/runtime/graphics:/inference/venv/lib:"))
            self.assertIn("sam31-israel-w8a8-packed", env["TORCHINDUCTOR_CACHE_DIR"])
            verify.assert_called_once_with(Path("/bundle"))
            WorkerClient({**config,"model":"sam3.1-mask"}).launch()
            self.assertNotIn("/bundle/runtime/graphics", launch.call_args.kwargs["env"].get("LD_LIBRARY_PATH", ""))
            verify.assert_called_once()

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
                WorkerClient({**config, "model": "sam3.1-mask", "sam31_allow_unqualified_w8a8": True}).launch()
                self.assertNotIn("--w8a8-development-bundle", launch.call_args.args[0])
                self.assertNotIn("--allow-unqualified-w8a8", launch.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
