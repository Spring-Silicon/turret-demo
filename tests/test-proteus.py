#!/usr/bin/env python3
"""CPU contracts for experimental model routing and truthful capability flags."""
import ast
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from spring_turret.models import PROTEUS_MODELS, is_session_model, is_tracking_model, model_available, model_prompts
from spring_turret.detection import validate_config, validate_tracking_result
from spring_turret.hardware import inference_runtime
from spring_turret.proteus_models import construction
from spring_turret.proteus_worker import CurrentRawFrame, memory_frames


class ProteusTests(unittest.TestCase):
    def test_only_configured_hardware_exposes_experiments(self):
        for model in PROTEUS_MODELS:
            cfg = dict(enabled=True, model=model, python='/python', checkpoint='/weights',
                       cache_dir='/cache', proteus_bundle='/bundle', proteus_memory_bundle='/memory')
            validate_config(cfg)
            self.assertTrue(model_available(model, cfg))
            self.assertTrue(is_session_model(model))
            self.assertEqual(is_tracking_model(model), model in ('efficient-tracking', 'efficient-memory'))
            self.assertFalse(model_available(model, {**cfg, 'device_type':'cuda'}))
            for change in ({'proteus_bundle':'relative'}, {'proteus_bundle':None}, {'device_type':'cuda'}):
                with self.assertRaises(ValueError):
                    validate_config({**cfg, **change})

    def test_no_silent_multi_prompt_truncation(self):
        for model in ('efficient-nomem', 'hybrid-nomem', 'efficient-memory'):
            self.assertEqual(model_prompts(model, ['ball']), ['ball'])
            with self.assertRaisesRegex(ValueError, 'remove the extra'):
                model_prompts(model, ['ball', 'person'])
        self.assertEqual(model_prompts('efficient-tracking', ['ball','person']), ['ball','person'])

    def test_private_runtime_does_not_replace_regular_python(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = dict(enabled=True, model='hybrid-nomem', python='/existing/python',
                       checkpoint='/weights', cache_dir=directory, proteus_bundle=directory)
            private = Path(directory) / 'proteus-composed-b580-20260907/private-host/usr'
            for path in (private/'bin/python3.12', private/'lib/x86_64-linux-gnu/ld-linux-x86-64.so.2'):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            spec = inference_runtime(cfg).prepare()
            self.assertNotIn('/existing/python', spec.command)
            self.assertIn('--profile', spec.command)
            self.assertEqual(spec.command[spec.command.index('--profile')+1], 'hybrid-nomem')
            self.assertEqual(spec.environment['ONEAPI_DEVICE_SELECTOR'], 'level_zero:0')
            self.assertIn('compiler/2025.3', spec.environment['CXX'])

    def test_thor_no_memory_uses_existing_compiled_image_mask_path(self):
        self.assertFalse(is_session_model('sam3.1-nomem'))
        self.assertFalse(is_tracking_model('sam3.1-nomem'))
        with tempfile.TemporaryDirectory() as directory:
            cfg = dict(model='sam3.1-nomem', device_type='cuda', python='/python',
                       checkpoint='/weights', cache_dir=directory, sam31_tracking_bundle='/sam')
            spec = inference_runtime(cfg).prepare()
            self.assertTrue(any('sam31_mask_worker.py' in arg for arg in spec.command))
            self.assertNotIn('/sam', spec.command)
            self.assertEqual(spec.environment['TORCHINDUCTOR_FREEZING'], '1')
            self.assertNotIn('PYTHONHOME', spec.environment)

    def test_no_memory_cannot_claim_temporal_memory(self):
        result = dict(temporal_tracking=False, tracking_backend='proteus-efficient-nomem',
                      tracking_frame=1, memory_frames=0, boxes=[], active_instance_ids=[])
        validate_tracking_result(result, ['ball'], temporal=False, backend='proteus-efficient-nomem')
        for extra in ({'memory_frames':1}, {'temporal_tracking':True}, {'tracking_backend':'sam31-object-multiplex'}):
            with self.assertRaises(RuntimeError):
                validate_tracking_result({**result, **extra}, ['ball'], temporal=False, backend='proteus-efficient-nomem')
        with self.assertRaises(RuntimeError):
            validate_tracking_result(result, ['ball'])

    def test_memory_requires_its_own_source_and_actual_encoded_history(self):
        cfg = dict(enabled=True, python='/python', checkpoint='/weights', cache_dir='/cache',
                   model='efficient-memory', proteus_bundle='/bundle')
        self.assertFalse(model_available('efficient-memory', cfg))
        with self.assertRaises(ValueError):
            validate_config(cfg)
        for invalid in (None, '', 'relative'):
            with self.assertRaises(ValueError):
                validate_config({**cfg, 'proteus_memory_bundle':invalid})
        encoded = {'maskmem_features':object(), 'maskmem_pos_enc':[object()]}
        session = {'output_dict':{'cond_frame_outputs':{0:encoded}, 'non_cond_frame_outputs':{1:encoded}}}
        self.assertEqual(memory_frames(SimpleNamespace(num_maskmem=7), session, True), 2)
        with self.assertRaisesRegex(RuntimeError, 'produced temporal memory'):
            memory_frames(SimpleNamespace(num_maskmem=0), session, False)
        with self.assertRaisesRegex(RuntimeError, 'failed to produce'):
            memory_frames(SimpleNamespace(num_maskmem=7), {'output_dict':{'cond_frame_outputs':{}}}, True)

    def test_pinned_builder_never_executes_cli_or_postprocessing(self):
        source = b'def main():\n    raise RuntimeError("admission")\n    model = "old/path"\n    ready = True\n    frames = forbidden()\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'snapshot.py'
            path.write_bytes(source)
            result = construction(path, hashlib.sha256(source).hexdigest(), 'model =', 'frames =', {}, {'old/path':'new/path'})
            self.assertEqual(result, {**result, 'model':'new/path', 'ready':True})
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                construction(path, '0'*64, 'model =', 'frames =', {}, {})

    def test_only_current_frame_is_available(self):
        image = object()
        frame = CurrentRawFrame(image, 7)
        self.assertIs(frame[7], image)
        for index in (0, 6, 8):
            with self.assertRaises(IndexError):
                frame[index]


if __name__ == '__main__':
    unittest.main()
