"""Mask model routing and discrete output semantics without hardware access."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from spring_turret.models import model_available, model_prompts
from spring_turret.detection import DetectionController, WorkerClient, validate_config


class MaskProfileTests(unittest.TestCase):
    def config(self,device):
        return {'enabled':True,'python':'/bin/python','checkpoint':'/model.pt','cache_dir':'/tmp/cache',
                'device_type':device,'model':'sam3.1-mask','precision':'float16'}

    def test_distinct_profile_and_availability(self):
        cuda=self.config('cuda'); xpu=self.config('xpu')
        validate_config(cuda)
        self.assertTrue(model_available('sam3.1-mask',cuda))
        self.assertFalse(model_available('sam3.1-mask',xpu))
        with self.assertRaises(ValueError):validate_config(xpu)
        xpu['sam31_mask_bundle']='/frozen/masks';validate_config(xpu)
        self.assertEqual(model_prompts('sam3.1-mask',['cup','hand']),['cup','hand'])
        cuda['sam31_mask_bundle']='/intel-only'
        with self.assertRaises(ValueError):validate_config(cuda)

    def test_prompt_state_exists_for_new_model(self):
        controller=DetectionController(self.config('cuda'),object())
        controller.set_model('sam3.1-mask')
        controller.set_prompts(['cup'])
        controller.set_model('sam3.1');controller.set_prompts(['person'])
        controller.set_model('sam3.1-mask')
        self.assertEqual(controller.prompts,['cup'])

    def test_worker_uses_own_bundle_and_cache(self):
        for device in ('xpu','cuda'):
            with TemporaryDirectory() as directory,patch('spring_turret.detection.subprocess.Popen') as popen:
                config=self.config(device);config['cache_dir']=directory
                if device=='xpu':config['sam31_mask_bundle']='/frozen/masks'
                worker=WorkerClient(config);worker.launch()
                command=popen.call_args.args[0];env=popen.call_args.kwargs['env']
                self.assertTrue(any(str(v).endswith('/sam31_mask_worker.py') for v in command))
                self.assertEqual(command[command.index('--device-type')+1],device)
                self.assertEqual('--mask-bundle' in command,device=='xpu')
                self.assertIn('sam31-mask-20260908',env['TORCHINDUCTOR_CACHE_DIR'])


if __name__=='__main__':unittest.main()
