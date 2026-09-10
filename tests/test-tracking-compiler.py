"""Compiler routing contracts; real output agreement is qualified on the GPU."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from spring_turret import sam31_tracking_regions as regions


def module(name):
    def forward(*args, **kwargs): pass
    forward.module_name = name
    return NS(forward=forward)


class CompilerPolicyTests(unittest.TestCase):
    def install(self, device, **kwargs):
        compiled, stages, sharing = {}, {}, []
        def compile_fn(fn, **options):
            compiled[fn.module_name] = options
            return fn
        def stage(torch, fn, name, kind, **options):
            stages[name] = options
            return NS(name=name, **options)
        def detection(torch, model, region, *, share_image):
            sharing.append(share_image)
            region('image_and_detection',lambda:None,'inductor-image+aot_eager-heads')
            if share_image:
                region('additional_prompt_detection',lambda:None,'aot_eager-heads')
        detector = NS(backbone=NS(vision_backbone=module('image'),
            language_backbone=NS(encoder=module('text'))),
            transformer=NS(encoder=module('detection_encoder'),decoder=module('detection_decoder')),
            segmentation_head=module('detection_masks'), geometry_encoder=NS(_encode_boxes=lambda:None))
        tracker = NS(maskmem_backbone=module('memory_encoder'),
            transformer=NS(encoder=module('memory_attention')), sam_mask_decoder=module('tracking_masks'),
            _encode_new_memory=lambda:None)
        model = NS(detector=detector,tracker=NS(model=tracker))
        with patch.object(regions,'TrackingGraphStage',side_effect=stage), \
             patch.object(regions,'DetectionRegion',side_effect=detection), \
             patch.object(regions,'PropagationRegion',return_value=NS(prepare_memory=None,heads=None)), \
             patch.object(regions,'tensor_indexed_memory',return_value=lambda:None):
            regions.install_tracking_regions(NS(compile=compile_fn),model,device,**kwargs)
        return compiled, stages, sharing

    def test_cuda_uses_inductor_without_changing_replay_boundaries(self):
        compiled, stages, sharing = self.install('cuda')
        self.assertEqual(set(compiled),{'image','detection_encoder','detection_decoder','detection_masks'})
        for options in compiled.values():
            self.assertEqual(options['backend'],'inductor')
            self.assertTrue(options['fullgraph'])
            self.assertFalse(options['dynamic'])
            self.assertTrue(options['options']['emulate_precision_casts'])
            self.assertFalse(options['options']['triton.cudagraphs'])
        self.assertEqual(stages['text_encoder']['backend'],'aot_eager')
        self.assertEqual(stages['image_and_detection']['backend'],'inductor-image+inductor-heads')
        for name in ('memory_encoder','memory_attention','tracking_masks'):
            self.assertEqual(stages[name]['backend'],'inductor')
        self.assertEqual(stages['memory_attention']['cache_policy'], 'retain')
        self.assertEqual(stages['memory_attention']['max_variants'], 16)
        self.assertEqual(stages['memory_attention']['capture_repetitions'], 3)
        self.assertNotIn('cache_policy', stages['memory_encoder'])
        self.assertNotIn('memory_attention_and_mask',stages)
        self.assertEqual(sharing,[False])

    def test_xpu_policy_is_unchanged(self):
        compiled, stages, sharing = self.install('xpu')
        self.assertEqual(compiled['image']['backend'],'inductor')
        for name,options in compiled.items():
            if name != 'image':
                self.assertEqual(options,{'backend':'aot_eager','fullgraph':True,'dynamic':False})
        self.assertEqual(stages['text_encoder']['backend'],'aot_eager')
        self.assertEqual(stages['image_and_detection']['backend'],'inductor-image+aot_eager-heads')
        self.assertEqual(sharing,[True])

    def test_explicit_baseline_override(self):
        compiled, stages, _ = self.install('cuda',heads_backend='aot_eager')
        self.assertEqual(compiled['image']['backend'],'inductor')
        self.assertEqual(compiled['detection_encoder']['backend'],'aot_eager')
        self.assertEqual(stages['memory_attention']['backend'],'aot_eager')

    def test_unknown_backend_rejected(self):
        with self.assertRaisesRegex(ValueError,'compiler backend'):
            regions.install_tracking_regions(None,None,'cuda',heads_backend='unknown')


if __name__ == '__main__': unittest.main()
