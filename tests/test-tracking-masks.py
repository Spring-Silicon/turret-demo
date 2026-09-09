"""Display-mask contracts; no camera, GPU or servo interfaces."""
import base64
import colorsys
from contextlib import nullcontext
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spring_turret.tracking_masks import encode_mask_overlay, instance_color
from spring_turret.prompts import COLORS

IMAGE_DEPS = all(importlib.util.find_spec(n) is not None for n in ('numpy', 'PIL'))


class ColorTests(unittest.TestCase):
    def test_stable_distinct_shades_stay_in_class_color_family(self):
        def hue(color):
            return colorsys.rgb_to_hls(*(int(color[i:i+2],16)/255 for i in (1,3,5)))[0]
        for base in COLORS:
            colors = {i:instance_color(base,i) for i in range(1,129)}
            self.assertEqual(len(set(colors.values())),128)
            for i,color in reversed(list(colors.items())):
                self.assertEqual(instance_color(base,i),color)
                delta = abs(hue(color)-hue(base))
                self.assertLessEqual(min(delta,1-delta)*360,12.5)


@unittest.skipUnless(IMAGE_DEPS, 'Inference image dependencies not installed')
class MaskTests(unittest.TestCase):
    def decode(self, overlay):
        from PIL import Image
        return Image.open(io.BytesIO(base64.b64decode(overlay['png']))).convert('RGBA')

    def test_native_resolution_exact_masks_transparency_and_stable_overlap(self):
        import numpy as np
        large = np.zeros((12,20),dtype=bool); large[2:10,3:18] = True
        small = np.zeros_like(large); small[4:6,5:7] = True
        original = large.copy()
        instances = [(large,'#12ab34',1),(small,'#56cd78',2)]
        overlay = encode_mask_overlay(instances,20,12)
        rgba = np.asarray(self.decode(overlay))
        self.assertEqual(overlay['width'],20)
        self.assertEqual(overlay['height'],12)
        np.testing.assert_array_equal(rgba[:,:,3] > 0, large | small)
        np.testing.assert_array_equal(large,original)
        self.assertEqual(tuple(rgba[3,4]),(0x12,0xab,0x34,112))
        self.assertEqual(tuple(rgba[4,5]),(0x56,0xcd,0x78,112))
        self.assertEqual(encode_mask_overlay(list(reversed(instances)),20,12),overlay)
        self.assertIsNone(encode_mask_overlay([],20,12))

    def test_all_128_instances_and_display_payload_bound(self):
        import numpy as np
        labels = np.arange(128,dtype=np.uint8).reshape(8,16)
        instances = [(labels == i,instance_color(COLORS[i//16],i+1),i+1) for i in range(128)]
        rgba = np.asarray(self.decode(encode_mask_overlay(instances,16,8)))
        for i in range(128):
            self.assertTrue((rgba[i//16,i%16,:3] == [int(instances[i][1][j:j+2],16) for j in (1,3,5)]).all())
        noise = np.random.default_rng(0).integers(0,2,(512,512)).astype(bool)
        with patch('spring_turret.tracking_masks.MAX_PNG_BYTES',1000):
            small = encode_mask_overlay([(noise,'#12ab34',1)],512,512)
        self.assertLessEqual(len(base64.b64decode(small['png'])),1000)
        self.assertLess(small['width'],512)
        self.assertEqual(small['source_width'],512)

    def test_invalid_mask_shape_or_type_is_rejected(self):
        import numpy as np
        with self.assertRaises(ValueError): encode_mask_overlay([(np.zeros((1,1),bool),'#ffffff',1)],2,2)
        with self.assertRaises(ValueError): encode_mask_overlay([(np.zeros((2,2),float),'#ffffff',1)],2,2)

    def test_worker_exports_only_visible_masks_without_extra_model_pass(self):
        import numpy as np
        from PIL import Image
        from spring_turret.sam31_tracking_worker import TrackingEngine
        class Pixels:
            def __sub__(self,v): return self
            def __truediv__(self,v): return self
            def to(self,v): return self
        tensor = Pixels()
        transform = SimpleNamespace(resize=lambda *a:None,to_tensor=lambda *a:tensor)
        engine = TrackingEngine.__new__(TrackingEngine)
        engine.torch = SimpleNamespace(inference_mode=nullcontext,autocast=lambda *a,**kw:nullcontext(),
                                      bfloat16='bf16',__version__='test')
        engine.device = SimpleNamespace(type='xpu')
        engine.accelerator = SimpleNamespace(synchronize=lambda *a:None,get_device_name=lambda *a:'test')
        engine.confidence,engine.graph_stages,engine.sources = .5,{},{}
        engine.ids,engine.next_id = {},1
        class Preprocess:
            validated = True
            def __call__(self, jpeg, *, prepared=None): return tensor, (8,8)
        engine.preprocess = Preprocess()
        engine.device_name, engine.source_count, engine.source_digest = 'test', -1, None
        engine.last_captured_at,engine.previous_duration = None,0
        engine.session_key = (('table',),8,8,None,None)
        masks = np.zeros((3,8,8),dtype=bool)
        masks[0,1:3,1:3]=True; masks[1,4:6,4:6]=True; masks[2,6:8,6:8]=True
        calls = []
        class Session:
            index = 2
            def step(self,pixels,**kwargs):
                calls.append(pixels)
                return {'active_ids':[1,2,3], 'propagated_ids':[1,2], 'memory_frames':4,
                    'retired_ids':[], 'dropped_objects':0,
                    'out_obj_ids':[1,2,3], 'out_probs':[.9,.8,.1],
                    'out_boxes_xywh':[[.125,.125,.25,.25],[.5,.5,.25,.25],[.75,.75,.25,.25]],
                    'out_binary_masks':masks}
        engine.sessions = [Session()]
        jpeg = io.BytesIO(); Image.new('RGB',(8,8)).save(jpeg,format='JPEG')
        with patch.dict(sys.modules,{'torchvision.transforms':SimpleNamespace(functional=transform)}):
            result = engine.detect({'jpeg':jpeg.getvalue(),'prompts':['table']})
        self.assertEqual(len(calls),1)
        self.assertEqual(result['categories'][0]['count'],2)
        self.assertEqual(result['categories'][0]['color'],COLORS[0])
        self.assertNotEqual(result['boxes'][0]['color'],result['boxes'][1]['color'])
        self.assertIn('mask_overlay_ms',result['timing'])
        rgba = np.asarray(self.decode(result['mask_overlay']))
        np.testing.assert_array_equal(rgba[:,:,3] > 0, masks[0] | masks[1])


if __name__ == '__main__': unittest.main()
