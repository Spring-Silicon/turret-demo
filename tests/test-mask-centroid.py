"""Centroid geometry, worker transport and simulated control; no hardware access."""
import copy
from contextlib import nullcontext
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

# PYTHONPATH may point at a staged deployment instead of this checkout.
if not importlib.util.find_spec('spring_turret'):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from spring_turret.tracking import TrackingController, nearest_box, tracking_point
from spring_turret.tracking_masks import mask_centroid
from spring_turret.detection import validate_tracking_result

IMAGE_DEPS=all(importlib.util.find_spec(m) is not None for m in ('numpy','PIL'))

def box(point=None, *, masked=True):
    result={'prompt':'person','score':.9,'xyxy':[.2,.2,.8,.8],'instance_id':7}
    if masked:result['mask_centroid']=point
    return result


class ControlTests(unittest.TestCase):
    def test_box_fallback_only_when_no_mask_is_supplied(self):
        self.assertEqual(tracking_point(box(masked=False)),(.5,.5))
        self.assertEqual(tracking_point(box([.2,.75])),(.2,.75))
        for bad in (None,[],[.2],['.2',.5],[True,.5],[float('nan'),.5],[-.1,.5],[1.1,.5]):
            self.assertIsNone(tracking_point(box(bad)))
            self.assertIsNone(nearest_box([box(bad)],'person',1280,720))

    def test_acquisition_uses_centroid_pixel_distance(self):
        a,b=box([.6,.5]),box([.5,.65])
        b['instance_id']=8
        self.assertIs(nearest_box([a,b],'person',1280,720),b)
        self.assertIsNone(nearest_box([a,b],'cup',1280,720))

    def make_controller(self, point):
        pose={'sampled_at':9.95,'axes':{a:{'degrees':0.,'goal_degrees':0.} for a in ('x','y')}}
        data={'enabled':True,'state':'running','prompts':['person'],'model':'sam3.1-tracking',
              'revision':1,'frame_sequence':1,'frame_age_ms':40,'captured_at':9.96,
              'frame_pose':pose,'boxes':[box(point)],'temporal_tracking':True,'active_instance_ids':[7]}
        detection=SimpleNamespace(status=lambda:copy.deepcopy(data))
        state={'armed':True,'online':True,'axes':{a:{'goal_degrees':0.} for a in ('x','y')}}
        servo=SimpleNamespace(status=lambda:copy.deepcopy(state),
            point=Mock(return_value={'goal_degrees':{'x':12.,'y':-9.},'limited':False}),track=Mock())
        camera=SimpleNamespace(status=lambda:{'online':True,'width':1280,'height':720})
        controller=TrackingController({'calibrated':True},detection,servo,camera)
        controller.target='person';controller.ignore_before=0.
        return controller,data,servo

    def test_centroid_drives_both_axes_even_with_box_centered(self):
        controller,data,servo=self.make_controller([.75,.25])
        with patch('spring_turret.tracking.time.monotonic',return_value=10.):controller._tick()
        self.assertEqual(servo.point.call_args.args[0],{'x':40.,'y':22.5})
        self.assertEqual(controller.error_pixels,[320.,-180.])
        self.assertEqual(controller.instance_id,7)

    def test_calibrated_geometry_receives_full_frame_centroid_pixels(self):
        controller,data,servo=self.make_controller([.75,.25])
        controller.geometry=SimpleNamespace(check_binding=Mock(),world_direction=Mock(return_value=(0,0,1)),
            solve_direction=Mock(return_value={'goals':{'x':12.,'y':-9.},'limited':False}))
        with patch('spring_turret.tracking.time.monotonic',return_value=10.):controller._tick()
        self.assertEqual(controller.geometry.world_direction.call_args.args[:2],(960.,180.))
        self.assertEqual(servo.point.call_args.args[0],{'x':12.,'y':-9.})

    def test_empty_mask_does_not_command_a_box_center(self):
        controller,data,servo=self.make_controller(None)
        with patch('spring_turret.tracking.time.monotonic',return_value=10.):controller._tick()
        servo.point.assert_not_called()

    def test_worker_centroid_validation(self):
        result={'temporal_tracking':True,'tracking_backend':'sam31-object-multiplex',
                'tracking_frame':1,'memory_frames':1,'active_instance_ids':[7],'boxes':[box([.2,.7])]}
        validate_tracking_result(result,['person'])
        validate_tracking_result({**result,'boxes':[box(None)]},['person'])
        for bad in ([float('inf'),.5],[True,.5],[.5,2.],[]):
            with self.assertRaisesRegex(RuntimeError,'centroid'):
                validate_tracking_result({**result,'boxes':[box(bad)]},['person'])


@unittest.skipUnless(IMAGE_DEPS,'NumPy/Pillow are required')
class MaskTests(unittest.TestCase):
    def test_full_empty_single_pixel_and_asymmetric_masks(self):
        import numpy as np
        self.assertEqual(mask_centroid(np.ones((4,8),bool)),[.5,.5])
        self.assertIsNone(mask_centroid(np.zeros((4,8),bool)))
        mask=np.zeros((4,8),bool);mask[0,0]=True
        self.assertEqual(mask_centroid(mask),[.5/8,.5/4])
        mask[1,0]=mask[0,1]=True
        expected=[(1/3+.5)/8,(1/3+.5)/4]
        np.testing.assert_allclose(mask_centroid(mask),expected)
        self.assertNotEqual(mask_centroid(mask),[1/8,1/4]) # Not bounding-box center.

    def test_disconnected_holes_and_strides_match_coordinate_reference(self):
        import numpy as np
        mask=np.random.default_rng(5).integers(0,2,(15,31)).astype(bool)[:,::2]
        saved=mask.copy();ys,xs=np.nonzero(mask)
        np.testing.assert_allclose(mask_centroid(mask),[(xs.mean()+.5)/mask.shape[1],(ys.mean()+.5)/mask.shape[0]])
        np.testing.assert_array_equal(mask,saved)

    def test_bad_masks_rejected(self):
        import numpy as np
        for value in (np.zeros((2,2),float),np.zeros((0,2),bool),np.ones(3,bool),[[True]]):
            with self.assertRaises(ValueError):mask_centroid(value)

    def test_both_workers_export_centroid_without_another_model_pass(self):
        import numpy as np
        from PIL import Image
        from spring_turret.sam31_tracking_worker import TrackingEngine
        from spring_turret.sam31_tracking_native_worker import NativeTrackingEngine
        mask=np.zeros((8,8),bool);mask[1,1:5]=True;mask[2,1]=True
        class Pixels:
            def __sub__(self,other):return self
            def __truediv__(self,other):return self
            def to(self,other):return self
        pixels=Pixels()
        transform=SimpleNamespace(resize=lambda *a:pixels,to_tensor=lambda *a:pixels)
        output={'active_ids':[1],'propagated_ids':[1],'memory_frames':4,'retired_ids':[],'dropped_objects':0,
            'out_obj_ids':[1],'out_probs':[.9],'out_boxes_xywh':[[.125,.125,.5,.25]],'out_binary_masks':[mask]}
        jpeg=io.BytesIO();Image.new('RGB',(8,8)).save(jpeg,format='JPEG')
        for cls in (TrackingEngine,NativeTrackingEngine):
            with self.subTest(worker=cls.__name__):
                e=cls.__new__(cls)
                e.torch=SimpleNamespace(inference_mode=nullcontext,autocast=lambda *a,**kw:nullcontext(),
                    bfloat16='bf16',__version__='test',xpu=SimpleNamespace(synchronize=lambda:None,get_device_name=lambda *a:'test'))
                e.accelerator=SimpleNamespace(synchronize=lambda *a:None)
                e.device=SimpleNamespace(type='xpu');e.device_name='test'
                class Preprocess:
                    validated=True
                    def __call__(self,jpeg,*,prepared=None):return pixels,(8,8)
                e.preprocess=Preprocess();e.image_size=1008
                e.confidence=.5;e.graph_stages={};e.sources={};e.source_count=-1;e.source_digest=None
                e.ids={};e.next_id=1;e.session_key=(('person',),8,8,None,None)
                e.last_captured_at=None;e.previous_duration=0.
                stage=SimpleNamespace(calls=0,direct_calls=0,replay_calls=0)
                if cls is NativeTrackingEngine:
                    e.graph_stages={'image_and_detection':stage}
                    e.cache_warmup_frames=64
                def step(*args,**kwargs):
                    stage.calls += 1
                    return output
                session=SimpleNamespace(index=2,step=Mock(side_effect=step));e.sessions=[session]
                e.model=SimpleNamespace(_native_work_observations={},_native_preflight_stats={});e.receipt={}
                with patch.dict(sys.modules,{'torchvision.transforms':SimpleNamespace(functional=transform)}), \
                        patch('spring_turret.sam31_tracking_worker.tracking_graph_status',return_value={}):
                    result=e.detect({'jpeg':jpeg.getvalue(),'prompts':['person']})
                session.step.assert_called_once()
                np.testing.assert_allclose(result['boxes'][0]['mask_centroid'],mask_centroid(mask))
                self.assertEqual(result['boxes'][0]['xyxy'],[.125,.125,.625,.375])
                self.assertEqual(result['active_instance_ids'],[1])


if __name__=='__main__':unittest.main()
