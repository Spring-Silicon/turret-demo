import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from spring_turret.bearing_filter import BearingFilter
from spring_turret.geometry import body_ray
from spring_turret.tracking import validate_config

def angle(a,b):
    return math.degrees(math.acos(max(-1,min(1,sum(x*y for x,y in zip(a,b))))))

class BearingFilterTests(unittest.TestCase):
    def test_acquisition_retarget_long_gap_and_large_jump_are_immediate(self):
        f=BearingFilter(.5,40)
        for ray,at,key in [([0,0,1],1,1),(body_ray([0,0,1],1,0),1.08,2),
                           (body_ray([0,0,1],2,0),2,2),(body_ray([0,0,1],25,0),2.08,2)]:
            self.assertLess(angle(f.update(ray,at,key),ray),1e-5)
        f.reset()
        ray=body_ray([0,0,1],-40,20)
        self.assertLess(angle(f.update(ray,3,2),ray),1e-5)

    def test_duplicate_and_out_of_order_samples_cannot_move_the_filter(self):
        f=BearingFilter(.5,40)
        expected=f.update([0,0,1],5,1)
        self.assertEqual(f.update([1,0,0],5,1),expected)
        self.assertEqual(f.update([1,0,0],4,1),expected)
        self.assertEqual(f.timestamp,5)

    def test_stationary_noise_is_reduced_and_outputs_remain_unit_rays(self):
        f=BearingFilter(.5,40)
        raw_energy=filtered_energy=0.
        for i in range(200):
            raw=[.008*(-1)**i,.005*math.cos(i),1.]
            norm=math.sqrt(sum(x*x for x in raw)); raw=[x/norm for x in raw]
            out=f.update(raw,i*.08,1)
            self.assertAlmostEqual(sum(x*x for x in out),1,places=12)
            if i>10:
                raw_energy+=angle(raw,[0,0,1])**2
                filtered_energy+=angle(out,[0,0,1])**2
        self.assertLess(filtered_energy,.7*raw_energy)

    def test_filter_is_invariant_to_world_coordinate_rotation(self):
        a,b=BearingFilter(.5,40),BearingFilter(.5,40)
        for i in range(100):
            ray=body_ray([0,0,1],5*math.sin(i*.1),3*math.cos(i*.1))
            expected=body_ray(a.update(ray,i*.08,1),73,-25)
            actual=b.update(body_ray(ray,73,-25),i*.08,1)
            self.assertLess(math.dist(expected,actual),1e-12)

    def test_fast_constant_motion_has_small_added_lag(self):
        f=BearingFilter(.5,40)
        errors=[]
        for i in range(100):
            ray=body_ray([0,0,1],i*.08*90,0)
            result=f.update(ray,i*.08,1)
            if i>10:errors.append(angle(ray,result))
        self.assertLess(max(errors),.3)

    def test_invalid_inputs_and_configuration(self):
        for args in [(-1,1),(1,-1),(float('nan'),1)]:
            with self.assertRaises(ValueError):BearingFilter(*args)
        f=BearingFilter(.5,40)
        for ray,at in [([0,0,0],1),([0,1],1),([0,0,1],float('nan'))]:
            with self.assertRaises(ValueError):f.update(ray,at,1)
        good={'geometry_file':'/tmp/geometry.json','bearing_filter':{'min_cutoff_hz':.5,'speed_gain':40}}
        validate_config(good)
        for bad in [True,{}, {'min_cutoff_hz':0,'speed_gain':40}, {'min_cutoff_hz':1,'speed_gain':True}]:
            with self.assertRaises(ValueError):validate_config({**good,'bearing_filter':bad})
        with self.assertRaises(ValueError):validate_config({'bearing_filter':good['bearing_filter']})

if __name__=='__main__':unittest.main()
