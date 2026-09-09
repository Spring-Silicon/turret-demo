"""Synthetic known-ray checks, independent of the scene fitting optimizer."""
import copy
import math
import os
import random
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from spring_turret.geometry import Geometry, body_ray, transform, usb_identity, bounded_pointing, dot
from spring_turret.instances import InstanceAssociator

def fixture():
    roll=math.radians(3)
    return {"version":1,"model":"equidistant-k2","camera":{"identity":"test:usb:123","width":1280,"height":720},
        "intrinsics":{"fx":530.,"fy":520.,"cx":645.,"cy":370.,"k1":-.01,"k2":-.005},
        "mount_rotation":[[math.cos(roll),-math.sin(roll),0],[math.sin(roll),math.cos(roll),0],[0,0,1]],
        "axes":{"x":{"id":2,"direction":1,"origin":2010},"y":{"id":1,"direction":1,"origin":962}},
        "tracking_directions":{"x":1,"y":-1},"qualification":{"passed":True,"heldout_poses":6,"heldout_points":1000,
        "heldout_median_px":1.,"heldout_p90_px":2.,"baseline_median_px":5.,"baseline_p90_px":10.}}

def pose(x=0,y=0):
    return {"axes":{k:{**v,"degrees":x if k=="x" else y,"goal_degrees":x if k=="x" else y}
                    for k,v in fixture()["axes"].items()}}

class GeometryTests(unittest.TestCase):
    def setUp(self): self.g=Geometry(fixture())

    def test_usb_identity_uses_device_number_not_container_alias_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            usb = root / "usb"
            device = usb / "interface"
            device.mkdir(parents=True)
            for key, value in {"idVendor":"0c45", "idProduct":"0261", "serial":"UC684"}.items():
                (usb/key).write_text(value)
            char = root/"char"/"81:2"
            char.mkdir(parents=True)
            (char/"device").symlink_to(device)
            node = SimpleNamespace(stat=lambda:SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=os.makedev(81,2)))
            def path(value):
                return node if value == "/dev/camera-alias" else root/"char" if value == "/sys/dev/char" else Path(value)
            with patch("spring_turret.geometry.Path", side_effect=path):
                self.assertEqual(usb_identity("/dev/camera-alias"), "0c45:0261:UC684")
            self.assertIsNone(usb_identity(str(usb/"serial")))
            self.assertIsNone(usb_identity(str(root/"missing")))

    def test_pixel_ray_roundtrip_full_frame(self):
        for u in range(0,1281,80):
            for v in range(0,721,60):
                back=self.g.pixel(self.g.ray(u,v))
                self.assertLess(math.hypot(back[0]-u,back[1]-v),1e-6)

    def test_exact_coupled_pointing_over_poses_and_frame_positions(self):
        rng=random.Random(4)
        checked=0
        for _ in range(300):
            before=pose(rng.uniform(-85,85),rng.uniform(-60,60))
            u,v=rng.uniform(80,1200),rng.uniform(50,670)
            try: goal=self.g.goals(u,v,before,before)
            except ValueError: continue  # Some rays cannot be centered by an offset mount.
            after=pose(goal["x"],goal["y"])
            # Independently compare world directions, not the fitting loss.
            source=body_ray(transform(self.g.mount,self.g.ray(u,v)),before["axes"]["x"]["degrees"],before["axes"]["y"]["degrees"])
            aimed=body_ray(self.g.center,goal["x"],goal["y"])
            self.assertLess(math.dist(source,aimed),1e-8)
            centered=self.g.reproject(u,v,before,after)
            self.assertLess(math.dist(centered,[640,360]),1e-5)
            checked+=1
        self.assertGreater(checked,280)

    def test_frame_center_is_not_optical_center(self):
        before=pose(33,47)
        goal=self.g.goals(640,360,before,before)
        self.assertAlmostEqual(goal["x"],33,places=6)
        self.assertAlmostEqual(goal["y"],47,places=6)

    def test_infeasible_nearest_branch_does_not_hide_reachable_alternative(self):
        before=pose(-79,-83)
        for axis in before['axes'].values():
            axis.update(min_degrees=-90,max_degrees=90)
        result=self.g.solve(1270,568,before,before)
        self.assertFalse(result['limited'])
        self.assertTrue(all(-90<=q<=90 for q in result['goals'].values()))
        self.assertLess(math.dist(self.g.reproject(1270,568,before,pose(**result['goals'])),[640,360]),1e-5)

    def test_unreachable_target_optimizes_both_axes_inside_bounds(self):
        center=[0.,0.,1.]
        world=body_ray(center,120,40)
        (pan,tilt),error=bounded_pointing(world,center,(0,0),[(-45,45),(-60,60)])
        self.assertAlmostEqual(pan,45)
        self.assertAlmostEqual(tilt,60)
        old=math.degrees(math.acos(dot(world,body_ray(center,45,40))))
        self.assertLess(error,old-3)

    def test_bounded_solution_dominates_dense_grid_for_random_mounts_and_bounds(self):
        rng=random.Random(51)
        for _ in range(60):
            center=body_ray([0,0,1],rng.uniform(-15,15),rng.uniform(-15,15))
            world=body_ray([0,0,1],rng.uniform(-180,180),rng.uniform(-90,90))
            bounds=[sorted([rng.uniform(-160,160),rng.uniform(-160,160)]) for _ in range(2)]
            current=[rng.uniform(-180,180),rng.uniform(-180,180)]
            q,error=bounded_pointing(world,center,current,bounds)
            self.assertTrue(all(lo-1e-9<=v<=hi+1e-9 for v,(lo,hi) in zip(q,bounds)))
            score=dot(world,body_ray(center,*q))
            for i in range(13):
                for j in range(13):
                    sample=[lo+(hi-lo)*n/12 for n,(lo,hi) in zip((i,j),bounds)]
                    self.assertGreaterEqual(score+1e-10,dot(world,body_ray(center,*sample)))

    def test_pole_ties_preserve_pan_and_periodic_equivalents_are_considered(self):
        q,error=bounded_pointing([0,-1,0],[0,0,1],(31,89),[(-90,90),(-90,90)])
        self.assertAlmostEqual(q[0],31)
        self.assertAlmostEqual(q[1],90)
        self.assertLess(error,1e-5)
        q,error=bounded_pointing(body_ray([0,0,1],-170,10),[0,0,1],(190,10),[(170,210),(-20,20)])
        self.assertAlmostEqual(q[0],190)
        self.assertAlmostEqual(q[1],10)
        self.assertLess(error,1e-5)
        # Degenerate side-facing optical ray: tilt cannot change the aim, so
        # do not move tilt needlessly while finding the best pan.
        q,error=bounded_pointing(body_ray([1,0,0],20,0),[1,0,0],(0,7),[(-90,90),(-10,10)])
        self.assertAlmostEqual(q[0],20)
        self.assertAlmostEqual(q[1],7)

    def test_reversed_axes_and_recalibrated_zero_keep_hardware_bounds(self):
        d=fixture(); d['tracking_directions']={'x':-1,'y':1}
        g=Geometry(d)
        before=pose(10,-20)
        before['axes']['x']['origin']+=100
        for axis in before['axes'].values():
            axis.update(min_degrees=-30,max_degrees=30)
        result=g.solve(1200,650,before,before)
        self.assertTrue(result['limited'])
        self.assertTrue(all(-30-1e-10<=q<=30+1e-10 for q in result['goals'].values()))

    def test_zero_recalibration_preserves_physical_geometry_including_rollover(self):
        old=pose(10,-20)
        expected=self.g.goals(780,270,old,old)
        new=copy.deepcopy(old)
        for axis, ticks in (("x",300),("y",-1200)):
            new["axes"][axis]["origin"]=(old["axes"][axis]["origin"]+ticks)%4096
            new["axes"][axis]["degrees"]-=ticks*360/4096
        result=self.g.goals(780,270,new,new)
        self.assertAlmostEqual(result["x"],expected["x"]-300*360/4096)
        self.assertAlmostEqual(result["y"],expected["y"]+1200*360/4096)

    def test_reversed_joint_directions(self):
        d=fixture(); d["tracking_directions"]={"x":-1,"y":1}
        g=Geometry(d); before=pose(20,-30)
        goal=g.goals(900,250,before,before)
        self.assertLess(math.dist(g.reproject(900,250,before,pose(**goal)),[640,360]),1e-5)

    def test_binding_rejects_camera_resolution_ids_and_directions(self):
        camera=fixture()["camera"]; servo=pose(); config={"x_direction":1,"y_direction":-1}
        self.g.check_binding(camera,servo,config)
        for update in ({"identity":"another"},{"width":640}):
            with self.assertRaises(ValueError): self.g.check_binding({**camera,**update},servo,config)
        for key,value in (("id",4),("direction",-1)):
            bad=copy.deepcopy(servo); bad["axes"]["x"][key]=value
            with self.assertRaises(ValueError): self.g.check_binding(camera,bad,config)
        with self.assertRaises(ValueError): self.g.check_binding(camera,servo,{"x_direction":-1})

    def test_invalid_calibration_rejected(self):
        for path,value in ((["qualification","passed"],False),(["qualification","heldout_p90_px"],9),
                (["intrinsics","fx"],float("nan")),(["intrinsics","k1"],-1),
                (["mount_rotation"],[[1,0,0],[0,1,0],[0,0,-1]])):
            data=fixture(); target=data
            for key in path[:-1]: target=target[key]
            target[path[-1]]=value
            with self.assertRaises(ValueError): Geometry(data)

    def test_instance_prediction_uses_fisheye_motion(self):
        association=InstanceAssociator({}); association.geometry=self.g
        old,new=pose(0,10),pose(5,15)
        coords=[.4,.4,.6,.6]
        predicted=association._predicted({"box":{"xyxy":coords},"pose":old},new)
        self.assertLess(predicted[0],coords[0])
        self.assertGreater(predicted[1],coords[1])

if __name__=="__main__": unittest.main()
