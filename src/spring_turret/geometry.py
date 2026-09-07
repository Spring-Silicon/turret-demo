"""Qualified fisheye rays and coupled pan/tilt geometry; standard library only."""
import json
import math
import os
import stat
from pathlib import Path


def usb_identity(device):
    """Bind calibration to a USB camera, not its changeable videoN number."""
    try:
        info = Path(device).stat()
        if not stat.S_ISCHR(info.st_mode):
            return None
        # Character-device identity survives container aliases such as
        # /dev/spring-turret-camera; the basename need not be videoN.
        path = (Path("/sys/dev/char") /
                f"{os.major(info.st_rdev)}:{os.minor(info.st_rdev)}" / "device").resolve()
        for parent in (path, *path.parents):
            if (parent / "idVendor").exists():
                return ":".join((parent / name).read_text().strip() for name in ("idVendor", "idProduct", "serial"))
    except OSError:
        pass
    return None


def dot(a, b):
    return sum(x*y for x,y in zip(a,b))


def transform(matrix, vector):
    return [dot(row, vector) for row in matrix]


def body_ray(vector, pan, tilt):
    """Ry(pan) Rx(tilt), camera coordinates right/down/forward."""
    p,t = math.radians(pan), math.radians(tilt)
    x,y,z = vector
    y,z = math.cos(t)*y-math.sin(t)*z, math.sin(t)*y+math.cos(t)*z
    return [math.cos(p)*x+math.sin(p)*z, y, -math.sin(p)*x+math.cos(p)*z]


class Geometry:
    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def __init__(self, data):
        self.data = data
        try:
            assert data["version"] == 1 and data["model"] == "equidistant-k2"
            camera = data["camera"]
            assert isinstance(camera["identity"], str) and len(camera["identity"]) > 5
            assert all(type(camera[k]) is int and camera[k] > 0 for k in ("width", "height"))
            self.width,self.height = camera["width"],camera["height"]
            self.lens = data["intrinsics"]
            assert set(self.lens) == {"fx","fy","cx","cy","k1","k2"}
            assert all(type(v) in (int,float) and math.isfinite(v) for v in self.lens.values())
            assert all(.1*self.width < self.lens[k] < 3*self.width for k in ("fx","fy"))
            assert 0 < self.lens["cx"] < self.width and 0 < self.lens["cy"] < self.height
            assert all(1+3*self.lens["k1"]*t*t+5*self.lens["k2"]*t**4 > .2 for t in [i*1.6/100 for i in range(101)])
            self.mount = data["mount_rotation"]
            assert len(self.mount) == 3 and all(len(row)==3 for row in self.mount)
            assert all(type(v) in (int,float) and math.isfinite(v) for row in self.mount for v in row)
            assert all(abs(dot(a,b)-(i==j)) < 1e-6 for i,a in enumerate(self.mount) for j,b in enumerate(self.mount))
            a,b,c = self.mount
            assert dot(a,[b[1]*c[2]-b[2]*c[1],b[2]*c[0]-b[0]*c[2],b[0]*c[1]-b[1]*c[0]]) > .999999
            assert set(data["axes"]) == {"x","y"} and set(data["tracking_directions"]) == {"x","y"}
            for axis in data["axes"].values():
                assert set(axis)=={"id","direction","origin"} and all(type(v) is int for v in axis.values())
                assert axis["direction"] in (-1,1) and 0<=axis["id"]<=252 and 0<=axis["origin"]<4096
            assert data["axes"]["x"]["id"] != data["axes"]["y"]["id"]
            assert all(type(v) is int and v in (-1,1) for v in data["tracking_directions"].values())
            q = data["qualification"]
            assert q["passed"] is True and q["heldout_poses"] >= 4 and q["heldout_points"] >= 500
            assert 0 <= q["heldout_median_px"] < min(1.5, .6*q["baseline_median_px"])
            assert 0 <= q["heldout_p90_px"] < min(4, .8*q["baseline_p90_px"])
            # Prove every point in the rectangular sensor has a unique ray.
            for u,v in ((0,0),(self.width,0),(0,self.height),(self.width,self.height)):
                self.ray(u,v)
        except (AssertionError,KeyError,TypeError,ValueError,OverflowError) as error:
            raise ValueError("Invalid or unqualified fisheye calibration") from error
        self.center = transform(self.mount, self.ray(self.width/2, self.height/2))

    def ray(self, u, v):
        p=self.lens
        x,y=(u-p["cx"])/p["fx"],(v-p["cy"])/p["fy"]
        radius=math.hypot(x,y)
        def distort(t): return t*(1+p["k1"]*t*t+p["k2"]*t**4)
        if not math.isfinite(radius) or radius > distort(1.6):
            raise ValueError("Point outside calibrated fisheye domain")
        if radius < 1e-12:
            return [0.,0.,1.]
        low,high=0.,1.6
        for _ in range(35):
            mid=(low+high)/2
            if distort(mid) < radius: low=mid
            else: high=mid
        theta=(low+high)/2
        scale=math.sin(theta)/radius
        return [x*scale,y*scale,math.cos(theta)]

    def pixel(self, ray):
        x,y,z=ray
        radius=math.hypot(x,y)
        theta=math.atan2(radius,z)
        if theta > 1.6:
            raise ValueError("Ray outside calibrated fisheye domain")
        p=self.lens
        rd=theta*(1+p["k1"]*theta**2+p["k2"]*theta**4)
        scale=rd/radius if radius>1e-12 else 1.
        return [p["cx"]+p["fx"]*x*scale,p["cy"]+p["fy"]*y*scale]

    def check_binding(self, camera, servo, config):
        if any(camera.get(k) != self.data["camera"][k] for k in ("identity","width","height")):
            raise ValueError("Camera identity/resolution differs from calibration")
        for name, axis in self.data["axes"].items():
            if (any(servo["axes"][name].get(k)!=axis[k] for k in ("id","direction"))
                    or config.get(name+"_direction", 1 if name=="x" else -1) != self.data["tracking_directions"][name]):
                raise ValueError("Servo identity/direction differs from calibration")

    def reference_pose(self, pose, servo):
        offsets={name: (((servo["axes"][name]["origin"]-axis["origin"]+2048)%4096)-2048)*360/4096*axis["direction"]
            for name,axis in self.data["axes"].items()}
        return {name: pose["axes"][name]["degrees"]+offsets[name] for name in offsets}, offsets

    def goals(self, u, v, pose, servo):
        """Aim the requested pixel ray at the FRAME center, not optical center.

        Enumerate the two tilt solutions and select the nearest pan/tilt pose.
        Servo limits are applied by the existing controller, after solving.
        """
        q, offsets=self.reference_pose(pose,servo)
        sx,sy=self.data["tracking_directions"]["x"],-self.data["tracking_directions"]["y"]
        pan,tilt=q["x"]*sx,q["y"]*sy
        world=body_ray(transform(self.mount,self.ray(u,v)),pan,tilt)
        x,y,z=self.center
        radius=math.hypot(y,z)
        if radius<1e-6 or abs(world[1])>radius+1e-8:
            raise ValueError("Target ray is unreachable with this camera mounting")
        beta=math.atan2(y,z)
        angle=math.asin(max(-1,min(1,world[1]/radius)))
        candidates=[]
        def near(value, reference): return reference+(value-reference+180)%360-180
        for t in (beta-angle,beta-(math.pi-angle)):
            tz=y*math.sin(t)+z*math.cos(t)
            p=math.atan2(world[0],world[2])-math.atan2(x,tz)
            pp,tt=near(math.degrees(p),pan),near(math.degrees(t),tilt)
            candidates.append(((pp-pan)**2+(tt-tilt)**2,pp,tt))
        _,pp,tt=min(candidates)
        return {"x":pp/sx-offsets["x"],"y":tt/sy-offsets["y"]}

    def reproject(self, u, v, before, after):
        """Predict scene-feature motion using the same calibrated camera model."""
        qa,_=self.reference_pose(before,before)
        qb,_=self.reference_pose(after,after)
        sx,sy=self.data["tracking_directions"]["x"],-self.data["tracking_directions"]["y"]
        x,y,z=body_ray(transform(self.mount,self.ray(u,v)),qa["x"]*sx,qa["y"]*sy)
        p,t=math.radians(qb["x"]*sx),math.radians(qb["y"]*sy)
        x,z=math.cos(p)*x-math.sin(p)*z,math.sin(p)*x+math.cos(p)*z
        y,z=math.cos(t)*y+math.sin(t)*z,-math.sin(t)*y+math.cos(t)*z
        return self.pixel(transform(list(zip(*self.mount)),[x,y,z]))
