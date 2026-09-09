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


def bounded_pointing(world, center, current, limits):
    """Global closest direction for a rectangular pan/tilt angular domain.

    Maximize the dot product of two unit rays. Interior regular maxima align
    the rays; singular extrema have zero post-tilt z. On each boundary the
    remaining objective is A*cos(angle)+B*sin(angle), solved analytically.
    Include corners and periodic equivalents, then break equal-error ties by
    joint travel. This avoids selecting an infeasible IK branch then clamping it.
    """
    def equivalents(angle, bounds):
        low, high = bounds
        return [min(high, max(low, angle + 360*k))
                for k in range(math.ceil((low-angle-1e-9)/360),
                               math.floor((high-angle+1e-9)/360)+1)]
    def clamp(value, bounds):
        return min(bounds[1], max(bounds[0], value))
    if (any(not math.isfinite(v) for v in (*world,*center,*current,*limits[0],*limits[1]))
            or any(not 0 <= hi-lo <= 360 for lo,hi in limits)):
        raise ValueError("Invalid pointing domain")
    wx,wy,wz = world
    x,y,z = center
    candidates = [(clamp(current[0], limits[0]), clamp(current[1], limits[1]))]
    def add_pan(tilt):
        t = math.radians(tilt)
        tz = y*math.sin(t)+z*math.cos(t)
        a,b = wx*x+wz*tz, wx*tz-wz*x
        pans = (equivalents(math.degrees(math.atan2(b,a)), limits[0])
                if math.hypot(a,b)>1e-14 else [clamp(current[0], limits[0])])
        candidates.extend((p,tilt) for p in (*limits[0],*pans))
    # Both tilt edges, plus rank-deficient interior configurations.
    beta = math.atan2(y,z)
    tilts = [*limits[1],clamp(current[1],limits[1])]
    for t in (beta-math.pi/2,beta+math.pi/2):
        tilts.extend(equivalents(math.degrees(t), limits[1]))
    radius = math.hypot(y,z)
    if radius > 1e-12 and abs(wy) <= radius+1e-12:
        angle = math.asin(max(-1,min(1,wy/radius)))
        for t in (beta-angle,beta-(math.pi-angle)):
            tilts.extend(equivalents(math.degrees(t), limits[1]))
    for tilt in tilts:
        add_pan(tilt)
    # Pan edges: rotate the target into the fixed-pan body frame.
    for pan in limits[0]:
        p = math.radians(pan)
        target_z = wx*math.sin(p)+wz*math.cos(p)
        a,b = wy*y+target_z*z, target_z*y-wy*z
        tilts = (equivalents(math.degrees(math.atan2(b,a)), limits[1])
                 if math.hypot(a,b)>1e-14 else [clamp(current[1], limits[1])])
        candidates.extend((pan,t) for t in tilts)
    scored = [(dot(world,body_ray(center,*q)),q) for q in candidates]
    score = max(value for value,q in scored)
    result = min((q for value,q in scored if value >= score-1e-12),
                 key=lambda q:sum((v-c)**2 for v,c in zip(q,current)))
    return result, math.degrees(math.acos(max(-1,min(1,score))))


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
        return self.solve(u,v,pose,servo)["goals"]

    def solve(self, u, v, pose, servo):
        """Aim the requested pixel ray at the FRAME center, not optical center.

        Solve within the actual joint limits, including both IK branches. For
        unreachable rays return the best reachable direction, not a clamped
        unconstrained solution. The servo still independently enforces limits.
        """
        return self.solve_direction(self.world_direction(u,v,pose,servo),pose,servo)

    def world_direction(self, u, v, pose, servo):
        q,_=self.reference_pose(pose,servo)
        sx,sy=self.data["tracking_directions"]["x"],-self.data["tracking_directions"]["y"]
        return body_ray(transform(self.mount,self.ray(u,v)),q["x"]*sx,q["y"]*sy)

    def solve_direction(self, world, pose, servo):
        q, offsets=self.reference_pose(pose,servo)
        sx,sy=self.data["tracking_directions"]["x"],-self.data["tracking_directions"]["y"]
        pan,tilt=q["x"]*sx,q["y"]*sy
        limits=[]
        for name,sign,reference in (("x",sx,pan),("y",sy,tilt)):
            axis=servo["axes"][name]
            limits.append(sorted(sign*(axis[key]+offsets[name]) for key in ("min_degrees","max_degrees"))
                          if "min_degrees" in axis and "max_degrees" in axis
                          else [reference-180,reference+180])
        (pp,tt),error=bounded_pointing(world,self.center,(pan,tilt),limits)
        return {"goals":{"x":pp/sx-offsets["x"],"y":tt/sy-offsets["y"]},
                "angular_error_degrees":error,"limited":error>1e-4}

    def reproject(self, u, v, before, after):
        """Predict scene-feature motion using the same calibrated camera model."""
        return self.image_point(self.world_direction(u,v,before,before),after,after)

    def image_point(self, world, pose, servo):
        """Project a world-space ray into this pose's calibrated sensor."""
        qb,_=self.reference_pose(pose,servo)
        sx,sy=self.data["tracking_directions"]["x"],-self.data["tracking_directions"]["y"]
        x,y,z=world
        p,t=math.radians(qb["x"]*sx),math.radians(qb["y"]*sy)
        x,z=math.cos(p)*x-math.sin(p)*z,math.sin(p)*x+math.cos(p)*z
        y,z=math.cos(t)*y+math.sin(t)*z,-math.sin(t)*y+math.cos(t)*z
        return self.pixel(transform(list(zip(*self.mount)),[x,y,z]))
