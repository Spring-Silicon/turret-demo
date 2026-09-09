"""Operator-authorized bounded pointing check on three natural scene features."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import urllib.request
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from spring_turret.geometry import Geometry

p=argparse.ArgumentParser()
p.add_argument("--geometry",required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--allow-motion",action="store_true",required=True)
a=p.parse_args(); g=Geometry.load(a.geometry)
assert not a.output.exists()
cv2.setNumThreads(2)
sift=cv2.SIFT_create(nfeatures=6000,contrastThreshold=.01)
bf=cv2.BFMatcher()
def api(path="/api/status",body=None):
    req=urllib.request.Request("http://localhost:8080"+path,data=None if body is None else json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=4) as r: return json.load(r)
initial=api(); assert initial["servo"]["ready"] and not initial["servo"]["armed"]
home={k:v["degrees"] for k,v in initial["servo"]["axes"].items()}
origins={k:v["origin"] for k,v in initial["servo"]["axes"].items()}
expected=dict(home)
results=[]; armed=False
def checked():
    s=api()
    assert s["servo"]["armed"] and s["servo"]["online"] and s["tracking"]["target"] is None,"Operator stopped/changed controls"
    for k,v in s["servo"]["axes"].items():
        assert v["origin"]==origins[k] and abs(v["goal_degrees"]-expected[k])<.15,"Operator changed goal/zeros"
        assert abs(v["degrees"]-home[k])<11
    api("/api/servo/keepalive",{})
    return s["servo"]
def move(goal):
    assert all(abs(goal[k]-home[k])<= (8 if k=="x" else 6) for k in goal),"Goal outside authorized envelope"
    for k in goal:
        while abs(goal[k]-expected[k])>.001:
            checked()
            value=expected[k]+max(-3,min(3,goal[k]-expected[k]))
            api("/api/servo/position",{"axis":k,"degrees":value}); expected[k]=value
            time.sleep(.14)
    history=[]; deadline=time.monotonic()+4
    while time.monotonic()<deadline:
        s=checked(); history.append(s)
        if len(history)>=5 and all(max(h["axes"][k]["degrees"] for h in history[-5:])-min(h["axes"][k]["degrees"] for h in history[-5:])<=.18 for k in goal): return s
        time.sleep(.12)
    raise RuntimeError("Not settled")
def snapshot():
    before=checked()
    with urllib.request.urlopen("http://localhost:8080/stream.mjpg",timeout=3) as r:
        b=b""
        while b"\xff\xd9" not in b:
            b+=r.read(4096); assert len(b)<4_000_000
    after=checked()
    assert all(abs(before["axes"][k]["degrees"]-after["axes"][k]["degrees"])<=.18 for k in home)
    image=cv2.imdecode(np.frombuffer(b[b.index(b"\xff\xd8"):b.index(b"\xff\xd9")+2],dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    keys,desc=sift.detectAndCompute(image,None)
    return after,keys,desc
def locate(reference,keys,desc):
    points,descriptors,center=reference
    pairs=[(points[m.queryIdx],keys[m.trainIdx].pt) for m,n in bf.knnMatch(descriptors,desc,k=2)
           if m.distance<.65*n.distance]
    assert len(pairs)>=12,"Not enough unambiguous local feature matches"
    pairs=np.array(pairs,dtype=np.float32)
    affine,mask=cv2.estimateAffine2D(np.ascontiguousarray(pairs[:,0]),np.ascontiguousarray(pairs[:,1]),method=cv2.RANSAC,ransacReprojThreshold=1.5,
                                   maxIters=2000,confidence=.995,refineIters=10)
    assert affine is not None and int(mask.sum())>=10,"Local feature group is inconsistent"
    inliers=pairs[mask.ravel().astype(bool)]
    hull=cv2.convexHull(np.ascontiguousarray(inliers[:,0]))
    assert cv2.contourArea(hull)>500 and cv2.pointPolygonTest(hull,center,False)>=0,"Target not bracketed by supporting features"
    residual=np.linalg.norm(inliers[:,0]@affine[:,:2].T+affine[:,2]-inliers[:,1],axis=1)
    assert np.median(residual)<.75,"Local warp residual too large"
    return tuple(float(v) for v in affine@np.array([*center,1.]))
try:
    api("/api/tracking/target",{"target":None})
    armed=True
    s=api("/api/servo/arm",{})["servo"]
    expected={k:v["goal_degrees"] for k,v in s["axes"].items()}
    move(home)
    _,keys,desc=snapshot()
    _,repeat_keys,repeat_desc=snapshot()
    targets=[]
    for desired in ((600,330),(680,330),(660,395)):
        indices=[i for i,k in enumerate(keys) if math.dist(k.pt,desired)<110]
        reference=(np.array([keys[i].pt for i in indices]),desc[indices].copy(),desired)
        assert math.dist(locate(reference,repeat_keys,repeat_desc),desired)<2,"Reference features are not repeatable"
        targets.append(reference)
    for index,reference in enumerate(targets):
        for method in ("linear","fisheye"):
            move(home)
            before,keys,desc=snapshot(); u,v=locate(reference,keys,desc)
            geometric=g.goals(u,v,before,before)
            goal=geometric if method=="fisheye" else {"x":before["axes"]["x"]["degrees"]+(u-640)*161.6/1280,
                "y":before["axes"]["y"]["degrees"]-(v-360)*82.8/720}
            move(goal)
            after,keys,desc=snapshot(); observed=locate(reference,keys,desc)
            record={"feature":index,"method":method,"before_pixel":[u,v],"goal":goal,
                "actual":{k:after["axes"][k]["degrees"] for k in home},"after_pixel":observed,
                "first_move_error_px":math.dist(observed,[640,360]),
                "geometry_prediction_error_px":math.dist(g.reproject(u,v,before,after),observed)}
            if method=="fisheye":
                for _ in range(2):
                    if math.dist(observed,[640,360])<2: break
                    correction=g.goals(*observed,after,after)
                    # Settled goal minus readback estimates holding bias; this
                    # is not inferred from an in-flight measurement.
                    refined={k:correction[k]+expected[k]-after["axes"][k]["degrees"] for k in home}
                    move(refined)
                    after,keys,desc=snapshot(); observed=locate(reference,keys,desc)
                record["refined_error_px"]=math.dist(observed,[640,360])
            results.append(record); print(json.dumps(record),flush=True)
    move(home)
finally:
    if armed: api("/api/servo/disable",{})
    a.output.write_text(json.dumps({"initial":initial["servo"],"results":results,"final":api()["servo"]},indent=2)+"\n")
