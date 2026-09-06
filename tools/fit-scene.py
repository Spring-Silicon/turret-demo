"""Fit a two-term equidistant fisheye + pan/tilt mount from settled scene tracks.

Offline only; requires numpy, scipy and OpenCV. Never controls hardware.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

p = argparse.ArgumentParser()
p.add_argument("capture", type=Path)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
cv2.setNumThreads(2)
receipt = json.loads((a.capture / "capture.json").read_text())
samples = receipt["samples"]
assert len(samples) == 15 and not receipt["final"]["armed"] and "restored" in receipt
width, height = (receipt["initial"]["camera"][k] for k in ("width", "height"))
detector = cv2.SIFT_create(nfeatures=5000, contrastThreshold=.02)
features = [detector.detectAndCompute(cv2.imread(str(a.capture / s["image"]), cv2.IMREAD_GRAYSCALE), None) for s in samples]
matcher = cv2.BFMatcher()
pairs = []
for i in range(1, len(samples)):
    ka, da = features[0]
    kb, db = features[i]
    forward = {m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(da, db, k=2) if m.distance < .65*n.distance}
    reverse = {m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(db, da, k=2) if m.distance < .65*n.distance}
    indices = [(j,k) for j,k in forward.items() if reverse.get(k) == j]
    # Spatial cap prevents one highly textured region dominating the fit.
    cells = {}
    for j,k in indices:
        x,y = ka[j].pt
        cell = (min(7,int(x/width*8)), min(5,int(y/height*6)))
        bucket = cells.setdefault(cell, [])
        if len(bucket) < 35:
            bucket.append((ka[j].pt, kb[k].pt))
    matched = np.array([v for bucket in cells.values() for v in bucket])
    assert len(matched) >= 150 and len(cells) >= 24, (i,len(matched),len(cells))
    pairs.append((i, matched[:,0], matched[:,1]))
    print(json.dumps({"frame": i, "matches": len(matched), "cells_of_48": len(cells), "split": samples[i]["split"]}), flush=True)

def body(q):
    return Rotation.from_euler("Y", q["x"], degrees=True).as_matrix() @ Rotation.from_euler("X", q["y"], degrees=True).as_matrix()
rotations = [body(s["pose"]) for s in samples]
def unproject(pixels, p):
    xy = (pixels-p[2:4])/p[:2]
    rd = np.linalg.norm(xy, axis=1)
    t = np.minimum(rd, 1.6)
    for _ in range(12):
        t2=t*t
        derivative=1+3*p[4]*t2+5*p[5]*t2*t2
        t = np.clip(t-(t*(1+p[4]*t2+p[5]*t2*t2)-rd)/np.maximum(derivative,.1),0,2.5)
    scale = np.divide(np.sin(t), rd, out=np.ones_like(t), where=rd>1e-12)
    return np.column_stack([xy*scale[:,None], np.cos(t)])
def project(ray, p):
    r=np.linalg.norm(ray[:,:2],axis=1)
    t=np.arctan2(r,ray[:,2]); t2=t*t
    rd=t*(1+p[4]*t2+p[5]*t2*t2)
    scale=np.divide(rd,r,out=np.ones_like(r),where=r>1e-12)
    return ray[:,:2]*scale[:,None]*p[:2]+p[2:4]
def predict(i, points, p):
    mount=Rotation.from_rotvec(p[6:]).as_matrix()
    return project(unproject(points,p) @ (rotations[0]@mount).T @ (rotations[i]@mount),p)
train=[pair for pair in pairs if samples[pair[0]]["split"] == "train"]
def residual(p):
    r=np.concatenate([(predict(i,points,p)-observed).ravel() for i,points,observed in train])
    theta=np.linspace(0,1.6,20)
    penalty=np.minimum(0,1+3*p[4]*theta**2+5*p[5]*theta**4-.2)*1000
    return np.r_[r,penalty]
initial=np.array([520,520,width/2,height/2,0,0,0,0,0.])
low=[250,250,width*.35,height*.35,-.3,-.15,-1.3,-.6,-.6]
high=[1000,1000,width*.65,height*.65,.3,.15,1.3,.6,.6]
fit=least_squares(residual,initial,bounds=(low,high),loss="soft_l1",f_scale=1.5,x_scale="jac",max_nfev=160,ftol=1e-9)
print(json.dumps({"parameters": fit.x.tolist(), "evaluations": fit.nfev, "success": bool(fit.success)}),flush=True)
assert fit.success
reports=[]
all_test=[]; all_baseline=[]
for i,points,observed in pairs:
    error=np.linalg.norm(predict(i,points,fit.x)-observed,axis=1)
    dq=np.array([samples[0]["pose"]["x"]-samples[i]["pose"]["x"], samples[0]["pose"]["y"]-samples[i]["pose"]["y"]])
    baseline=np.linalg.norm(points+dq*np.array([width/161.6,-height/82.8])-observed,axis=1)
    report={"frame":i,"split":samples[i]["split"],"points":len(points),"median_px":float(np.median(error)),
        "p90_px":float(np.percentile(error,90)),"baseline_median_px":float(np.median(baseline)),"baseline_p90_px":float(np.percentile(baseline,90))}
    reports.append(report)
    if samples[i]["split"] == "test":
        all_test.extend(error); all_baseline.extend(baseline)
    print(json.dumps(report),flush=True)
test=np.array(all_test); baseline=np.array(all_baseline)
qualification={"heldout_median_px":float(np.median(test)),"heldout_p90_px":float(np.percentile(test,90)),
    "baseline_median_px":float(np.median(baseline)),"baseline_p90_px":float(np.percentile(baseline,90)),
    "heldout_points":len(test),"heldout_poses":6,"reports":reports}
qualification["passed"] = bool(np.median(test)<1.5 and np.percentile(test,90)<4
    and np.median(test)<.6*np.median(baseline) and np.percentile(test,90)<.8*np.percentile(baseline,90)
    and all(r["median_px"]<2 and r["p90_px"]<5 for r in reports if r["split"]=="test"))
model={"version":1,"model":"equidistant-k2","camera":{"identity":receipt["identity"],"width":width,"height":height},
    "intrinsics":{"fx":fit.x[0],"fy":fit.x[1],"cx":fit.x[2],"cy":fit.x[3],"k1":fit.x[4],"k2":fit.x[5]},
    "mount_rotation":Rotation.from_rotvec(fit.x[6:]).as_matrix().tolist(),
    "axes":{k:{"id":v["id"],"direction":1,"origin":v["origin"]} for k,v in receipt["initial"]["servo"]["axes"].items()},
    "tracking_directions":{"x":1,"y":-1},"qualification":qualification}
a.output.write_text(json.dumps(model,indent=2)+"\n")
print(json.dumps(qualification),flush=True)
if not qualification["passed"]:
    raise SystemExit("REJECTED: do not activate this model")
