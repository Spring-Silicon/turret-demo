"""Short-lived spatial instance IDs; no appearance re-identification or GPU work."""

import math
from spring_turret.geometry import GeometryResource


class InstanceAssociator:
    def __init__(self, config):
        self.config = config
        self.geometry_resource = GeometryResource(config.get("geometry_file"))
        self.geometry = self.geometry_resource.value
        self.tracks = {}
        self.next_id = 1

    def clear(self):
        self.tracks.clear()  # Never reuse an old clickable ID after a prompt reset.

    def _predicted(self, track, pose):
        coords = list(track["box"]["xyxy"])
        if pose and track["pose"] and not self.geometry_resource.error:
            if self.geometry:
                try:
                    w,h=self.geometry.width,self.geometry.height
                    points=[self.geometry.reproject(x*w,y*h,track["pose"],pose)
                            for x,y in ((coords[0],coords[1]),(coords[2],coords[1]),(coords[0],coords[3]),(coords[2],coords[3]))]
                    return [min(p[0] for p in points)/w,min(p[1] for p in points)/h,
                            max(p[0] for p in points)/w,max(p[1] for p in points)/h]
                except (KeyError,ValueError):
                    return coords  # Missing pose or out-of-view; never invent a new identity.
            for axis, indices, default_scale, default_direction in (
                ("x", (0, 2), 160, 1), ("y", (1, 3), 90, -1)
            ):
                delta = track["pose"]["axes"][axis]["degrees"] - pose["axes"][axis]["degrees"]
                shift = delta / (self.config.get(f"{axis}_degrees_per_frame", default_scale) *
                                 self.config.get(f"{axis}_direction", default_direction))
                for index in indices:
                    coords[index] += shift
        return coords

    @staticmethod
    def _cost(a, b):
        aw, ah, bw, bh = a[2]-a[0], a[3]-a[1], b[2]-b[0], b[3]-b[1]
        if min(aw, ah, bw, bh) <= 0 or not .3 <= (bw*bh)/(aw*ah) <= 3.3:
            return None
        dx, dy = abs((a[0]+a[2]-b[0]-b[2])/2), abs((a[1]+a[3]-b[1]-b[3])/2)
        gx, gy = max(.035, aw, bw), max(.035, ah, bh)
        if dx > gx or dy > gy:
            return None
        intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
        iou = intersection / (aw*ah + bw*bh - intersection)
        return math.hypot(dx/gx, dy/gy) + (1-iou)*.25

    def update(self, boxes, pose, captured_at):
        if self.geometry_resource.error:
            self.geometry = self.geometry_resource.refresh()
        self.tracks = {i: t for i, t in self.tracks.items() if 0 <= captured_at-t["at"] <= .75}
        boxes = [dict(b) for b in boxes if isinstance(b, dict) and
                 isinstance(b.get("xyxy"), (list, tuple)) and len(b["xyxy"]) == 4 and
                 all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in b["xyxy"]) and
                 b["xyxy"][0] < b["xyxy"][2] and b["xyxy"][1] < b["xyxy"][3]]
        by_track, by_box = {}, {}
        for identity, track in self.tracks.items():
            predicted = self._predicted(track, pose)
            for index, box in enumerate(boxes):
                if box.get("prompt") != track["box"].get("prompt"):
                    continue
                cost = self._cost(predicted, box["xyxy"])
                if cost is not None:
                    by_track.setdefault(identity, []).append((cost, index))
                    by_box.setdefault(index, []).append((cost, identity))
        for candidates in (*by_track.values(), *by_box.values()):
            candidates.sort()

        def unambiguous(candidates):
            return len(candidates) == 1 or candidates[1][0]-candidates[0][0] > .3

        matches = {}
        for identity, candidates in by_track.items():
            _, index = candidates[0]
            reverse = by_box[index]
            # Mutual, unambiguous nearest association: do not hand a clicked
            # identity to an arbitrary neighboring box during a crossing.
            if reverse[0][1] == identity and unambiguous(candidates) and unambiguous(reverse):
                matches[index] = identity
        # Retire identities that competed for a visible detection but lost or
        # became ambiguous. Keeping those alongside replacement IDs makes every
        # later frame ambiguous too, even once the objects stop moving/apart.
        # Truly absent tracks (no viable candidate) can still bridge a dropout.
        matched = set(matches.values())
        self.tracks = {i: t for i, t in self.tracks.items() if i in matched or i not in by_track}
        for index, box in enumerate(boxes):
            identity = matches.get(index)
            if identity is None:
                identity = self.next_id
                self.next_id += 1
            box["instance_id"] = identity
            self.tracks[identity] = {"box": box, "pose": pose, "at": captured_at}
        return boxes
