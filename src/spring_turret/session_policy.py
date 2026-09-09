"""Shared lifetime policy around dense CUDA and pinned native XPU sessions.

The model adapter owns neural execution and attention memory. This wrapper owns
when a missing identity is retired, and keeps all model identity tables aligned.
"""
import time
try:
    from .policy import TRACK_RETENTION_SECONDS, TRACK_RETENTION_FRAMES
    from .track_lifetime import TrackLifetime, retire_tracks
except ImportError:
    from policy import TRACK_RETENTION_SECONDS, TRACK_RETENTION_FRAMES
    from track_lifetime import TrackLifetime, retire_tracks


class ManagedSession:
    def __init__(self, session, *, confidence):
        self.session, self.confidence = session, confidence
        self.lifetime = TrackLifetime(seconds=TRACK_RETENTION_SECONDS, frames=TRACK_RETENTION_FRAMES)
        self.previous_active = set()

    @property
    def index(self):
        return self.session.index

    def step(self, pixels, *, timestamp=None):
        result = self.session.step(pixels)
        active = set(map(int, result["active_ids"]))
        visible = {int(i) for i, score, box, mask in zip(
            result["out_obj_ids"], result["out_probs"], result["out_boxes_xywh"],
            result["out_binary_masks"], strict=True)
            if float(score) >= self.confidence and box[2] > 0 and box[3] > 0 and mask.any()}
        expired = self.lifetime.update(active, visible, self.index,
                                       time.monotonic() if timestamp is None else timestamp)
        if expired:
            retire_tracks(self.session.model, self.session.trackers, self.session.metadata, expired)
            keep = [i for i, identity in enumerate(result["out_obj_ids"]) if int(identity) not in expired]
            for key in ("out_obj_ids", "out_probs", "out_boxes_xywh", "out_binary_masks"):
                values = result[key]
                result[key] = ([values[i] for i in keep] if isinstance(values, (list, tuple)) else values[keep])
        active -= expired
        result["active_ids"] = sorted(active)
        result["propagated_ids"] = sorted(set(result["propagated_ids"]) & active)
        result["retired_ids"] = sorted(self.previous_active - active)
        result["dropped_objects"] = result.get("dropped_objects", 0)
        self.previous_active = active
        return result
