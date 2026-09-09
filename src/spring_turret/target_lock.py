"""Conservative target continuity in calibrated world-bearing coordinates.

This is spatial association, not an appearance/re-identification model. A lost
target is never replaced just because another detection is nearer the camera
center. Reacquisition needs three consecutive, unambiguous nearby observations.
No smoothing, prediction, confidence changes or motor speed caps are applied.
"""
from dataclasses import dataclass
import math


def angle(a, b):
    return math.degrees(math.acos(max(-1., min(1., sum(x*y for x, y in zip(a, b))))))


@dataclass(frozen=True)
class Observation:
    box: dict
    direction: tuple
    size: float  # Angular diagonal; not the changing screen-space box area.

    @property
    def identity(self):
        return self.box.get("instance_id")


class TargetLock:
    # Association tolerances, not servo motion limits. Missing targets do not
    # widen these gates over time: prolonged loss must not capture a neighbor.
    REACQUIRE_DEGREES = 3.
    CONTINUITY_DEGREES = 6.
    AMBIGUITY_DEGREES = 1.
    CONFIRM_FRAMES = 3

    def __init__(self):
        self.reset()

    def reset(self):
        self.accepted = self.pending = None
        self.pending_count = 0
        self.missing = False
        self.reason = None

    def hold(self, reason="missing"):
        self.missing = True
        self.pending = None
        self.pending_count = 0
        self.reason = reason

    def accept(self, observation):
        self.accepted = observation
        self.pending = None
        self.pending_count = 0
        self.missing = False
        self.reason = None
        return observation

    def choose(self, observations, identity=None, *, temporal_active=False):
        if self.accepted is None:
            # Input order is nearest-to-frame-center. A click must first see
            # that exact ID in a fresh, pose-paired result; never invent a match.
            options = [o for o in observations if identity is None or o.identity == identity]
            if options:
                return self.accept(options[0])
            self.hold()
            return None

        anchor = self.accepted
        same = [o for o in observations if o.identity == anchor.identity]
        if not self.missing and len(same) == 1:
            candidate = same[0]
            if (angle(anchor.direction, candidate.direction) <= self.CONTINUITY_DEGREES
                    and .4 <= candidate.size / anchor.size <= 2.5):
                return self.accept(candidate)
            # ID reuse or a wildly changed mask is not evidence of continuity.
            self.hold("discontinuous")
            return None

        self.missing = True
        options = []
        for o in observations:
            if temporal_active and o.identity != anchor.identity:
                continue  # Respect an occluded but still-live SAM track.
            distance = angle(anchor.direction, o.direction)
            if distance <= self.REACQUIRE_DEGREES and .5 <= o.size / anchor.size <= 2.:
                options.append((distance, o))
        options.sort(key=lambda item: item[0])
        if not options:
            self.hold()
            return None
        if len(options) > 1 and options[1][0] - options[0][0] < self.AMBIGUITY_DEGREES:
            self.hold("ambiguous")
            return None
        candidate = options[0][1]
        if (self.pending is not None and self.pending.identity == candidate.identity
                and angle(self.pending.direction, candidate.direction) <= self.REACQUIRE_DEGREES):
            self.pending_count += 1
        else:
            self.pending_count = 1
        self.pending = candidate
        self.reason = "confirming"
        if self.pending_count >= self.CONFIRM_FRAMES:
            return self.accept(candidate)
        return None
