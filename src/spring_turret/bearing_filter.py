"""Adaptive low-pass on world-space unit rays, never on moving image pixels.

Uses the speed-adaptive cutoff idea of Casiez et al.'s 1 Euro filter
(https://gery.casiez.net/1euro/), with a shared vector cutoff and normalization.
This is a target estimator, not a motor speed/acceleration limiter.
"""
import math


class BearingFilter:
    def __init__(self, min_cutoff_hz, speed_gain):
        if (not math.isfinite(min_cutoff_hz) or min_cutoff_hz <= 0
                or not math.isfinite(speed_gain) or speed_gain < 0):
            raise ValueError("Invalid bearing filter parameters")
        self.min_cutoff_hz, self.speed_gain = min_cutoff_hz, speed_gain
        self.reset()

    def reset(self):
        self.timestamp = self.identity = self.direction = None
        self.velocity = [0.,0.,0.]

    def update(self, direction, timestamp, identity):
        if (len(direction) != 3 or not math.isfinite(timestamp)
                or any(not math.isfinite(v) for v in direction)):
            raise ValueError("Invalid timestamp or target ray")
        norm = math.sqrt(sum(v*v for v in direction))
        if norm < 1e-12:
            raise ValueError("Zero target ray")
        raw = [v/norm for v in direction]
        dt = timestamp-self.timestamp if self.timestamp is not None else None
        if dt is not None and dt <= 0 and identity == self.identity:
            return list(self.direction)
        # First acquisition, retargets, resumed inference and large jumps keep
        # the full measured correction. No prediction through a long gap.
        if (dt is None or identity != self.identity or dt > .5
                or sum((a-b)**2 for a,b in zip(raw,self.direction)) > .1743114855**2):
            self.timestamp, self.identity, self.direction = timestamp, identity, raw
            self.velocity = [0.,0.,0.]
            return list(raw)
        def alpha(cutoff):
            r = 2*math.pi*cutoff*dt
            return r/(1+r)
        derivative_alpha = alpha(1.)
        derivative = [(a-b)/dt for a,b in zip(raw,self.direction)]
        self.velocity = [v+derivative_alpha*(d-v) for v,d in zip(self.velocity,derivative)]
        cutoff = self.min_cutoff_hz+self.speed_gain*math.sqrt(sum(v*v for v in self.velocity))
        blend = alpha(cutoff)
        filtered = [a+blend*(b-a) for a,b in zip(self.direction,raw)]
        length = math.sqrt(sum(v*v for v in filtered))
        self.direction = [v/length for v in filtered]
        self.timestamp = timestamp
        return list(self.direction)
