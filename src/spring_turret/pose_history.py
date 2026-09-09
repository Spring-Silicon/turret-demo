"""Retrospective encoder interpolation for a camera frame timestamp.

There is no motor I/O or extrapolation here. Inference completion normally leaves
measurements on both sides, unlike sampling at inference submission. This does
not convert a camera receipt timestamp into a true hardware exposure timestamp.
"""
import math


def interpolate_pose(history, captured_at, now, max_gap=.1):
    if (not history or any(type(t) not in (int,float) or not math.isfinite(t)
                           for t in (captured_at,now)) or captured_at > now):
        return None
    latest = history[-1]
    if not 0 <= now-latest['read_completed_at'] <= max_gap:
        return None
    axes, completed, spans = {}, 0., {}
    for name in ('x','y'):
        left = right = None
        for sample in history:
            if sample['read_completed_at'] > now:
                break
            axis = sample['axes'][name]
            observed_at = axis.get('observed_at', sample['sampled_at'])
            if observed_at <= captured_at:
                left = (sample, axis, observed_at)
            if observed_at >= captured_at:
                right = (sample, axis, observed_at)
                break
        if left is None or right is None:
            return None
        sa,a,ta = left
        sb,b,tb = right
        if not 0 <= tb-ta <= max_gap:
            return None
        if any(a[k] != b[k] for k in ('origin','id','direction')):
            return None  # Never interpolate across a zero/identity change.
        fraction = (captured_at-ta)/(tb-ta) if tb > ta else 0.
        axes[name] = {**a, 'degrees':a['degrees']+fraction*(b['degrees']-a['degrees']),
                      'observed_at':captured_at}
        completed = max(completed, sa['read_completed_at'], sb['read_completed_at'])
        spans[name] = (tb-ta)*1000
    return {'sampled_at':captured_at, 'read_completed_at':completed, 'axes':axes,
            'interpolated':True, 'interpolation_span_ms':spans}
