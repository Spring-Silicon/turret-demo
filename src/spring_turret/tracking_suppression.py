"""Carry SAM's GPU hotstart suppression decisions into live frame metadata.

The pinned multiplex planner consumes its removal mask but drops the sibling
suppression mask. OnlineSession filters outputs using the CPU hidden-ID table,
so an unmatched mask can otherwise remain visible with its old detection score.
Keep the model's decisions unchanged and publish them after the frame's GPU work.
"""
from functools import wraps
import inspect


def install_tracking_suppression(model):
    if getattr(model, '_live_tracking_suppression', False):
        return
    hotstart = model._process_hotstart_gpu
    signature = inspect.signature(hotstart)
    frame = model._det_track_one_frame
    pending = []

    @wraps(hotstart)
    def remember(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        result = hotstart(*args, **kwargs)
        identities = tuple(map(int, bound['tracker_metadata_prev']['obj_ids_all_gpu']))
        pending.append((bound['frame_idx'], identities, result[1]))
        return result

    @wraps(frame)
    def publish(*args, **kwargs):
        pending.clear()
        try:
            result = frame(*args, **kwargs)
            metadata = result[3]
            for index, identities, mask in pending:
                decisions = mask.cpu().tolist()
                if len(decisions) != len(identities):
                    raise RuntimeError('Misaligned SAM suppression decisions')
                hidden = metadata['rank0_metadata']['suppressed_obj_ids'].setdefault(index, set())
                hidden.update(identity for identity, suppress in zip(identities, decisions, strict=True) if suppress)
            return result
        finally:
            pending.clear()

    model._process_hotstart_gpu = remember
    model._det_track_one_frame = publish
    model._live_tracking_suppression = True
