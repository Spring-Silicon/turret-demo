"""One device-independent target policy. No Torch, CUDA, XPU or serial imports."""

POLICY_VERSION = "sam-shared-v2"
DEAD_BAND = .003
DETECTION_LOSS_GRACE_SECONDS = .2
TRACK_RETENTION_SECONDS = 5.0
TRACK_RETENTION_FRAMES = 16


def continuity(model):
    return "temporal-id" if model == "sam3.1-tracking" else "nearest-of-class"


def select_candidates(detection, prompt, instance_id, clicked):
    """Prefer an explicit/native ID; only live temporal IDs survive absence.

    Returning an empty candidate set with holding=True prevents a switch to a
    neighbor during occlusion. The worker retires inactive temporal identities.
    Box/Mask IDs are short-lived associations, so their absence permits immediate
    nearest-of-class reacquisition. A changed class can never inherit an ID.
    """
    boxes = [b for b in detection.get("boxes", []) if b.get("prompt") == prompt]
    if instance_id is None:
        return boxes, None, False, False
    selected = [b for b in boxes if b.get("instance_id") == instance_id]
    if selected:
        return selected, instance_id, clicked, False
    if detection.get("temporal_tracking") is True and instance_id in detection.get("active_instance_ids", [instance_id]):
        return [], instance_id, clicked, True
    return boxes, None, False, False
