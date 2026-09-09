"""Bounded live-stream identities; short occlusions retain their memory/ID."""


class TrackLifetime:
    def __init__(self, *, seconds=5.0, frames=16):
        self.seconds, self.frames = seconds, frames
        self.missing = {}

    def update(self, active, visible, frame, timestamp):
        active, visible = set(active), set(visible)
        for key in list(self.missing):
            if key not in active or key in visible:
                del self.missing[key]
        expired = set()
        for key in active - visible:
            first_frame, first_time = self.missing.setdefault(key, (frame, timestamp))
            if frame - first_frame >= self.frames and timestamp - first_time >= self.seconds:
                expired.add(key)
                del self.missing[key]
        return expired


def retire_tracks(model, trackers, metadata, retired):
    """Compact all CPU/GPU identity tables along with SAM's object memories.

    The upstream interactive remove_object path does not compact the multiplex
    GPU metadata. Using only that API leaves CPU IDs and GPU rows misaligned.
    This adapter is deliberately single-device and checks the schema before any
    mutation. IDs are never reused; max_obj_id is untouched.
    """
    import numpy as np
    ids = metadata["obj_ids_all_gpu"]
    retired = set(map(int, ids)) & set(retired)
    if not retired:
        return
    if model.world_size != 1 or model.rank != 0:
        raise RuntimeError("Live track retirement requires a single-device session")
    keep = np.flatnonzero(~np.isin(ids, list(retired))).tolist()
    gpu = metadata["gpu_metadata"]
    vectors = ("obj_first_frame", "consecutive_unmatch_count", "trk_keep_alive",
               "removed_mask", "last_occluded_tensor")
    expected = {"N_obj", "overlap_pair_counts", *vectors}
    if set(gpu) != expected or gpu["N_obj"] != len(ids):
        raise RuntimeError("Unsupported or misaligned SAM multiplex metadata")
    if any(len(gpu[key]) != len(ids) for key in vectors) or tuple(gpu["overlap_pair_counts"].shape) != (len(ids), len(ids)):
        raise RuntimeError("Misaligned SAM multiplex identity tensors")
    rank = metadata["rank0_metadata"]
    confirmation = rank["masklet_confirmation"]
    if any(len(value) != len(ids) for value in confirmation.values()):
        raise RuntimeError("Misaligned SAM confirmation metadata")
    compact = {key: gpu[key][keep] for key in vectors}
    compact["overlap_pair_counts"] = gpu["overlap_pair_counts"][keep][:, keep]
    compact["N_obj"] = len(keep)
    model._tracker_remove_objects(trackers, sorted(retired))
    metadata["obj_ids_all_gpu"] = ids[keep]
    metadata["obj_ids_per_gpu"] = [ids[keep].copy()]
    metadata["num_obj_per_gpu"][0] = len(keep)
    metadata["num_buc_per_gpu"][0] = model._count_buckets_in_states(trackers)
    metadata["gpu_metadata"] = compact
    for key in confirmation:
        confirmation[key] = confirmation[key][keep]
    for mapping in (metadata["obj_id_to_score"], metadata["obj_id_to_last_occluded"],
                    *metadata["obj_id_to_sam2_score_frame_wise"].values(),
                    rank["obj_first_frame_idx"], rank["unmatched_frame_inds"], rank["trk_keep_alive"]):
        for key in retired:
            mapping.pop(key, None)
    rank["removed_obj_ids"].difference_update(retired)
    for suppressed in rank["suppressed_obj_ids"].values():
        suppressed.difference_update(retired)
    for pair in list(rank["overlap_pair_to_frame_inds"]):
        if retired.intersection(pair):
            del rank["overlap_pair_to_frame_inds"][pair]
