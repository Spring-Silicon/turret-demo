"""Track-capacity and alignment regressions; no GPU/servo needed."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spring_turret.track_lifetime import TrackLifetime, retire_tracks


class LifetimeTests(unittest.TestCase):
    def test_occlusion_grace_requires_both_time_and_frames(self):
        lifetime = TrackLifetime()
        self.assertFalse(lifetime.update([1], [1], 0, 0))
        self.assertFalse(lifetime.update([1], [], 1, 1))
        self.assertFalse(lifetime.update([1], [], 2, 20))
        self.assertEqual(lifetime.update([1], [], 17, 21), {1})

    def test_reappearance_resets_grace_and_removed_ids_do_not_accumulate(self):
        lifetime = TrackLifetime()
        for frame in range(1000):
            ids = list(range(frame, frame + 16))
            self.assertFalse(lifetime.update(ids, [], frame, frame))
            self.assertLessEqual(len(lifetime.missing), 16)
        lifetime = TrackLifetime()
        lifetime.update([1], [], 0, 0)
        lifetime.update([1], [1], 16, 6)
        self.assertFalse(lifetime.update([1], [], 17, 7))
        self.assertFalse(lifetime.update([1], [], 32, 12))
        self.assertEqual(lifetime.update([1], [], 33, 13), {1})

    def fixture(self):
        ids = np.array([2, 5, 9])
        mapping = dict.fromkeys(ids, 1)
        metadata = {
            "obj_ids_all_gpu":ids, "obj_ids_per_gpu":[ids.copy()],
            "num_obj_per_gpu":[3], "num_buc_per_gpu":[2], "max_obj_id":99,
            "obj_id_to_score":mapping.copy(), "obj_id_to_last_occluded":mapping.copy(),
            "obj_id_to_sam2_score_frame_wise":{0:mapping.copy()},
            "gpu_metadata":{"N_obj":3, "overlap_pair_counts":np.arange(9).reshape(3,3),
                **{k:np.array([20,50,90]) for k in ("obj_first_frame", "consecutive_unmatch_count",
                    "trk_keep_alive", "removed_mask", "last_occluded_tensor")}},
            "rank0_metadata":{
                "masklet_confirmation":{"status":np.array([0,1,2]), "consecutive_det_num":np.array([3,4,5])},
                "removed_obj_ids":{5}, "suppressed_obj_ids":{0:{2,5}},
                "obj_first_frame_idx":mapping.copy(), "unmatched_frame_inds":mapping.copy(),
                "trk_keep_alive":mapping.copy(), "overlap_pair_to_frame_inds":{(2,5):[0],(2,9):[0]}}}
        calls = []
        model = SimpleNamespace(world_size=1, rank=0,
            _tracker_remove_objects=lambda states, ids:calls.append(ids),
            _count_buckets_in_states=lambda states:1)
        return model, metadata, calls

    def test_retirement_compacts_every_indexed_table_without_reusing_ids(self):
        model, meta, calls = self.fixture()
        retire_tracks(model, [], meta, {5,700})
        self.assertEqual(calls, [[5]])
        self.assertEqual(meta["obj_ids_all_gpu"].tolist(), [2,9])
        self.assertEqual(meta["num_obj_per_gpu"], [2])
        self.assertEqual(meta["num_buc_per_gpu"], [1])
        self.assertEqual(meta["max_obj_id"], 99)
        self.assertEqual(meta["gpu_metadata"]["N_obj"], 2)
        self.assertEqual(meta["gpu_metadata"]["overlap_pair_counts"].tolist(), [[0,2],[6,8]])
        self.assertEqual(meta["rank0_metadata"]["masklet_confirmation"]["status"].tolist(), [0,2])
        self.assertNotIn(5, meta["obj_id_to_sam2_score_frame_wise"][0])
        self.assertEqual(meta["rank0_metadata"]["overlap_pair_to_frame_inds"], {(2,9):[0]})
        retire_tracks(model, [], meta, {2,9})
        self.assertEqual(meta["gpu_metadata"]["overlap_pair_counts"].shape, (0,0))
        self.assertEqual(meta["gpu_metadata"]["N_obj"], 0)
        self.assertEqual(meta["max_obj_id"], 99)

    def test_unknown_schema_fails_before_hardware_or_state_mutation(self):
        model, meta, calls = self.fixture()
        meta["gpu_metadata"]["future_field"] = np.array([1,2,3])
        with self.assertRaises(RuntimeError): retire_tracks(model, [], meta, {5})
        self.assertEqual(calls, [])
        self.assertEqual(meta["obj_ids_all_gpu"].tolist(), [2,5,9])


if __name__ == "__main__": unittest.main()
