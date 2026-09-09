"""Identity/dropout/ambiguity tests; no camera or motor I/O."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spring_turret.target_lock import Observation, TargetLock


def observation(identity=1, degrees=0., size=10.):
    theta = math.radians(degrees)
    return Observation({"instance_id": identity}, (math.sin(theta), 0., math.cos(theta)), size)


class TargetLockTests(unittest.TestCase):
    def setUp(self):
        self.lock = TargetLock()
        self.first = observation()
        self.lock.choose([self.first])

    def test_keeps_id_when_neighbor_moves_nearer_frame_center(self):
        self.assertEqual(self.lock.choose([observation(2, 2), self.first]), self.first)

    def test_missing_never_falls_back_to_far_neighbor_even_after_many_frames(self):
        for _ in range(2000):
            self.assertIsNone(self.lock.choose([observation(2, 15)]))
        self.assertEqual(self.lock.accepted, self.first)

    def test_original_id_reappearance_requires_three_fresh_observations(self):
        self.lock.choose([])
        for _ in range(2):
            self.assertIsNone(self.lock.choose([self.first]))
        self.assertEqual(self.lock.choose([self.first]), self.first)

    def test_changed_id_near_anchor_reacquires_after_confirmation(self):
        new = observation(99, 1.)
        self.assertIsNone(self.lock.choose([new]))
        self.assertIsNone(self.lock.choose([new]))
        self.assertEqual(self.lock.choose([new]), new)
        self.assertEqual(self.lock.accepted.identity, 99)

    def test_gap_and_changed_candidate_reset_consecutive_confirmation(self):
        a, b = observation(2, 1), observation(3, 1)
        self.lock.choose([a]); self.lock.choose([a])
        self.lock.choose([])
        self.assertIsNone(self.lock.choose([a]))
        self.assertIsNone(self.lock.choose([b]))
        self.assertIsNone(self.lock.choose([b]))
        self.assertEqual(self.lock.choose([b]), b)

    def test_two_similar_candidates_are_ambiguous_even_if_one_has_old_id(self):
        self.lock.choose([])
        for _ in range(10):
            self.assertIsNone(self.lock.choose([observation(1, .5), observation(2, .9)]))
        self.assertEqual(self.lock.reason, "ambiguous")
        self.assertEqual(self.lock.pending_count, 0)

    def test_continuous_same_id_big_jump_is_not_trusted_or_allowed_to_walk_anchor(self):
        for angle in (10.2, 14.9, 8.8, 6.1):
            self.assertIsNone(self.lock.choose([observation(1, angle)]))
            self.assertEqual(self.lock.accepted, self.first)

    def test_continuous_motion_is_unsmoothed_and_not_confirmation_delayed(self):
        for degrees in range(0, 90, 5):
            raw = observation(1, degrees)
            self.assertIs(self.lock.choose([raw]), raw)

    def test_size_changes_reject_reacquisition_and_gross_same_id_mask_change(self):
        self.assertIsNone(self.lock.choose([observation(1, size=30)]))
        for _ in range(5):
            self.assertIsNone(self.lock.choose([observation(2, size=30)]))
        self.assertEqual(self.lock.accepted, self.first)

    def test_active_temporal_id_never_replaced_but_retired_id_can_reassociate(self):
        new = observation(2, 1)
        for _ in range(5):
            self.assertIsNone(self.lock.choose([new], temporal_active=True))
        self.lock.choose([new], temporal_active=False)
        self.lock.choose([new], temporal_active=False)
        self.assertEqual(self.lock.choose([new], temporal_active=False), new)

    def test_new_click_does_not_acquire_other_object_while_clicked_id_absent(self):
        self.lock.reset()
        self.assertIsNone(self.lock.choose([self.first], identity=2))
        second = observation(2, 30)
        self.assertEqual(self.lock.choose([self.first, second], identity=2), second)

    def test_explicit_reselect_clears_memory_and_acquires_nearest_immediately(self):
        self.lock.reset()
        new = observation(2, 50)
        self.assertEqual(self.lock.choose([new, self.first]), new)


if __name__ == "__main__":
    unittest.main()
