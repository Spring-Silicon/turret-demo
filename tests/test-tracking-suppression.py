"""CPU checks of the GPU-decision/CPU-output handoff; no model or motor access."""
from collections import defaultdict
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spring_turret.tracking_suppression import install_tracking_suppression

class Mask:
    def __init__(self, values): self.values = values
    def cpu(self): return self
    def tolist(self): return list(self.values)

class Model:
    def __init__(self):
        self.metadata = {'obj_ids_all_gpu': [11, 42],
                         'rank0_metadata': {'suppressed_obj_ids': defaultdict(set)}}
        self.decision = Mask([True, False])
        self.run_hotstart = True
        self.fail_frame = False
    def _process_hotstart_gpu(self, *, frame_idx, tracker_metadata_prev):
        return Mask([False, False]), self.decision, {}
    def _det_track_one_frame(self, frame_idx):
        if self.run_hotstart:
            self._process_hotstart_gpu(frame_idx=frame_idx, tracker_metadata_prev=self.metadata)
            # The real planner compacts/replaces IDs after deciding suppression.
            self.metadata['obj_ids_all_gpu'] = [42, 99]
        if self.fail_frame: raise ValueError('interrupted frame')
        return {}, {}, [], self.metadata, {}, None

class SuppressionTests(unittest.TestCase):
    def test_hidden_ids_use_original_order_before_removal_and_new_births(self):
        model = Model(); install_tracking_suppression(model)
        model.metadata['rank0_metadata']['suppressed_obj_ids'][7].add(300)
        output = model._det_track_one_frame(7)
        self.assertIs(output[3], model.metadata)
        self.assertEqual(output[3]['rank0_metadata']['suppressed_obj_ids'][7], {11, 300})
        self.assertNotIn(99, output[3]['rank0_metadata']['suppressed_obj_ids'][7])

    def test_each_frame_and_prompt_gets_only_its_own_decisions(self):
        model = Model(); install_tracking_suppression(model)
        model._det_track_one_frame(0)
        model.decision = Mask([False, True])
        model._det_track_one_frame(1)
        hidden = model.metadata['rank0_metadata']['suppressed_obj_ids']
        self.assertEqual(hidden[0], {11})
        self.assertEqual(hidden[1], {99})
        model.metadata = {'obj_ids_all_gpu': [60, 70], 'rank0_metadata': {'suppressed_obj_ids': defaultdict(set)}}
        model._det_track_one_frame(0)
        self.assertEqual(model.metadata['rank0_metadata']['suppressed_obj_ids'][0], {70})

    def test_warmup_and_failed_frames_do_not_leak_previous_decisions(self):
        model = Model(); install_tracking_suppression(model)
        model.fail_frame = True
        with self.assertRaises(ValueError): model._det_track_one_frame(0)
        model.fail_frame = False; model.run_hotstart = False
        model._det_track_one_frame(1)
        self.assertEqual(dict(model.metadata['rank0_metadata']['suppressed_obj_ids']), {})

    def test_shape_mismatch_fails_instead_of_hiding_wrong_identity(self):
        model = Model(); install_tracking_suppression(model)
        model.decision = Mask([True])
        with self.assertRaisesRegex(RuntimeError, 'Misaligned'): model._det_track_one_frame(0)

    def test_install_once_and_empty_decisions(self):
        model = Model(); install_tracking_suppression(model)
        wrapped = model._det_track_one_frame
        install_tracking_suppression(model)
        self.assertIs(model._det_track_one_frame, wrapped)
        model.metadata['obj_ids_all_gpu'] = []; model.decision = Mask([])
        model._det_track_one_frame(0)
        self.assertEqual(model.metadata['rank0_metadata']['suppressed_obj_ids'][0], set())

if __name__ == '__main__': unittest.main()
