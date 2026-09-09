"""The frozen detector's mask selection contract, independent of GPU hardware."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
try:
    import torch
except ImportError:
    torch=None
else:
    from spring_turret.sam31_mask_layers import BinaryMaskOutput


@unittest.skipIf(torch is None,'Torch required for mask output tests')
class MaskOutputTests(unittest.TestCase):
    def test_capacity_ties_empty_and_changed_inputs(self):
        torch.set_num_threads(2)
        output=BinaryMaskOutput(10,.5)
        logits=torch.full((1,200,1),-10.)
        presence=torch.full((1,1),10.)
        native=torch.ones((1,200,2,2))
        for count in (0,1,10,11,200,0):
            logits.fill_(-10);logits[0,:count,0]=2
            binary,scores,queries,retained,overflow=output(logits,presence,native)
            kept=min(count,10)
            self.assertEqual(int(retained),kept)
            self.assertEqual(int(overflow),count-kept)
            self.assertEqual(queries.tolist(),list(range(kept))+[-1]*(10-kept))
            self.assertTrue(binary[:kept].all())
            self.assertFalse(binary[kept:].any())
            self.assertTrue((scores[:kept]>.5).all())
            self.assertTrue((scores[kept:]==0).all())

    def test_strict_confidence_and_mask_threshold(self):
        output=BinaryMaskOutput()
        presence=torch.full((1,1),100.)
        logits=torch.zeros((1,200,1))
        native=torch.zeros((1,200,2,2))
        self.assertEqual(int(output(logits,presence,native)[3]),0)
        logits[0,7,0]=2
        binary,_,queries,count,_=output(logits,presence,native)
        self.assertEqual(int(count),1)
        self.assertEqual(int(queries[0]),7)
        self.assertFalse(binary.any()) # sigmoid(0)==.5 is not foreground.


if __name__=='__main__':unittest.main()
