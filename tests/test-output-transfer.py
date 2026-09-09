"""Readback ownership and synchronization order without accelerator hardware."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from spring_turret.output_transfer import OutputTransfer


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.events=[]
        outer=self
        class Buffer:
            def __init__(self,shape,**kwargs):
                outer.events.append(('allocate',shape,kwargs))
                self.value=None
            def copy_(self,source,**kwargs):
                outer.events.append(('copy',kwargs))
                self.value=source.value
            def numpy(self):
                outer.events.append(('numpy',))
                return self.value
        stream=SimpleNamespace(synchronize=lambda:self.events.append(('sync',)))
        self.transfer=OutputTransfer(SimpleNamespace(empty=Buffer),SimpleNamespace(current_stream=lambda:stream))
    def source(self,value,dtype='float32',shape=(10,)):
        return SimpleNamespace(value=value,shape=shape,dtype=dtype,device='gpu:0')
    def test_all_async_copies_precede_single_wait_and_cpu_access(self):
        self.assertEqual(self.transfer([self.source(1),self.source(2)]),[1,2])
        self.assertEqual([e[0] for e in self.events],['allocate','allocate','copy','copy','sync','numpy','numpy'])
        self.assertTrue(all(e[2]['pin_memory'] for e in self.events if e[0]=='allocate'))
        self.assertTrue(all(e[1]['non_blocking'] for e in self.events if e[0]=='copy'))
    def test_reuses_buffers_but_never_old_contents(self):
        self.transfer([self.source(1)])
        buffers=self.transfer.buffers
        self.events.clear()
        self.assertEqual(self.transfer([self.source(9)]),[9])
        self.assertIs(self.transfer.buffers,buffers)
        self.assertNotIn('allocate',[e[0] for e in self.events])
    def test_changed_count_shape_dtype_each_reallocates(self):
        for outputs in ([self.source(1)], [self.source(1),self.source(2)],
                        [self.source(3,shape=(2,))], [self.source(4,dtype='bool',shape=(2,))]):
            old=self.transfer.buffers
            self.transfer(outputs)
            self.assertIsNot(self.transfer.buffers,old)

if __name__=='__main__':unittest.main()
