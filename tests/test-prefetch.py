"""Concurrency/ownership checks without GPU or motor operations."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from spring_turret.prefetch import LatestPreparation, RequestInbox
from spring_turret.worker_protocol import encode_request, JPEG_BYTES
from spring_turret.detection import WorkerClient, DetectionController


def eventually(check):
    deadline = time.monotonic()+2
    while time.monotonic()<deadline:
        if check(): return
        time.sleep(.001)
    raise AssertionError('timeout')


class PrefetchTests(unittest.TestCase):
    def test_mask_preparation_checks_new_camera_frames_without_five_ms_delay(self):
        for model,interval in (('sam3.1-mask',.001),('sam3.1-tracking',.005),('sam3.1',.005)):
            worker=WorkerClient({'model':model})
            worker.process=SimpleNamespace(stdout=object())
            selector=MagicMock()
            selector.__enter__.return_value=selector
            selector.select.return_value=[]
            def tick():
                worker.pending=b'{"type":"result","id":1}\n'
            with patch('spring_turret.detection.selectors.DefaultSelector',return_value=selector):
                self.assertEqual(worker.receive(2,tick=tick)['id'],1)
            selector.select.assert_called_once_with(interval)

    def test_mask_wait_reuses_exact_inflight_work_and_mismatch_never_waits(self):
        entered,release=threading.Event(),threading.Event()
        p=LatestPreparation(lambda jpeg:(entered.set(),release.wait(1),jpeg)[2])
        self.addCleanup(p.close);self.addCleanup(release.set)
        p.submit('a',b'a');self.assertTrue(entered.wait(1))
        started=time.monotonic()
        self.assertIsNone(p.take('other',b'a',wait_seconds=.025))
        self.assertLess(time.monotonic()-started,.02)
        threading.Timer(.005,release.set).start()
        self.assertEqual(p.take('a',b'a',wait_seconds=.025),b'a')

    def test_latest_only_nonblocking_exact_token_and_bytes_and_close(self):
        entered, release = threading.Event(), threading.Event()
        def prepare(jpeg):
            if jpeg == b'a': entered.set(); release.wait(2)
            return [jpeg]
        p = LatestPreparation(prepare)
        self.addCleanup(p.close)
        p.submit('a', b'a'); self.assertTrue(entered.wait(1))
        self.assertIsNone(p.take('a', b'a'))  # Does not wait for preparation.
        p.submit('b', b'b'); p.submit('c', b'c')
        release.set()
        eventually(lambda: p.ready is not None)
        self.assertEqual(p.ready[0], 'c')  # Neither inflight a nor queued b wins.
        self.assertIsNone(p.take('b', b'c'))
        p.submit('d', b'd'); eventually(lambda: p.ready is not None)
        self.assertIsNone(p.take('d', b'wrong'))
        p.submit('e', b'e'); eventually(lambda: p.ready is not None)
        self.assertEqual(p.take('e', b'e'), [b'e'])
        self.assertIsNone(p.take('e', b'e'))
        p.close(); p.submit('f', b'f')
        self.assertIsNone(p.pending); self.assertIsNone(p.ready)

    def test_failed_speculation_falls_back_without_a_partial_buffer(self):
        def bad(jpeg): raise ValueError('bad JPEG')
        p = LatestPreparation(bad); self.addCleanup(p.close)
        p.submit('a', b'a')
        eventually(lambda: p.pending is None)
        self.assertIsNone(p.take('a', b'a'))

    def test_reader_does_not_wait_for_cpu_preparation_and_preserves_requests(self):
        release = threading.Event()
        p = LatestPreparation(lambda jpeg: release.wait(2))
        self.addCleanup(p.close); self.addCleanup(release.set)
        stream = io.BytesIO(encode_request({'kind':'prepare','token':'a'},b'a',JPEG_BYTES)
            + encode_request({'id':1},b'b',JPEG_BYTES))
        inbox = iter(RequestInbox(stream,p))
        self.assertEqual(next(inbox)['jpeg'], b'b')
        with self.assertRaises(StopIteration): next(inbox)
        self.assertIsNone(p.take('a',b'a'))
        release.set()

    def test_reader_propagates_malformed_frame(self):
        p=LatestPreparation(lambda jpeg: jpeg); self.addCleanup(p.close)
        with self.assertRaises(ValueError):
            next(iter(RequestInbox(io.BytesIO(b'not-json\n'),p)))

    def test_live_worker_pipe_prepares_without_inference_and_reuses_exact_frame(self):
        script = '''
import json, sys, time
from spring_turret.prefetch import LatestPreparation, RequestInbox
from spring_turret.worker_protocol import JPEG_BYTES
p=LatestPreparation(lambda jpeg: jpeg.hex())
print(json.dumps({'type':'ready','request_transport':JPEG_BYTES,'cpu_prefetch':True}),flush=True)
for r in RequestInbox(sys.stdin.buffer,p):
    value=p.take(r.get('prepared_token'),r['jpeg'])
    time.sleep(.04)
    print(json.dumps({'type':'result','id':r['id'],'prepared':value}),flush=True)
'''
        w=WorkerClient({})
        w.process=subprocess.Popen([sys.executable,'-u','-c',script],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,start_new_session=True,env={**os.environ,'PYTHONPATH':str(ROOT/'src')})
        self.addCleanup(w.stop)
        self.assertEqual(w.receive(5)['type'],'ready')
        self.assertTrue(w.prefetch_supported)
        source=lambda:{'token':'next','jpeg':b'second'}
        self.assertFalse(w.detect(1,b'cold',['chair'],None,prepare_next=source)['prefetch_sent'])
        result=w.detect(2,b'first',['chair'],None,prepare_next=source)
        self.assertTrue(result['prefetch_sent']); self.assertIsNone(result['prepared'])
        self.assertEqual(result['prefetch_frames_sent'], 1)  # Duplicate candidate not resent.
        self.assertEqual(w.detect(3,b'second',['chair'],None,prepared_token='next')['prepared'], b'second'.hex())
        self.assertIsNone(w.detect(4,b'third',['chair'],None,prepared_token='next')['prepared'])
        self.assertFalse(w.detect(5,b'changed',['cup'],None,prepare_next=source)['prefetch_sent'])
        candidates = []
        def advancing_camera():
            token = str(len(candidates))
            candidate = {'token':token, 'jpeg':token.encode()}
            candidates.append(candidate)
            return candidate
        result = w.detect(6,b'live',['cup'],None,prepare_next=advancing_camera)
        self.assertGreaterEqual(result['prefetch_frames_sent'], 2)
        self.assertEqual(result['prefetch_frames_sent'], len(candidates))
        latest = candidates[-1]
        result = w.detect(7,latest['jpeg'],['cup'],None,prepared_token=latest['token'])
        self.assertEqual(result['prepared'], latest['jpeg'].hex())

    def test_controller_uses_only_latest_frame_and_current_revision(self):
        for scenario in ('matching','newer_camera','changed_prompt'):
            with self.subTest(scenario=scenario):
                class Camera:
                    sequence=1
                    def wait_for_sample(self, previous, timeout):
                        return self.sequence, str(self.sequence).encode(), time.monotonic()
                    def status(self): return {'online':True}
                camera=Camera(); calls=[]
                class Worker:
                    prefetch_supported=True
                    def launch(self): pass
                    def receive(self, timeout): return {'type':'ready'}
                    def stop(self): pass
                    def detect(self, rid, jpeg, prompts, progress, **kwargs):
                        calls.append((rid,jpeg,kwargs.get('prepared_token'),prompts))
                        if len(calls)==1:
                            camera.sequence=2
                            candidate=kwargs['prepare_next']()
                            self.assert_candidate=candidate
                            if scenario=='newer_camera': camera.sequence=3
                            if scenario=='changed_prompt': controller.set_prompts(['cup'])
                        return {'type':'result','id':rid,'boxes':[],'torch_compile':True,
                                'sycl_graph':True,'client_overlay':True}
                worker=Worker()
                controller=DetectionController({'enabled':True},camera,lambda _:worker)
                controller.start()
                try:
                    controller.set_prompts(['chair'])
                    eventually(lambda: len(calls)>=2)
                    self.assertEqual(calls[1][2], '1:2:2' if scenario=='matching' else None)
                    self.assertEqual(calls[1][1], b'3' if scenario=='newer_camera' else b'2')
                    self.assertEqual(calls[1][3], ['cup'] if scenario=='changed_prompt' else ['chair'])
                finally: controller.stop()


if __name__=='__main__': unittest.main()
