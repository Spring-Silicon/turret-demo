"""No GPU: framing, negotiation, exact payloads and legacy compatibility."""
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from spring_turret.worker_protocol import JPEG_BYTES, MAX_JPEG, encode_request, iter_requests
from spring_turret.detection import WorkerClient


class ShortReads(io.BytesIO):
    def read(self, length=-1):
        return super().read(min(length, 3))


class ProtocolTests(unittest.TestCase):
    def test_adjacent_binary_and_legacy_frames_preserve_arbitrary_bytes(self):
        jpeg = bytes(range(256))*7 + b'\n{"id":999}\n'
        payload = b''.join(encode_request({'id':i,'prompts':['chair']}, jpeg+bytes([i]), mode)
                           for i,mode in enumerate([JPEG_BYTES, 'json-base64', JPEG_BYTES]))
        requests = list(iter_requests(ShortReads(payload)))
        self.assertEqual([r['id'] for r in requests], [0,1,2])
        self.assertEqual([r['jpeg'] for r in requests], [jpeg+bytes([i]) for i in range(3)])

    def test_invalid_lengths_transports_and_truncation_are_fatal(self):
        for length in (None, True, 0, -1, 1.2, '3', MAX_JPEG+1):
            header = json.dumps({'transport':JPEG_BYTES, 'jpeg_length':length}).encode()+b'\n'
            with self.assertRaises(ValueError): list(iter_requests(io.BytesIO(header)))
        for data in (b'{}', b'[]\n', b'{broken}\n', b'{"transport":"unknown"}\n',
                     b'{"jpeg":"bad*"}\n', b'{"jpeg":""}\n'):
            with self.assertRaises((ValueError, KeyError)): list(iter_requests(io.BytesIO(data)))
        with self.assertRaises(EOFError):
            list(iter_requests(io.BytesIO(b'{"transport":"jpeg-bytes-v1","jpeg_length":10}\nshort')))
        with self.assertRaises(ValueError):
            list(iter_requests(io.BytesIO(b'{"transport":"jpeg-bytes-v1","jpeg_length":1,"jpeg":"x"}\nx')))

    def test_encoder_rejects_empty_oversized_and_nonbyte_inputs(self):
        for jpeg in (b'', b'x'*(MAX_JPEG+1), 'jpeg', None):
            with self.assertRaises(ValueError): encode_request({}, jpeg, JPEG_BYTES)
        with self.assertRaises(ValueError): encode_request({'prompt':'x'*65536}, b'jpeg', JPEG_BYTES)

    def test_raw_payload_avoids_base64_size_expansion(self):
        jpeg=b'x'*120000
        raw=encode_request({'id':1},jpeg,JPEG_BYTES)
        legacy=encode_request({'id':1},jpeg,'json-base64')
        self.assertLess(len(raw), len(jpeg)+100)
        self.assertGreater(len(legacy), len(jpeg)*4/3)

    def test_live_pipe_negotiation_ordered_responses_and_legacy_fallback(self):
        script = '''
import hashlib, json, sys
from spring_turret.worker_protocol import iter_requests, JPEG_BYTES
ready={'type':'ready'}
if sys.argv[1]=='binary': ready['request_transport']=JPEG_BYTES
print(json.dumps(ready),flush=True)
for r in iter_requests(sys.stdin.buffer):
    print(json.dumps({'type':'result','id':r['id'],'hash':hashlib.sha256(r['jpeg']).hexdigest()}),flush=True)
'''
        import hashlib
        for mode in ('binary','legacy'):
            client=WorkerClient({})
            client.process=subprocess.Popen([sys.executable,'-u','-c',script,mode],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,start_new_session=True,
                env={**os.environ,'PYTHONPATH':str(ROOT/'src')})
            try:
                self.assertEqual(client.receive(5)['type'],'ready')
                for i in range(3):
                    jpeg=bytes(range(256))*4096+bytes([i])
                    result=client.detect(i,jpeg,['chair'],None)
                    self.assertEqual(result['hash'],hashlib.sha256(jpeg).hexdigest())
                    self.assertEqual(result['request_transport'],JPEG_BYTES if mode=='binary' else 'json-base64')
            finally: client.stop()


if __name__=='__main__': unittest.main()
