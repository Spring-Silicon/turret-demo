import importlib.util
import io
import json
from pathlib import Path
import unittest
from urllib.request import Request

path = Path(__file__).resolve().parents[1] / 'deploy/startup/start-inference.py'
spec = importlib.util.spec_from_file_location('startup', path)
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)


class StartupTests(unittest.TestCase):
    def test_current_arc_startup_uses_only_direct_ethernet(self):
        repo=path.parents[2]
        unit=(repo/'deploy/startup/spring-turret-frontend.service').read_text()
        self.assertIn('--thor http://192.168.249.2:8080',unit)
        self.assertIn('--thor-interface enp7s0 --thor-local-address 192.168.249.1',unit)
        self.assertNotIn('100.88.90.48',unit)
        rules=(repo/'deploy/direct-ethernet/arc.nft').read_text()
        self.assertEqual(rules.count('ip daddr 192.168.249.2 oifname != "enp7s0"'),2)

    def run_case(self, prompts, failures=0, post_error=False, enabled=True):
        calls = []
        def open_url(request, timeout):
            calls.append(request)
            if isinstance(request, Request):
                self.assertEqual(request.full_url, 'http://127.0.0.1:8080/api/detection/prompts')
                self.assertEqual(request.method, 'POST')
                self.assertEqual(json.loads(request.data), {'prompts': ['person']})
                if post_error:
                    raise OSError('response lost')
                return io.BytesIO(b'{}')
            self.assertEqual(request, 'http://127.0.0.1:8080/api/status')
            if len(calls) <= failures:
                raise OSError('not ready')
            return io.BytesIO(json.dumps({'detection': {'enabled': enabled, 'prompts': prompts}}).encode())
        try:
            startup.start_inference(open_url, lambda _: None, attempts=3)
        finally:
            self.calls = calls

    def test_empty_prompts_seeded_once(self):
        self.run_case([])
        self.assertEqual(len(self.calls), 2)

    def test_user_prompts_preserved(self):
        self.run_case(['cup', 'bottle'])
        self.assertEqual(len(self.calls), 1)

    def test_only_readiness_gets_retry(self):
        self.run_case([], failures=2)
        self.assertEqual(len(self.calls), 4)

    def test_command_never_retried(self):
        with self.assertRaises(OSError):
            self.run_case([], post_error=True)
        self.assertEqual(len(self.calls), 2)

    def test_unavailable_backend_is_bounded(self):
        with self.assertRaises(RuntimeError):
            self.run_case([], failures=3)
        self.assertEqual(len(self.calls), 3)

    def test_disabled_inference_not_overridden(self):
        with self.assertRaises(RuntimeError):
            self.run_case([], enabled=False)
        self.assertEqual(len(self.calls), 1)


if __name__ == '__main__':
    unittest.main()
