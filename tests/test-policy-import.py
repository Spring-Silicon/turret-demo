"""Policy imports work in both the backend package and file-launched workers."""
from pathlib import Path
import subprocess
import sys
import unittest


class PolicyImportTests(unittest.TestCase):
    def test_package_and_standalone_worker_imports(self):
        source = Path(__file__).resolve().parents[1] / "src"
        for standalone in (False, True):
            with self.subTest(standalone=standalone):
                directory = source / "spring_turret" if standalone else source
                prefix = "" if standalone else "spring_turret."
                code = (
                    f"import sys; sys.path.insert(0, {str(directory)!r}); "
                    f"from {prefix}policy import continuity, TRACK_RETENTION_SECONDS, TRACK_RETENTION_FRAMES; "
                    f"from {prefix}session_policy import ManagedSession; "
                    "assert continuity('sam3.1-mask') == 'nearest-of-class'; "
                    "assert continuity('sam3.1-tracking') == 'temporal-id'; "
                    "assert continuity('sam3.1-v18') == 'temporal-id'; "
                    "assert (TRACK_RETENTION_SECONDS, TRACK_RETENTION_FRAMES) == (5.0, 16); "
                    "assert 'torch' not in sys.modules"
                )
                result = subprocess.run([sys.executable, "-I", "-c", code],
                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
