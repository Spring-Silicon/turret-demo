"""The compressed mask extension must be explicitly pinned and self-contained."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spring_turret import sam31_mask_attention8 as extension


class ExtensionTests(unittest.TestCase):
    def fixture(self, root, **changes):
        selection = {'recipe': 'skip4_attention8', 'skipped_blocks': [24,26,28,30], 'projection_variant': 0}
        selection.update(changes)
        (root / 'selection.json').write_text(json.dumps(selection))
        (root / 'manifest.json').write_text(json.dumps({'selection.json': extension.digest(root / 'selection.json')}))
        return extension.digest(root / 'manifest.json')

    def test_pinned_valid_extension(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(extension, 'MANIFEST', self.fixture(root)):
                self.assertEqual(extension.verify_extension(root)['recipe'], 'skip4_attention8')

    def test_source_drift_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(extension, 'MANIFEST', self.fixture(root)):
                (root / 'selection.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'artifact drift'):
                    extension.verify_extension(root)

    def test_manifest_drift_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                extension.verify_extension(root)

    def test_other_recipe_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(extension, 'MANIFEST', self.fixture(root, projection_variant=1)):
                with self.assertRaisesRegex(ValueError, 'Unexpected'):
                    extension.verify_extension(root)

    def test_escape_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps({'../outside': '0'*64}))
            with patch.object(extension, 'MANIFEST', extension.digest(root / 'manifest.json')):
                with self.assertRaisesRegex(ValueError, 'artifact drift'):
                    extension.verify_extension(root)


if __name__ == '__main__':
    unittest.main()
