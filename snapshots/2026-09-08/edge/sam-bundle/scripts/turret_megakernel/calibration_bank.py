"""Load frozen calibration artifacts and enforce their explicit fitting split."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/'results/turret_megakernel'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''): digest.update(chunk)
    return digest.hexdigest()


def add_arguments(parser):
    for name, suffix in (('manifest', 'manifest.json'), ('features', 'features.pt'), ('references', 'refs.pt')):
        parser.add_argument('--calibration-'+name, type=Path, default=RESULTS/('downstream_calibration_'+suffix))
    parser.add_argument('--calibration-split', type=Path,
                        help='Frozen sequence split; required for any nonlegacy calibration bank')


def load(args):
    paths = {name: getattr(args, 'calibration_'+name) for name in ('manifest', 'features', 'references')}
    manifest = json.loads(paths['manifest'].read_text())
    development = json.loads((RESULTS/'all_manifest.json').read_text())
    count = len(manifest)
    assert count and len({r['sha256'] for r in manifest}) == count
    assert not {r['sha256'] for r in manifest} & {r['sha256'] for r in development}
    assert all(sha(r['path']) == r['sha256'] and r['prompts'] for r in manifest)
    if args.calibration_split is None:
        assert paths['manifest'].resolve() == (RESULTS/'downstream_calibration_manifest.json').resolve(), 'Nonlegacy bank requires an explicit frozen split'
        assert count == 16
        fit, evaluation = list(range(0, count, 2)), list(range(1, count, 2))
        split = {'kind': 'legacy_alternating_frames', 'fit_frame_indices': fit, 'evaluation_frame_indices': evaluation}
    else:
        paths['split'] = args.calibration_split
        split = json.loads(paths['split'].read_text())
        assert split['frozen_before_extraction'] and split['manifest_sha256'] == sha(paths['manifest'])
        fit, evaluation = split['fit_frame_indices'], split['evaluation_frame_indices']
        assert fit and evaluation and len(fit)+len(evaluation) == count
        assert set(fit).isdisjoint(evaluation) and set(fit+evaluation) == set(range(count))
        fit_sequences = {Path(manifest[i]['path']).parent.name for i in fit}
        evaluation_sequences = {Path(manifest[i]['path']).parent.name for i in evaluation}
        excluded = {Path(r['path']).parent.name for r in development}
        assert fit_sequences.isdisjoint(evaluation_sequences)
        assert (fit_sequences | evaluation_sequences).isdisjoint(excluded)
        assert fit_sequences == set(split['fit_sequences']) and evaluation_sequences == set(split['evaluation_sequences'])
        assert excluded == set(split['excluded_development_sequences'])
        for partition, indices in (('fit', fit), ('evaluation', evaluation)):
            assert all(manifest[i]['partition'] == partition and manifest[i]['sequence'] == Path(manifest[i]['path']).parent.name for i in indices)
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        bank = torch.load(paths['features'], map_location='cpu', weights_only=True)
        references = torch.load(paths['references'], map_location='cpu', weights_only=True)
    assert bank['identity'] == references['identity']
    assert bank['identity']['manifest_sha256'] == hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    assert len(bank['features']) == len(references['features']) == len(references['pixels']) == count
    assert set(references['refs']) == {(i, j) for i, row in enumerate(manifest) for j in range(len(row['prompts']))}
    for pairs in (bank['features'], references['features']):
        assert all(len(pair) == 2 and all(v.shape == (1, 256, 72, 72) and torch.isfinite(v).all() for v in pair) for pair in pairs)
    assert all(v.shape == (1, 3, 1008, 1008) and v.dtype == torch.float32 and torch.isfinite(v).all() for v in references['pixels'])
    assert all(len(values) == 3 and all(torch.isfinite(v).all() for v in values) for values in references['refs'].values())
    return SimpleNamespace(manifest=manifest, bank=bank, references=references, fit=fit, evaluation=evaluation,
                           split=split, paths=paths, hashes={str(p): sha(p) for p in paths.values()})
