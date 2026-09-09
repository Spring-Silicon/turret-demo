"""Destination-only frozen-output and multi-prompt qualification; no motor access."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spring_turret import sam31_worker as common
from spring_turret.sam31_w4a4_worker import W4A4Engine
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bundle',type=Path,required=True)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--device',type=int,default=0)
args=parser.parse_args()
common._PROTOCOL_OUTPUT = sys.stdout
bundle = args.bundle
engine = W4A4Engine(args.checkpoint, bundle, args.device,
                   allow_accuracy_tradeoff=True)
torch = engine.torch
expected = json.loads((bundle / 'fixtures/expected.json').read_text())
checks, timings = [], []
first = next(iter(expected))
with torch.inference_mode(), torch.autocast('xpu', dtype=torch.float16):
    from safetensors.torch import load_file
    cache = load_file(bundle / 'repository/outputs/sam31-turret-detector/model-comparison-20260907/fixtures/cache-0.safetensors')
    encoded, = engine._encode_prompts(['person'])
    text_error = float((encoded[0].cpu().float()-cache['memory'].float()).abs().max())
    print(json.dumps({'person_text_error':text_error, 'mask_equal':torch.equal(encoded[1].cpu(),cache['mask'])}),flush=True)
    torch.testing.assert_close(encoded[0].cpu(),cache['memory'],rtol=0,atol=.001)
    assert torch.equal(encoded[1].cpu(),cache['mask'])
    for key in [*expected, first]:
        jpeg = (bundle / f'fixtures/{key}.jpg').read_bytes()
        pixels = engine.preprocessor(jpeg)
        import io
        reference_pixels = engine.preprocessor.reference(engine.preprocessor.Image.open(io.BytesIO(jpeg)).convert('RGB')).unsqueeze(0)
        assert torch.equal(pixels.cpu(), reference_pixels), f'Preprocessing drift on {key}'
        actual = engine.run_model(pixels, ['person'])
        logits, boxes, presence = actual
        scores = (logits.sigmoid() * presence.sigmoid().unsqueeze(1)).squeeze(-1)[0].float().cpu()
        box_values = boxes[0].float().cpu()
        target_scores = torch.tensor(expected[key]['scores'])
        target_boxes = torch.tensor(expected[key]['boxes'])
        score_error = (scores - target_scores).abs().max().item()
        box_error = (box_values - target_boxes).abs().max().item()
        assert torch.equal(scores > .5, target_scores > .5), 'Changed retained detections'
        active = (scores >= .47) | (target_scores >= .47)
        # Rejected/no-object queries are diagnostics, not displayed detections.
        # Keep the original app's .47 near-threshold scope, with tighter .002
        # confidence and .001 box gates for source-candidate copy qualification.
        torch.testing.assert_close(scores[active], target_scores[active], rtol=0, atol=.002)
        torch.testing.assert_close(box_values[active], target_boxes[active], rtol=0, atol=.001)
        assert torch.isfinite(box_values).all() and torch.isfinite(scores).all()
        # Compare graph replay to the very same compiled candidate on changing frames.
        direct = engine.model_stage.compiled(*engine.model_stage.inputs)
        for a, b in zip(actual, direct, strict=True):
            torch.testing.assert_close(a, b, rtol=.001, atol=.001)
        check = dict(image_id=key, max_score_error=score_error, max_box_error=box_error,
                     active_box_error=float((box_values[active]-target_boxes[active]).abs().max()) if active.any() else 0,
                     exact_scores=torch.equal(scores,target_scores), exact_boxes=torch.equal(box_values,target_boxes))
        checks.append(check)
        print(json.dumps(check), flush=True)
    # Newly typed prompts, same embedding shapes; multiple independent heads share the image.
    for prompts in (['person'], ['table'], ['person', 'table'], ['person']):
        for i in range(13):
            r = engine.detect_many(jpeg, prompts, client_overlay=True)
            assert [c['prompt'] for c in r['categories']] == prompts
            if i >= 3:
                timings.append(dict(prompts=prompts, timing=r['timing'], latency_ms=r['latency_ms']))
        print(json.dumps({'prompts':prompts,'last_timing':r['timing']}), flush=True)
result = dict(passed=True, checks=checks, timings=timings, validation=engine.validation,
              note='Copy equivalence and replay checks, not a new ground-truth AP evaluation')
out=args.output
out.write_text(json.dumps(result, indent=2)+'\n')
print('PASS '+str(out), flush=True)
