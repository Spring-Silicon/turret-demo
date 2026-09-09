"""Palette-buffer encoding preserves all bytes, including shape/palette edges."""
import base64
import io
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from spring_turret.mask_postprocess import encode_labels


def reference(labels,colors,max_bytes=384*1024):
    if not colors:return None
    image=Image.fromarray(labels).convert('P')
    palette=[0]*768
    for i,color in enumerate(colors,1):
        palette[i*3:i*3+3]=[int(color[j:j+2],16) for j in (1,3,5)]
    image.putpalette(palette)
    alpha=bytes([0]+[112]*len(colors)+[0]*(255-len(colors)))
    bits=next(b for b in (1,2,4,8) if b>=math.ceil(math.log2(len(colors)+1)))
    while True:
        output=io.BytesIO()
        image.save(output,format='PNG',bits=bits,transparency=alpha,compress_level=1)
        data=output.getvalue()
        if len(data)<=max_bytes:break
        image=image.resize((max(1,image.width//2),max(1,image.height//2)),Image.Resampling.NEAREST)
    return {'format':'indexed-png','png':base64.b64encode(data).decode('ascii'),
            'width':image.width,'height':image.height,
            'source_width':labels.shape[1],'source_height':labels.shape[0]}


class MaskPngTests(unittest.TestCase):
    def test_exact_png_for_palette_sizes_strides_and_changed_frames(self):
        rng=np.random.default_rng(17)
        for shape in ((1,1),(13,21),(720,1280)):
            for count in (1,3,10,16,80):
                colors=[f'#{i*29097%16777216:06x}' for i in range(count)]
                labels=rng.integers(0,count+1,shape,dtype=np.uint8)
                for view in (labels,labels[:,::-1]):
                    expected=reference(view,colors)
                    self.assertEqual(encode_labels(view,colors),expected)
                labels[:]=0
                self.assertEqual(encode_labels(labels,colors),reference(labels,colors))
        self.assertIsNone(encode_labels(np.zeros((1,1),dtype=np.uint8),[]))

    def test_payload_limit_and_result_owns_its_bytes(self):
        rng=np.random.default_rng(81)
        labels=rng.integers(0,11,(127,193),dtype=np.uint8)
        colors=['#55e8ce']*10
        with patch('spring_turret.tracking_masks.MAX_PNG_BYTES',1000):
            result=encode_labels(labels,colors)
            self.assertEqual(result,reference(labels,colors,1000))
        saved=dict(result)
        labels[:]=0
        encode_labels(labels,colors)
        self.assertEqual(result,saved)


if __name__=='__main__':unittest.main()
