"""Regression: per-frame graph pixels must never share SAM's identity key."""
import argparse
import io
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from PIL import Image
from spring_turret.tracking_preprocess import VideoPreprocessor

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',required=True)
    args=parser.parse_args();device=torch.device(args.device)
    runtime=getattr(torch,device.type);runtime.set_device(device)
    preprocess=VideoPreprocessor(torch,device,image_size=672)
    def jpeg(color):
        buf=io.BytesIO();Image.new('RGB',(321,179),color).save(buf,format='JPEG');return buf.getvalue()
    red,green=jpeg((220,10,10)),jpeg((10,220,10))
    first,_=preprocess(red);saved=first.clone()
    second,_=preprocess(green)
    assert first is not second and first.data_ptr()!=second.data_ptr()
    assert torch.equal(first,saved),'Replaying frame B mutated frame A'
    assert not torch.equal(first,second),'New image pixels failed to change'
    third,_=preprocess(green,prepared=preprocess.prepare_cpu(green))
    assert third is not second and torch.equal(second,third)
    # SAM caches image features with `image is previous_image`, not contents.
    # Even identical repeated frames require a new causal frame identity.
    assert third.data_ptr()!=second.data_ptr()
    print('fresh identity, immutable old frame, exact prepared pixels: passed')

if __name__=='__main__':main()
