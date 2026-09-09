"""Integer composition must preserve centroid, occlusion, and PNG contracts."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
try:
    import torch
except ImportError:
    torch=None
from PIL import Image
from spring_turret.mask_postprocess import packed_mask_output,centroid_from_moments,encode_labels
from spring_turret.tracking_masks import mask_centroid,encode_mask_overlay,instance_color
from spring_turret.output_transfer import OutputTransfer
from spring_turret.sam31_graph import CompiledStage

def reference(outputs,width,height):
    cpu=[v.cpu().numpy() if isinstance(v,torch.Tensor) else v for v in outputs]
    masks=[];centroids=[];slot_colors={}
    for p in range(len(cpu)//6):
        binary,scores,queries,count,omitted,boxes=cpu[p*6:p*6+6]
        for i,mask in enumerate(binary):
            centroids.append(mask_centroid(mask) if i<int(count) else None)
            if i<int(count):
                color=instance_color(('#55e8ce','#ffb86b')[p%2],int(queries[i]))
                display=np.asarray(Image.fromarray(mask).resize((width,height),Image.Resampling.NEAREST),dtype=np.bool_)
                slot=p*len(binary)+i
                masks.append((display,color,p*200+int(queries[i]),slot))
                slot_colors[slot]=color
    ordered=sorted(masks,key=lambda v:(-np.count_nonzero(v[0]),v[2]))
    ranks=np.zeros(len(centroids),dtype=np.int64)
    for rank,(_,color,_,slot) in enumerate(ordered,1):ranks[slot]=rank
    return encode_mask_overlay([v[:3] for v in masks],width,height),centroids,ranks,slot_colors,cpu

def check(outputs,packed,width,height,size):
    labels,areas,x,y,ranks,*small=[v.cpu().numpy() for v in packed]
    overlay,centroids,expected_ranks,slot_colors,cpu=reference(outputs,width,height)
    np.testing.assert_array_equal(ranks,expected_ranks)
    got_centroids=[centroid_from_moments(a,b,c,size=size) for a,b,c in zip(areas,x,y)]
    assert got_centroids==centroids,(got_centroids,centroids)
    palette={int(ranks[slot]):color for slot,color in slot_colors.items()}
    actual=encode_labels(labels,[palette[i] for i in range(1,len(palette)+1)])
    assert actual==overlay,'PNG bytes/palette/overlap differ'
    for got,want in zip(small,[v for i,v in enumerate(cpu) if i%6!=0],strict=True):
        np.testing.assert_array_equal(got,want)

def make_outputs(device,size,counts,seed=17):
    generator=torch.Generator().manual_seed(seed)
    outputs=[]
    for p,count in enumerate(counts):
        masks=torch.rand((10,size,size),generator=generator)>.75
        masks[1]=masks[0] # Equal-area overlapping masks exercise stable ID ties.
        masks[2]=False # Empty retained mask still has a rank and no centroid.
        masks[count:]=False
        ids=torch.tensor([11,3,7,2,4,12,1,6,14,8]);ids[count:]=-1
        outputs.extend((masks.to(device),torch.rand(10,generator=generator).to(device),ids.to(device),
                        torch.tensor(count,dtype=torch.int32,device=device),
                        torch.tensor(max(count-8,0),dtype=torch.int32,device=device),
                        torch.rand((10,4),generator=generator).to(device)))
    return tuple(outputs)

def legacy_mask_work(cpu,width,height):
    # Only the previous worker's actual work: no reference-check bookkeeping,
    # no duplicate sorting, and no scans of unused padded mask slots.
    instances=[]
    for p in range(len(cpu)//6):
        binary,_,queries,count,_,_=cpu[p*6:p*6+6]
        for i in range(int(count)):
            mask_centroid(binary[i])
            color=instance_color(('#55e8ce','#ffb86b')[p%2],int(queries[i]))
            resized=np.asarray(Image.fromarray(binary[i]).resize((width,height),Image.Resampling.NEAREST),dtype=np.bool_)
            instances.append((resized,color,p*200+int(queries[i])))
    return encode_mask_overlay(instances,width,height)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu')
    parser.add_argument('--benchmark',action='store_true');args=parser.parse_args()
    if torch is None:
        if args.device!='cpu' or args.benchmark:raise RuntimeError('Torch required for GPU qualification')
        print('SKIP mask postprocessing: Torch is not installed')
        return
    torch.set_num_threads(4);device=torch.device(args.device)
    with torch.inference_mode():
        # Odd sizes, noninteger scales, empties, overlap, multiple classes.
        for width,height in ((21,13),(9,25),(1,1)):
            module=packed_mask_output(torch,width,height,size=17).to(device)
            for counts in ((0,0),(1,3),(10,10),(2,0)):
                outputs=make_outputs(device,17,counts)
                check(outputs,module(*outputs),width,height,17)
        # A full mask maximizes area and first moments. Exercise both the exact
        # int32 fast path and the bound-selected int64 fallback.
        for size in (1008,2048):
            module=packed_mask_output(torch,1,1,size=size).to(device)
            mask=torch.ones((1,size,size),dtype=torch.bool,device=device)
            outputs=(mask,torch.ones(1,device=device),torch.zeros(1,dtype=torch.int64,device=device),
                     torch.tensor(1,dtype=torch.int32,device=device),
                     torch.tensor(0,dtype=torch.int32,device=device),torch.ones((1,4),device=device))
            _,area,x,y,*_=module(*outputs)
            assert all(v.dtype==torch.int64 for v in (area,x,y))
            assert int(area[0])==size*size
            assert int(x[0])==int(y[0])==size*size*(size-1)//2
        print('exact eager centroids, ranks, metadata, PNG bytes: passed',flush=True)
        if not args.benchmark:return
        runtime=getattr(torch,device.type);runtime.set_device(device)
        module=packed_mask_output(torch,1280,720).to(device)
        stage=CompiledStage(torch,module,'mask-postprocess',lambda *s:print(s,flush=True))
        outputs=make_outputs(device,1008,(10,))
        check(outputs,stage(*outputs),1280,720,1008)
        # Owned capture must not retain old masks as values change.
        outputs=make_outputs(device,1008,(4,),seed=81)
        check(outputs,stage(*outputs),1280,720,1008)
        transfer=OutputTransfer(torch,runtime)
        old_transfer=OutputTransfer(torch,runtime)
        samples={'old':[],'new':[]}
        for iteration in range(35):
            for name in (('old','new') if iteration%2 else ('new','old')):
                runtime.synchronize();start=time.perf_counter()
                if name=='old':legacy_mask_work(old_transfer(outputs),1280,720)
                else:
                    labels,areas,x,y,ranks,*small=transfer(stage(*outputs))
                    colors={int(ranks[i]):instance_color('#55e8ce',int(small[1][i])) for i in range(int(small[2]))}
                    encode_labels(labels,[colors[i] for i in range(1,len(colors)+1)])
                    for i in range(int(small[2])):centroid_from_moments(areas[i],x[i],y[i])
                runtime.synchronize()
                if iteration>=5:samples[name].append((time.perf_counter()-start)*1000)
        print(json.dumps({'bitwise_equal':True,'median_ms':{k:statistics.median(v) for k,v in samples.items()}}),flush=True)
        multi=CompiledStage(torch,packed_mask_output(torch,1280,720).to(device),
                            'mask-postprocess-two-classes',lambda *s:print(s,flush=True))
        for counts in ((3,2),(0,0),(10,10)):
            outputs=make_outputs(device,1008,counts)
            check(outputs,multi(*outputs),1280,720,1008)
        print('compiled multi-class, zero-to-capacity transitions: passed',flush=True)

if __name__=='__main__':main()
