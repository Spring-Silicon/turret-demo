"""Exact integer mask geometry/composition on the GPU, outside the model."""
import base64
import io
import math


def packed_mask_output(torch, width, height, *, size=1008):
    import numpy as np
    from PIL import Image
    if not 0 < width <= 8192 or not 0 < height <= 8192:
        raise ValueError('Invalid display dimensions')
    # Use PIL itself to freeze its nearest-neighbor coordinate convention. This
    # includes odd-size ties and differs from some GPU interpolation operators.
    ix = np.asarray(Image.fromarray(np.arange(size,dtype=np.int32)[None,:])
                    .resize((width,1),Image.Resampling.NEAREST)).copy().reshape(-1)
    iy = np.asarray(Image.fromarray(np.arange(size,dtype=np.int32)[:,None])
                    .resize((1,height),Image.Resampling.NEAREST)).copy().reshape(-1)
    # At 1008 square, even a completely filled mask's first moment is only
    # 511,588,224. Exact int32 accumulation is cheaper on Arc; retain int64 for
    # larger diagnostic inputs whose mathematical upper bound would overflow.
    moment_dtype = (torch.int32 if max(size*size, size*size*(size-1)//2) <= 2**31-1
                    else torch.int64)

    class Packed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('ix',torch.from_numpy(ix).long())
            self.register_buffer('iy',torch.from_numpy(iy).long())
            self.register_buffer('coordinate',torch.arange(size,dtype=moment_dtype))

        def forward(self,*outputs):
            masks=torch.cat(outputs[0::6],dim=0)
            queries=torch.cat(outputs[2::6],dim=0)
            valid=queries>=0
            # Exact integer moments from ORIGINAL masks, before resize/overlap.
            columns=masks.sum(dim=1,dtype=moment_dtype)
            rows=masks.sum(dim=2,dtype=moment_dtype)
            # Preserve the public int64 output contract, not just its values.
            area=columns.sum(dim=1,dtype=moment_dtype).long()
            x=(columns*self.coordinate).sum(dim=1,dtype=moment_dtype).long()
            y=(rows*self.coordinate).sum(dim=1,dtype=moment_dtype).long()
            display=masks[:,self.iy[:,None],self.ix[None,:]]
            display_area=display.sum(dim=(1,2),dtype=torch.int32)  # <= 8192 squared.
            n=masks.shape[0]
            slots=torch.arange(n,device=masks.device)
            ids=queries+(slots//outputs[0].shape[0])*200
            # Rank the valid masks exactly as sorted((-display_area,stable_id)).
            # Pairwise integer comparisons avoid host sync and dynamic lengths.
            earlier=(display_area[:,None]>display_area[None,:]) | (
                (display_area[:,None]==display_area[None,:]) & (
                    (ids[:,None]<ids[None,:]) | (
                        (ids[:,None]==ids[None,:]) & (slots[:,None]<slots[None,:]))))
            ranks=((earlier & valid[:,None]).sum(dim=0)+1)*valid
            labels=torch.where(display & valid[:,None,None],ranks[:,None,None],0).amax(dim=0).to(torch.uint8)
            # Metadata remains in original prompt/score order, not paint order.
            small=tuple(v for i,v in enumerate(outputs) if i%6!=0)
            return labels,area,x,y,ranks,*small
    return Packed()


def centroid_from_moments(area,x,y,*,size=1008):
    area=int(area)
    if area==0:return None
    return [(int(x)/area+.5)/size,(int(y)/area+.5)/size]


def validate_packed(cpu,outputs,width,height):
    """One-time actual-frame check against the unchanged CPU implementation."""
    import numpy as np
    from PIL import Image
    if __package__:
        from .tracking_masks import mask_centroid
    else:
        from tracking_masks import mask_centroid
    labels,areas,x,y,ranks,*small=cpu
    original=[value.cpu().numpy() for value in outputs]
    masks=[]
    for p in range(len(original)//6):
        binary,_,queries,count,_,_=original[p*6:p*6+6]
        for i in range(int(count)):
            slot=p*len(binary)+i
            if centroid_from_moments(areas[slot],x[slot],y[slot])!=mask_centroid(binary[i]):
                raise RuntimeError('GPU mask centroid differs from original')
            resized=np.asarray(Image.fromarray(binary[i]).resize((width,height),Image.Resampling.NEAREST),dtype=np.bool_)
            masks.append((resized,p*200+int(queries[i]),slot))
    expected=np.zeros((height,width),dtype=np.uint8)
    for rank,(mask,_,slot) in enumerate(sorted(masks,key=lambda item:(-np.count_nonzero(item[0]),item[1])),1):
        expected[mask]=rank
        if int(ranks[slot])!=rank:raise RuntimeError('GPU mask occlusion order differs')
    np.testing.assert_array_equal(labels,expected)
    for got,want in zip(small,[v for i,v in enumerate(original) if i%6!=0],strict=True):
        np.testing.assert_array_equal(got,want)


def encode_labels(labels,colors):
    """Same indexed-PNG bytes as the existing mask compositor."""
    from PIL import Image
    if __package__:
        from .tracking_masks import MAX_PNG_BYTES
    else:
        from tracking_masks import MAX_PNG_BYTES
    if not colors:return None
    height,width=labels.shape
    palette=[0]*768
    for index,color in enumerate(colors,1):
        palette[index*3:index*3+3]=[int(color[i:i+2],16) for i in (1,3,5)]
    # The labels already ARE palette indices; avoid converting a full-size
    # grayscale image and copying it. Borrow only through this synchronous save.
    if not labels.flags.c_contiguous:
        labels=labels.copy()
    image=Image.frombuffer('P',(width,height),labels,'raw','P',0,1)
    image.putpalette(palette)
    alpha=bytes([0]+[112]*len(colors)+[0]*(255-len(colors)))
    needed=math.ceil(math.log2(len(colors)+1))
    bits=next(b for b in (1,2,4,8) if b>=needed)
    while True:
        output=io.BytesIO()
        image.save(output,format='PNG',bits=bits,transparency=alpha,compress_level=1)
        data=output.getvalue()
        if len(data)<=MAX_PNG_BYTES:break
        image=image.resize((max(1,image.width//2),max(1,image.height//2)),Image.Resampling.NEAREST)
    return {'format':'indexed-png','png':base64.b64encode(data).decode('ascii'),
            'width':image.width,'height':image.height,'source_width':width,'source_height':height}
