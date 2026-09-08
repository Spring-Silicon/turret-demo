"""Optional graph-capturable FP16 SDPA, scoped to the shared image call."""
import torch
from torch.nn import functional as F
from torch.overrides import TorchFunctionMode
import triton
import triton.language as tl


@triton.jit
def _attention(Q,K,V,O,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
               QB:tl.constexpr,QH:tl.constexpr,QN:tl.constexpr,
               KB:tl.constexpr,KH:tl.constexpr,KN:tl.constexpr,
               VB:tl.constexpr,VH:tl.constexpr,VN:tl.constexpr,
               SCALE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    bh=tl.program_id(1); batch=bh//H; head=bh%H
    dim=tl.arange(0,D)
    q=tl.load(Q+batch*QB+head*QH+row[:,None]*QN+dim[None,:],row[:,None]<N,0)
    maximum=tl.full((BM,),float('-inf'),tl.float32)
    total=tl.full((BM,),0.,tl.float32)
    acc=tl.full((BM,D),0.,tl.float32)
    for start in range(tl.cdiv(N,BN)):
        col=start*BN+tl.arange(0,BN)
        k=tl.load(K+batch*KB+head*KH+col[None,:]*KN+dim[:,None],col[None,:]<N,0)
        score=tl.dot(q,k)*SCALE
        score=tl.where(col[None,:]<N,score,float('-inf'))
        new_max=tl.maximum(maximum,tl.max(score,1))
        correction=tl.exp2(maximum-new_max)
        probability=tl.exp2(score-new_max[:,None])
        total=total*correction+tl.sum(probability,1)
        acc=acc*correction[:,None]
        v=tl.load(V+batch*VB+head*VH+col[:,None]*VN+dim[None,:],col[:,None]<N,0)
        acc+=tl.dot(probability.to(q.dtype),v)
        maximum=new_max
    out=acc/total[:,None]
    tl.store(O+(bh*N+row[:,None])*D+dim[None,:],out,row[:,None]<N)


def image_sdpa(q,k,v,block_m=32,block_n=64):
    if q.shape!=k.shape or q.shape!=v.shape or q.ndim!=4 or q.shape[-1]!=64:
        raise ValueError('Image SDPA expects equal [B,H,N,64] tensors')
    if any(x.dtype!=torch.float16 or x.stride(-1)!=1 for x in (q,k,v)):
        raise ValueError('Image SDPA requires FP16 contiguous head dimensions')
    b,h,n,d=q.shape
    if n not in (576,5184):
        raise ValueError('Only the detector 24x24 windows and 72x72 global attention are supported')
    out=torch.empty((b,h,n,d),dtype=q.dtype,device=q.device)
    with getattr(torch,q.device.type).device(q.device):
        _attention[(triton.cdiv(n,block_m),b*h)](q,k,v,out,n,h,d,
            *q.stride()[:3],*k.stride()[:3],*v.stride()[:3],
            d**-0.5*1.4426950408889634,block_m,block_n,num_warps=4,num_stages=2)
    return out


class ImageAttention(TorchFunctionMode):
    def __torch_function__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        if func is F.scaled_dot_product_attention:
            if len(args)!=3 or kwargs:
                raise ValueError('Unsupported image attention options')
            return image_sdpa(*args)
        return func(*args,**kwargs)
