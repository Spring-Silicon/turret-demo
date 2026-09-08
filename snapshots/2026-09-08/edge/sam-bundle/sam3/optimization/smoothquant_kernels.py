"""Fused activation quantization and INT32 epilogue for the image-only path."""
import torch
import triton
import triton.language as tl


@triton.jit
def _quant(X, S, Q, A, K: tl.constexpr, BF16: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row*K + col, col < K, 0)
    if BF16:
        x = x.to(tl.bfloat16).to(tl.float32)
    else:
        x = x.to(tl.float16).to(tl.float32)
    smooth = tl.load(S + col, col < K, 1)
    x = x / smooth
    scale = tl.maximum(tl.max(tl.abs(x), 0), 1.e-5) / 127.
    rounded = tl.extra.libdevice.nearbyint(x / scale)
    q = tl.minimum(tl.maximum(rounded, -127.), 127.).to(tl.int8)
    tl.store(Q + row*K + col, q, col < K)
    tl.store(A + row, scale)


@triton.jit
def _dequant(C, A, W, BIAS, Y, N: tl.constexpr, TOTAL: tl.constexpr,
             GELU: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    mask = idx < TOTAL
    c = tl.load(C + idx, mask, 0).to(tl.float32)
    a = tl.load(A + idx//N, mask, 0)
    w = tl.load(W + idx%N, mask, 0)
    b = tl.load(BIAS + idx%N, mask, 0)
    y = c*a*w+b
    if GELU:
        y = y.to(tl.bfloat16).to(tl.float32)
        y = 0.5*y*(1.+tl.erf(y*0.7071067811865476))
    tl.store(Y+idx, y, mask)


def quantize(x, smooth, operand_dtype):
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    m, k = flat.shape
    q = torch.empty((m,k),dtype=torch.int8,device=x.device)
    scale = torch.empty((m,1),dtype=torch.float32,device=x.device)
    with getattr(torch,x.device.type).device(x.device):
        _quant[(m,)](flat,smooth,q,scale,k,operand_dtype==torch.bfloat16,
                     triton.next_power_of_2(k), enable_fp_fusion=False)
    return q, scale


def dequantize(accum, a, w, bias, dtype, gelu):
    result = torch.empty(accum.shape,dtype=dtype,device=accum.device)
    with getattr(torch,accum.device.type).device(accum.device):
        _dequant[(triton.cdiv(accum.numel(),1024),)](
            accum,a,w,bias,result,accum.shape[1],accum.numel(),gelu,1024,
            enable_fp_fusion=False)
    return result
