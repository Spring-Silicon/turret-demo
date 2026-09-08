"""Install the latest retained image implementation without changing calibration."""
from contextlib import contextmanager
from calibration_image import retained_image
from calibration_bank import ROOT, sha
from native_mlp_owned import Workspace
from native_qkv_rope import prepare


@contextmanager
def current_image(engine):
    mlp = ROOT/'runtime/joint_mlp_cached_recip_vec16.so'
    qkv = ROOT/'runtime/joint_qkv_rope_grouped.so'
    assert sha(mlp) == 'bbb4c548e66b46dffbafed93f8bf01656d86ad0d5490f67a93e3039ab4aa360e'
    assert sha(qkv) == '3f5f5d2f2ab49fe0ed5baf98078ace1ecaf17b2d030ea9e747cf0036aa0f848b'
    with retained_image(engine) as current:
        for block in current.trunk.blocks:
            linear = block.attn.qkv
            packed, path = prepare(linear.packed_weight_t, qkv)
            linear.register_buffer('native_qkv_weight', packed)
            linear.native_qkv_library = path
            old = block.mlp.workspace
            work = Workspace(5184,1024,4736,1024,engine.device,shared=old,
                             **{**old.config,'native_owned_library':str(mlp)})
            assert work.state.shape == old.state.shape
            work.state = old.state; work.packed = old.packed; work.original_weights = old.original_weights
            work.prepare_smooth(block.mlp.fc2.smooth_scale)
            block.mlp.workspace = work
        try:
            yield current
        finally:
            for block in current.trunk.blocks:
                delattr(block.attn.qkv, 'native_qkv_weight')
                delattr(block.attn.qkv, 'native_qkv_library')
