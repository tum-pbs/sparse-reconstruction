from functools import partial
from typing import Optional

import torch

try:
    from mamba_ssm import Mamba
    from mamba_ssm.ops.triton.layer_norm import RMSNorm
except ImportError:
    pass

from timm.layers.drop import DropPath

from torch import Tensor, nn
from torch.nn import Dropout

from dataclasses import dataclass


@dataclass
class ContextEncoderConfig:

    enabled: bool = True

    use_checkpoint: bool = False

    n_layer: int = 12

    hidden_size: Optional[int] = None

    num_heads: int = 8

    mlp_ratio: int = 4

    skip_connection: bool = True

    grad_ratio: float = 1.0
    """Ratio of sequence to maintain gradients for better memory usage."""

    attention_method: str = "hyper"

    in_context_patches: int = -1
    """Number of in-context patches. Default -1 is infinity."""

    init_zero_proj: bool = True

class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        residual_in_fp32=False,
        reverse=False,
        transpose=False,
        split_head=False,
        drop_path_rate=0.0,
        drop_rate=0.0,
        use_mlp=False,
        downsample=False,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.split_head = split_head
        self.reverse = reverse
        self.transpose = transpose
        self.drop_path = DropPath(drop_prob=drop_path_rate)
        self.dropout = Dropout(p=drop_rate)


    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
        **kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if self.reverse:
            hidden_states = hidden_states.flip(1)

        hidden_states = self.norm(hidden_states)

        hidden_states = hidden_states + self.drop_path(
            self.mixer(hidden_states, inference_params=inference_params, **kwargs)
        )

        if self.reverse:
            hidden_states = hidden_states.flip(1)

        return hidden_states, None

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    layer_idx=None,
    device=None,
    dtype=None,
    reverse=None,
    is_2d=False,
    drop_rate=0.1,
    drop_path_rate=0.1,
    use_mlp=False,
    transpose=False,
    split_head=False,
    use_nd=False,
    downsample=False,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        residual_in_fp32=residual_in_fp32,
        reverse=reverse,
        transpose=transpose,
        drop_rate=drop_rate,
        use_mlp=use_mlp,
        drop_path_rate=drop_path_rate,
        split_head=split_head,
        downsample=downsample,
    )
    block.layer_idx = layer_idx
    return block


if __name__ == "__main__":
    ssm_cfg = {"d_state": 16}
    blk = create_block(
        d_model=768,
        ssm_cfg=ssm_cfg,
        residual_in_fp32=True,
        drop_rate=0.1,
        drop_path_rate=0.1,
        reverse=False,
        transpose=False,
        use_mlp=False,
        is_2d=False,
        rms_norm=False,
        split_head=False,
        use_nd=False,
        downsample=False,
    ).cuda()
    x = torch.rand(4, 322, 768).cuda()
    y, _ = blk(x)
    assert x.shape == y.shape

import torch
from torch import nn


class LLMAttention(nn.Module):
    def __init__(
        self,
        dim,
        inner_dim,
        num_heads,
        causal=False,
    ):
        super().__init__()
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.proj = nn.Linear(inner_dim, dim)
        assert inner_dim % num_heads == 0, (inner_dim, num_heads)
        self.num_heads = num_heads

        from .hyper_attn.hyper_attn import HyperAttention

        self.attn = HyperAttention(
            input_dim=inner_dim // num_heads,
            lsh_num_projs=7,
            block_size=256,
            sample_size=256,
            min_seq_len=4096,
        )
        self.causal = causal


    def forward(self, x):
        """
        X: N L H
        """
        B, L, D = x.shape
        q, k, v = (
            self.qkv(x).reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )  # B H L D // num_heads
        attn_out = self.attn(q, k, v, causal=self.causal).permute(
            0, 2, 1, 3
        )  # B H L D // num_heads
        attn_out = attn_out.reshape(B, L, -1).contiguous()
        attn_out = self.proj(attn_out)

        return attn_out


class ViTAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_mask=False,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_mask = use_mask
        if use_mask:
            self.att_mask = nn.Parameter(torch.Tensor(self.num_heads, 196, 196))

    def forward(self, x):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.use_mask:
            attn = attn * torch.sigmoid(self.att_mask).expand(B, -1, -1, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x