from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat, reduce
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_, orthogonal_
from einops.layers.torch import Rearrange

class PeriodicConv2d(nn.Module):
    """Wrapper for Conv2d with periodic padding"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 pad=1,
                 bias=False):
        super().__init__()
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        if not isinstance(stride, tuple):
            self.stride = (stride, stride)
        self.filters = nn.Parameter(torch.randn(out_channels, in_channels,
                                           kernel_size[0], kernel_size[1]))
        self.pad = pad
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels,))
        else:
            self.bias = None

    def forward(self, x):
        x = F.pad(x, pad=(self.pad, self.pad, self.pad, self.pad), mode='circular')
        if self.bias is not None:
            x = F.conv2d(x, weight=self.filters, bias=self.bias, stride=self.stride)
        else:
            x = F.conv2d(x, weight=self.filters, stride=self.stride)
        return x


class PeriodicConv3d(nn.Module):
    """Wrapper for Conv3d with periodic padding, the periodic padding only happens in the temporal dimension"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 spatial_pad=1,
                 temp_pad=1,
                 pad_mode='constant',      # this pad mode is for temporal padding
                 bias=False):
        super().__init__()
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        assert len(kernel_size) == 3
        if not isinstance(stride, tuple):
            self.stride = (stride, stride, stride)
        else:
            self.stride = stride
        assert len(stride) == 3
        self.filters = nn.Parameter(torch.randn(out_channels, in_channels,
                                                kernel_size[0], kernel_size[1], kernel_size[2]))
        self.spatial_pad = spatial_pad
        self.temp_pad = temp_pad
        self.pad_mode = pad_mode
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels,))
        else:
            self.bias = None

    def forward(self, x):
        # only pad spatial dimension with PBC
        x = F.pad(x, pad=(self.spatial_pad, self.spatial_pad, self.spatial_pad, self.spatial_pad, 0, 0), mode='circular')
        # now pad time dimension
        x = F.pad(x, pad=(0, 0, 0, 0, self.temp_pad, self.temp_pad), mode=self.pad_mode)

        if self.bias is not None:
            x = F.conv3d(x, weight=self.filters, bias=self.bias, stride=self.stride)
        else:
            x = F.conv3d(x, weight=self.filters, bias=None, stride=self.stride)
        return x


def UpBlock(in_planes, out_planes):
    """Simple upsampling block"""
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False, padding_mode='circular'),
        nn.InstanceNorm2d(out_planes * 2),
        nn.GLU(dim=1),
        nn.Conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False, padding_mode='circular'),
        nn.InstanceNorm2d(out_planes * 2),
        nn.GLU(dim=1),
    )

    return block

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))


class GeGELU(nn.Module):
    """https: // paperswithcode.com / method / geglu"""

    def __init__(self):
        super().__init__()
        self.fn = nn.GELU()

    def forward(self, x):
        c = x.shape[-1]  # channel last arrangement
        return self.fn(x[..., :int(c // 2)]) * x[..., int(c // 2):]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            GeGELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ReLUFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# New position encoding module
# modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, min_freq=1 / 64, scale=1.):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        # coordinates [b, n]
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)  # [b, n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
    # split t into first half and second half
    # t: [b, h, n, d]
    # freq_x/y: [b, n, d]
    d = t.shape[-1]
    t_x, t_y = t[..., :d // 2], t[..., d // 2:]

    return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
                      apply_rotary_pos_emb(t_y, freqs_y)), dim=-1)


class StandardAttention(nn.Module):
    """Standard scaled dot product attention"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., causal=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.causal = causal  # simple autogressive attention with upper triangular part being masked zero

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            if self.causal:
                raise Exception('Passing in mask while attention is not causal')
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)  # similarity score

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class LinearAttention(nn.Module):
    """
    Contains following two types of attention, as discussed in "Choose a Transformer: Fourier or Galerkin"

    Galerkin type attention, with instance normalization on Key and Value
    Fourier type attention, with instance normalization on Query and Key
    """

    def __init__(self,
                 dim,
                 attn_type,  # ['fourier', 'galerkin']
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 init_params=True,
                 relative_emb=False,
                 scale=1.,
                 init_method='orthogonal',  # ['xavier', 'orthogonal']
                 init_gain=None,
                 relative_emb_dim=2,
                 min_freq=1 / 64,  # 1/64 is for 64 x 64 ns2d,
                 cat_pos=False,
                 pos_dim=2,
                 ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.attn_type = attn_type

        self.heads = heads
        self.dim_head = dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if attn_type == 'galerkin':
            self.k_norm = nn.InstanceNorm1d(dim_head)
            self.v_norm = nn.InstanceNorm1d(dim_head)
        elif attn_type == 'fourier':
            self.q_norm = nn.InstanceNorm1d(dim_head)
            self.k_norm = nn.InstanceNorm1d(dim_head)
        else:
            raise Exception(f'Unknown attention type {attn_type}')

        if not cat_pos:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim + pos_dim * heads, dim),
                nn.Dropout(dropout)
            )

        if init_gain is None:
            self.init_gain = 1. / dim_head
            self.diagonal_weight = 1. / dim_head
        else:
            self.init_gain = init_gain
            self.diagonal_weight = init_gain

        self.init_method = init_method
        if init_params:
            self._init_params()

        self.cat_pos = cat_pos
        self.pos_dim = pos_dim

        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        if relative_emb:
            assert not cat_pos
            self.emb_module = RotaryEmbedding(dim_head // self.relative_emb_dim, min_freq=min_freq, scale=scale)

    def _init_params(self):
        if self.init_method == 'xavier':
            init_fn = xavier_uniform_
        elif self.init_method == 'orthogonal':
            init_fn = orthogonal_
        else:
            raise Exception('Unknown initialization')

        for param in self.to_qkv.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    if self.attn_type == 'fourier':
                        # for v
                        init_fn(param[(self.heads * 2 + h) * self.dim_head:(self.heads * 2 + h + 1) * self.dim_head, :],
                                gain=self.init_gain)
                        param.data[(self.heads * 2 + h) * self.dim_head:(self.heads * 2 + h + 1) * self.dim_head,
                        :] += self.diagonal_weight * \
                              torch.diag(torch.ones(
                                  param.size(-1),
                                  dtype=torch.float32))
                    else:  # for galerkin
                        # for q
                        init_fn(param[h * self.dim_head:(h + 1) * self.dim_head, :], gain=self.init_gain)
                        #
                        param.data[h * self.dim_head:(h + 1) * self.dim_head, :] += self.diagonal_weight * \
                                                                                    torch.diag(torch.ones(
                                                                                        param.size(-1),
                                                                                        dtype=torch.float32))

    def norm_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, 'b h n d -> (b h) n d')),
            '(b h) n d -> b h n d', b=b)

    def forward(self, x, pos=None, not_assoc=False):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        if pos is None and self.relative_emb:
            raise Exception('Must pass in coordinates when under relative position embedding mode')

        if self.attn_type == 'galerkin':
            k = self.norm_wrt_domain(k, self.k_norm)
            v = self.norm_wrt_domain(v, self.v_norm)
        else:  # fourier
            q = self.norm_wrt_domain(q, self.q_norm)
            k = self.norm_wrt_domain(k, self.k_norm)

        if self.relative_emb:
            if self.relative_emb_dim == 2:
                freqs_x = self.emb_module.forward(pos[..., 0], x.device)
                freqs_y = self.emb_module.forward(pos[..., 1], x.device)
                freqs_x = repeat(freqs_x, 'b n d -> b h n d', h=q.shape[1])
                freqs_y = repeat(freqs_y, 'b n d -> b h n d', h=q.shape[1])

                q = apply_2d_rotary_pos_emb(q, freqs_x, freqs_y)
                k = apply_2d_rotary_pos_emb(k, freqs_x, freqs_y)
            elif self.relative_emb_dim == 1:
                assert pos.shape[-1] == 1
                freqs = self.emb_module.forward(pos[..., 0], x.device)
                freqs = repeat(freqs, 'b n d -> b h n d', h=q.shape[1])
                q = apply_rotary_pos_emb(q, freqs)
                k = apply_rotary_pos_emb(k, freqs)
            else:
                raise Exception('Currently doesnt support relative embedding > 2 dimensions')
        elif self.cat_pos:
            assert pos.size(-1) == self.pos_dim
            pos = pos.unsqueeze(1)
            pos = pos.repeat([1, self.heads, 1, 1])
            q, k, v = [torch.cat([pos, x], dim=-1) for x in (q, k, v)]

        if not_assoc:
            # this is more efficient when n<<c
            score = torch.matmul(q, k.transpose(-1, -2))
            out = torch.matmul(score, v) * (1. / q.shape[2])
        else:
            dots = torch.matmul(k.transpose(-1, -2), v)
            out = torch.matmul(q, dots) * (1. / q.shape[2])
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossLinearAttention(nn.Module):
    def __init__(self,
                 dim,
                 attn_type,  # ['fourier', 'galerkin']
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 init_params=True,
                 relative_emb=False,
                 scale=1.,
                 init_method='orthogonal',  # ['xavier', 'orthogonal']
                 init_gain=None,
                 relative_emb_dim=2,
                 min_freq=1 / 64,  # 1/64 is for 64 x 64 ns2d,
                 cat_pos=False,
                 pos_dim=2,
                 ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.attn_type = attn_type

        self.heads = heads
        self.dim_head = dim_head

        # query is the classification token
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        if attn_type == 'galerkin':
            self.k_norm = nn.InstanceNorm1d(dim_head)
            self.v_norm = nn.InstanceNorm1d(dim_head)
        elif attn_type == 'fourier':
            self.q_norm = nn.InstanceNorm1d(dim_head)
            self.k_norm = nn.InstanceNorm1d(dim_head)
        else:
            raise Exception(f'Unknown attention type {attn_type}')

        if not cat_pos:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim + pos_dim * heads, dim),
                nn.Dropout(dropout)
            )

        if init_gain is None:
            self.init_gain = 1. / dim_head
            self.diagonal_weight = 1. / dim_head
        else:
            self.init_gain = init_gain
            self.diagonal_weight = init_gain
        self.init_method = init_method
        if init_params:
            self._init_params()

        self.cat_pos = cat_pos
        self.pos_dim = pos_dim

        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        if relative_emb:
            self.emb_module = RotaryEmbedding(dim_head // self.relative_emb_dim, min_freq=min_freq, scale=scale)

    def _init_params(self):
        if self.init_method == 'xavier':
            init_fn = xavier_uniform_
        elif self.init_method == 'orthogonal':
            init_fn = orthogonal_
        else:
            raise Exception('Unknown initialization')

        for param in self.to_kv.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    # for k
                    init_fn(param[h * self.dim_head:(h + 1) * self.dim_head, :], gain=self.init_gain)
                    param.data[h * self.dim_head:(h + 1) * self.dim_head, :] += self.diagonal_weight * \
                                                                                torch.diag(torch.ones(
                                                                                    param.size(-1),
                                                                                    dtype=torch.float32))

                    # for v
                    init_fn(param[(self.heads + h) * self.dim_head:(self.heads + h + 1) * self.dim_head, :],
                            gain=self.init_gain)
                    param.data[(self.heads + h) * self.dim_head:(self.heads + h + 1) * self.dim_head,
                    :] += self.diagonal_weight * \
                          torch.diag(torch.ones(
                              param.size(-1), dtype=torch.float32))

        for param in self.to_q.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    # for q
                    init_fn(param[h * self.dim_head:(h + 1) * self.dim_head, :], gain=self.init_gain)
                    param.data[h * self.dim_head:(h + 1) * self.dim_head, :] += self.diagonal_weight * \
                                                                                torch.diag(torch.ones(
                                                                                    param.size(-1),
                                                                                    dtype=torch.float32))

    def norm_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, 'b h n d -> (b h) n d')),
            '(b h) n d -> b h n d', b=b)

    def forward(self, x, z, x_pos=None, z_pos=None):
        # x (z^T z)
        # x [b, n1, d]
        # z [b, n2, d]
        n1 = x.shape[1]  # x [b, n1, d]
        n2 = z.shape[1]  # z [b, n2, d]

        q = self.to_q(x)

        kv = self.to_kv(z).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)

        if (x_pos is None or z_pos is None) and self.relative_emb:
            raise Exception('Must pass in coordinates when under relative position embedding mode')
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        if self.attn_type == 'galerkin':
            k = self.norm_wrt_domain(k, self.k_norm)
            v = self.norm_wrt_domain(v, self.v_norm)
        else:  # fourier
            q = self.norm_wrt_domain(q, self.q_norm)
            k = self.norm_wrt_domain(k, self.k_norm)

        if self.relative_emb:
            if self.relative_emb_dim == 2:

                x_freqs_x = self.emb_module.forward(x_pos[..., 0], x.device)
                x_freqs_y = self.emb_module.forward(x_pos[..., 1], x.device)
                x_freqs_x = repeat(x_freqs_x, 'b n d -> b h n d', h=q.shape[1])
                x_freqs_y = repeat(x_freqs_y, 'b n d -> b h n d', h=q.shape[1])

                z_freqs_x = self.emb_module.forward(z_pos[..., 0], z.device)
                z_freqs_y = self.emb_module.forward(z_pos[..., 1], z.device)
                z_freqs_x = repeat(z_freqs_x, 'b n d -> b h n d', h=q.shape[1])
                z_freqs_y = repeat(z_freqs_y, 'b n d -> b h n d', h=q.shape[1])

                q = apply_2d_rotary_pos_emb(q, x_freqs_x, x_freqs_y)
                k = apply_2d_rotary_pos_emb(k, z_freqs_x, z_freqs_y)

            elif self.relative_emb_dim == 1:
                assert x_pos.shape[-1] == 1 and z_pos.shape[-1] == 1
                x_freqs = self.emb_module.forward(x_pos[..., 0], x.device)
                x_freqs = repeat(x_freqs, 'b n d -> b h n d', h=q.shape[1])

                z_freqs = self.emb_module.forward(z_pos[..., 0], x.device)
                z_freqs = repeat(z_freqs, 'b n d -> b h n d', h=q.shape[1])

                q = apply_rotary_pos_emb(q, x_freqs)  # query from x domain
                k = apply_rotary_pos_emb(k, z_freqs)  # key from z domain
            else:
                raise Exception('Currently doesnt support relative embedding > 2 dimensions')
        elif self.cat_pos:
            assert x_pos.size(-1) == self.pos_dim and z_pos.size(-1) == self.pos_dim
            x_pos = x_pos.unsqueeze(1)
            x_pos = x_pos.repeat([1, self.heads, 1, 1])
            q = torch.cat([x_pos, q], dim=-1)

            z_pos = z_pos.unsqueeze(1)
            z_pos = z_pos.repeat([1, self.heads, 1, 1])
            k = torch.cat([z_pos, k], dim=-1)
            v = torch.cat([z_pos, v], dim=-1)

        dots = torch.matmul(k.transpose(-1, -2), v)

        out = torch.matmul(q, dots) * (1. / n2)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


# helpers

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


# helper
def knn(x1, x2, k):
    # x1 [b, n1, c], x2 [b, n2, c]
    inner = -2 * torch.matmul(x1, rearrange(x2, 'b n c -> b c n'))  # [b n1 n2]
    xx = torch.sum(x1 ** 2, dim=-1, keepdim=True)  # [b, n1, 1]
    yy = torch.sum(x2 ** 2, dim=-1, keepdim=True)  # [b, n2, 1]
    pairwise_distance = -xx - inner - yy.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, n1, k)
    return idx

class ProjDotProduct(nn.Module):
    """
    Dot product that emulates the Branch and Trunk in DeepONet,
    implementation based on:
    https://github.com/devzhk/PINO/blob/97654eba0e3244322079d85d39fe673ceceade11/baselines/model.py#L22
    """

    def __init__(self,
                 branch_dim,
                 trunk_dim,
                 inner_dim,
                 init_params=True,
                 init_method='orthogonal',  # ['xavier', 'orthogonal']
                 init_gain=None,
                 ):
        super().__init__()

        self.branch_proj = nn.Linear(branch_dim, inner_dim, bias=False)
        self.trunk_proj = nn.Linear(trunk_dim, inner_dim, bias=False)

        self.to_out = nn.Identity()

        if init_gain is None:
            self.init_gain = 1. / inner_dim
            self.diagonal_weight = 1. / inner_dim
        else:
            self.init_gain = init_gain
            self.diagonal_weight = init_gain

        self.init_method = init_method
        if init_params:
            self._init_params()

    def _init_params(self):
        if self.init_method == 'xavier':
            init_fn = xavier_uniform_
        elif self.init_method == 'orthogonal':
            init_fn = orthogonal_
        else:
            raise Exception('Unknown initialization')

        for param in self.branch_proj.parameters():
            if param.ndim > 1:
                # for q
                init_fn(param, gain=self.init_gain)

        for param in self.trunk_proj.parameters():
            if param.ndim > 1:
                # for k
                init_fn(param, gain=self.init_gain)

    def forward(self, x, z):
        # x [n1, d]
        # z [b, d]

        q = self.trunk_proj(x)
        k = self.branch_proj(z)

        out = torch.einsum('bi,ni->bn', k, q)

        return self.to_out(out)

class STTransformerCatNoCls(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 attn_type,  # ['standard', 'galerkin', 'fourier']
                 use_ln=False,
                 scale=32,  # can be list, or an int
                 dropout=0.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 attention_init='orthogonal'):
        super().__init__()
        assert attn_type in ['standard', 'galerkin', 'fourier']

        if isinstance(scale, int):
            scale = [scale] * depth

        self.layers = nn.ModuleList([])
        self.attn_type = attn_type
        self.use_ln = use_ln

        if attn_type == 'standard':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    # spatial
                    nn.ModuleList([
                    PreNorm(dim, StandardAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]),
                    # temporal
                    nn.ModuleList([
                    PreNorm(dim, StandardAttention(dim+1, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]),
                ]))
        else:

            for d in range(depth):
                # spatial
                attn_module1 = LinearAttention(dim, attn_type,
                                               heads=heads, dim_head=dim_head, dropout=dropout,
                                               relative_emb=True, scale=scale[d],
                                               relative_emb_dim=relative_emb_dim,
                                               min_freq=min_freq,
                                               init_method=attention_init
                                                )
                # temporal
                attn_module2 = LinearAttention(dim, attn_type,
                                               heads=heads, dim_head=dim_head, dropout=dropout,
                                               relative_emb=True, scale=1,
                                               relative_emb_dim=1,
                                               min_freq=1,
                                               init_method=attention_init)
                if not use_ln:

                    self.layers.append(nn.ModuleList([
                        # spatial
                        nn.ModuleList([
                        attn_module1,
                        FeedForward(dim, mlp_dim, dropout=dropout)]),

                        # temporal
                        nn.ModuleList([
                        attn_module2,
                        FeedForward(dim, mlp_dim, dropout=dropout)]),
                    ]))
                else:
                    self.layers.append(nn.ModuleList([
                        # spatial
                        nn.ModuleList([
                            nn.LayerNorm(dim),
                            attn_module1,
                            FeedForward(dim, mlp_dim, dropout=dropout),
                        ]),

                        # temporal
                        nn.ModuleList([
                            nn.LayerNorm(dim),
                            attn_module2,
                            FeedForward(dim, mlp_dim, dropout=dropout),
                        ]),
                    ]))

    def forward(self, x, pos_embedding):
        # x in [b t n c]
        b, t, n, c = x.shape
        pos_embedding = repeat(pos_embedding, 'b n c -> (b repeat) n c', repeat=t)  # [b*t, n, c]
        temp_embedding = torch.arange(t).float().to(x.device).view(1, t, 1)
        temp_embedding = repeat(temp_embedding, '() t c -> b t c', b=b*n)

        for layer_no, (spa_attn, temp_attn) in enumerate(self.layers):
            if layer_no == 0:
                x = rearrange(x, 'b t n c -> (b t) n c')
            else:
                x = rearrange(x, '(b n) t c -> (b t) n c', n=n)

            # spatial attention
            if not self.use_ln:
                [attn, ffn] = spa_attn

                x = attn(x, pos_embedding) + x
                x = ffn(x) + x

            else:
                [ln, attn, ffn] = spa_attn
                x = ln(x)
                x = attn(x, pos_embedding) + x
                x = ffn(x) + x

            # temporal attention
            x = rearrange(x, '(b t) n c -> (b n) t c', t=t)

            if not self.use_ln:
                [attn, ffn] = temp_attn

                x = attn(x, temp_embedding) + x
                x = ffn(x) + x
            else:
                [ln, attn, ffn] = temp_attn
                x = ln(x)
                x = attn(x, temp_embedding, not_assoc=True) + x
                x = ffn(x) + x

            if layer_no == len(self.layers) - 1:
                x = rearrange(x, '(b n) t c -> b n t c', n=n)
        return x

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class TransformerCatNoCls(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 attn_type,  # ['standard', 'galerkin', 'fourier']
                 use_ln=False,
                 scale=16,     # can be list, or an int
                 dropout=0.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 attention_init='orthogonal',
                 init_gain=None,
                 use_relu=False,
                 cat_pos=False,
                 ):
        super().__init__()
        assert attn_type in ['standard', 'galerkin', 'fourier']

        if isinstance(scale, int):
            scale = [scale] * depth
        assert len(scale) == depth

        self.layers = nn.ModuleList([])
        self.attn_type = attn_type
        self.use_ln = use_ln

        if attn_type == 'standard':
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList([
                    PreNorm(dim, StandardAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim,  FeedForward(dim, mlp_dim, dropout=dropout)
                                  if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout))]),
                )
        else:
            for d in range(depth):
                if scale[d] != -1 or cat_pos:
                    attn_module = LinearAttention(dim, attn_type,
                                                   heads=heads, dim_head=dim_head, dropout=dropout,
                                                   relative_emb=True, scale=scale[d],
                                                   relative_emb_dim=relative_emb_dim,
                                                   min_freq=min_freq,
                                                   init_method=attention_init,
                                                   init_gain=init_gain
                                                   )
                else:
                    attn_module = LinearAttention(dim, attn_type,
                                                  heads=heads, dim_head=dim_head, dropout=dropout,
                                                  cat_pos=True,
                                                  pos_dim=relative_emb_dim,
                                                  relative_emb=False,
                                                  init_method=attention_init,
                                                  init_gain=init_gain
                                                  )
                if not use_ln:
                    self.layers.append(
                        nn.ModuleList([
                                        attn_module,
                                        FeedForward(dim, mlp_dim, dropout=dropout)
                                        if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout)
                        ]),
                        )
                else:
                    self.layers.append(
                        nn.ModuleList([
                            nn.LayerNorm(dim),
                            attn_module,
                            nn.LayerNorm(dim),
                            FeedForward(dim, mlp_dim, dropout=dropout)
                            if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout),
                        ]),
                    )

    def forward(self, x, pos_embedding):
        # x in [b n c], pos_embedding in [b n 2]
        b, n, c = x.shape

        for layer_no, attn_layer in enumerate(self.layers):
            if not self.use_ln:
                [attn, ffn] = attn_layer

                x = attn(x, pos_embedding) + x
                x = ffn(x) + x
            else:
                [ln1, attn, ln2, ffn] = attn_layer
                x = ln1(x)
                x = attn(x, pos_embedding) + x
                x = ln2(x)
                x = ffn(x) + x
        return x

class CNNEncoder(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 in_grid_size,             # this should be the input image height/width
                 seq_len,                  # this should be the input sequence length
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 emb_dropout=0.1,           # dropout of embedding
                 ):
        super().__init__()
        h, w = pair(in_grid_size // 4)
        t = seq_len
        self.in_grid_size = in_grid_size

        self.to_embedding = nn.Sequential(
            PeriodicConv3d(input_channels, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                           spatial_pad=1, temp_pad=1, bias=False),
            nn.GELU(),
            PeriodicConv3d(64, in_emb_dim, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                           spatial_pad=1, temp_pad=0, bias=False),    # [t, h/2, w/2]
            nn.GELU(),
            PeriodicConv3d(in_emb_dim, in_emb_dim, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                           spatial_pad=1, temp_pad=0, bias=False),   # [t, h/4, w/4]
        )
        self.dropout = nn.Dropout(emb_dropout)

        self.net = nn.ModuleList([nn.Sequential(
                                PeriodicConv3d(in_emb_dim, in_emb_dim//4, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                                               spatial_pad=0, temp_pad=0, bias=False),
                                nn.GELU(),
                                PeriodicConv3d(in_emb_dim//4, in_emb_dim//4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                               spatial_pad=1, temp_pad=1, bias=False),
                                nn.GELU(),
                                PeriodicConv3d(in_emb_dim//4, in_emb_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                                               spatial_pad=0, temp_pad=0, bias=False),)
                                for _ in range(depth) ])

        # squeeze the temporal dimension
        self.to_init1 = nn.Sequential(
            nn.Conv3d(in_emb_dim, in_emb_dim,
                      kernel_size=(t, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False),
            nn.GELU()
        )

        self.to_init2 = nn.Sequential(
            nn.Conv2d(in_emb_dim, in_emb_dim, kernel_size=1, stride=1, padding=0, bias=False))

        # upsample the space resolution and go back to the original resolution
        self.up_block_num = 2
        self.up_layers = []
        for _ in range(self.up_block_num):
            self.up_layers.append(UpBlock(in_emb_dim, in_emb_dim))
        self.up_layers = nn.Sequential(*self.up_layers)

        self.to_out = nn.Sequential(
            nn.Conv2d(in_emb_dim, in_emb_dim, 1, 1, 0, bias=False),
            nn.GELU(),
            nn.Conv2d(in_emb_dim, in_emb_dim, 1, 1, 0, bias=False),
            nn.GELU(),
            nn.Conv2d(in_emb_dim, out_seq_emb_dim, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(out_seq_emb_dim))

        self.to_cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(in_emb_dim, in_emb_dim, 4, 4, 0, bias=False),
            nn.GELU(),
            nn.Conv2d(in_emb_dim, out_seq_emb_dim, 1, 1, 0, bias=False),
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(out_seq_emb_dim))

    def forward(self, x):
        x = self.to_embedding(x)
        x = self.dropout(x)
        for layer in self.net:
            x = layer(x) + x
        x = rearrange(self.to_init1(x), 'b c 1 h w -> b c h w')   # [b, c, h, w]
        x = self.to_init2(x)
        x_cls = self.to_cls(x).view(x.shape[0], -1)
        x = self.up_layers(x)
        x = self.to_out(x)
        return x, x_cls

class SpatialTemporalEncoder2D(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 heads,
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 ):
        super().__init__()

        self.to_embedding = nn.Sequential(
            # Rearrange('b c n -> b n c'),
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        if depth > 4:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', True, scale=[32, 16, 8, 8] +
                                                                             [1] * (depth - 4),
                                                     attention_init='orthogonal')
        else:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', True, scale=[32] + [16]*(depth-2) + [1],
                                                     attention_init='orthogonal')

        self.project_to_latent = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, t(*c)+2, n]
                input_pos,  # [b, n, 2]
                ):

        x = self.to_embedding(x)
        x = self.s_transformer.forward(x, input_pos)
        x = self.project_to_latent(x)

        return x


class SpatialEncoder2D(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 heads,
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 res,
                 use_ln=True,
                 emb_dropout=0.05,           # dropout of embedding
                 ):
        super().__init__()

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                 'galerkin',
                                                 use_relu=False,
                                                 use_ln=use_ln,
                                                 scale=[res, res//4] + [1]*(depth-2),
                                                 relative_emb_dim=2,
                                                 min_freq=1 / res,
                                                 dropout=0.03,
                                                 attention_init='orthogonal')

        self.to_out = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, n, c]
                input_pos,  # [b, n, 2]
                ):

        x = self.to_embedding(x)
        x = self.dropout(x)

        x = self.s_transformer.forward(x, input_pos)
        x = self.to_out(x)

        return x


class Encoder1D(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 emb_dropout=0.05,           # dropout of embedding
                 res=2048,
                 ):
        super().__init__()

        # self.dropout = nn.Dropout(emb_dropout)

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim-1, bias=False),
        )

        self.transformer = TransformerCatNoCls(in_emb_dim, depth, 1, in_emb_dim, in_emb_dim, 'fourier',
                                               scale=[8.] + [4.]*2 + [1.]*(depth-3),
                                               relative_emb_dim=1,
                                               min_freq=1/res,
                                               attention_init='orthogonal')

        self.project_to_latent = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, n, c]
                input_pos,  # [b, n, 1]
                ):
        x = self.to_embedding(x)
        # x = self.dropout(x)
        x = torch.cat((x, input_pos), dim=-1)
        x = self.transformer.forward(x, input_pos)
        x = self.project_to_latent(x)

        return x


# for ablation
class NoRelSpatialTemporalEncoder2D(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 heads,
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 ):
        super().__init__()

        self.to_embedding = nn.Sequential(
            # Rearrange('b c n -> b n c'),
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        if depth > 4:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', True, scale=-1,
                                                     attention_init='orthogonal')
        else:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', True, scale=-1,
                                                     attention_init='orthogonal')

        self.project_to_latent = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, t(*c)+2, n]
                input_pos,  # [b, n, 2]
                ):

        x = self.to_embedding(x)
        x = self.s_transformer.forward(x, input_pos)
        x = self.project_to_latent(x)

        return x

class AttentionPropagator2D(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 attn_type,         # ['none', 'galerkin', 'fourier']
                 mlp_dim,
                 scale,
                 use_ln=True,
                 dropout=0.):
        super().__init__()
        assert attn_type in ['none', 'galerkin', 'fourier']
        self.layers = nn.ModuleList([])

        self.attn_type = attn_type
        self.use_ln = use_ln
        for d in range(depth):
            attn_module = LinearAttention(dim, attn_type,
                                          heads=heads, dim_head=dim_head, dropout=dropout,
                                          relative_emb=True, scale=scale,
                                          relative_emb_dim=2,
                                          min_freq=1/64,
                                          init_method='orthogonal'
                                          )
            if use_ln:
                self.layers.append(
                    nn.ModuleList([
                        nn.LayerNorm(dim),
                        attn_module,
                        nn.LayerNorm(dim),
                        nn.Linear(dim+2, dim),
                        FeedForward(dim, mlp_dim, dropout=dropout)
                    ]),
                )
            else:
                self.layers.append(
                    nn.ModuleList([
                        attn_module,
                        nn.Linear(dim + 2, dim),
                        FeedForward(dim, mlp_dim, dropout=dropout)
                    ]),
                )

    def forward(self, x, pos):
        for layer_no, attn_layer in enumerate(self.layers):
            if self.use_ln:
                [ln1, attn, ln2, proj, ffn] = attn_layer
                x = attn(ln1(x), pos) + x
                x = ffn(
                    proj(torch.cat((ln2(x), pos), dim=-1))
                        ) + x
            else:
                [attn, proj, ffn] = attn_layer
                x = attn(x, pos) + x
                x = ffn(
                    proj(torch.cat((x, pos), dim=-1))
                        ) + x
        return x


class AttentionPropagator1D(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 attn_type,         # ['none', 'galerkin', 'fourier']
                 mlp_dim,
                 scale,
                 res,
                 dropout=0.):
        super().__init__()
        assert attn_type in ['none', 'galerkin', 'fourier']
        self.layers = nn.ModuleList([])

        self.attn_type = attn_type

        for d in range(depth):
            attn_module = LinearAttention(dim, attn_type,
                                          heads=heads, dim_head=dim_head, dropout=dropout,
                                          relative_emb=True,
                                          scale=scale,
                                          relative_emb_dim=1,
                                          min_freq=1 / res,
                                          )
            self.layers.append(
                nn.ModuleList([
                    attn_module,
                    FeedForward(dim, mlp_dim, dropout=dropout)
                ]),
            )

    def forward(self, x, pos):
        for layer_no, attn_layer in enumerate(self.layers):
            [attn, ffn] = attn_layer

            x = attn(x, pos) + x
            x = ffn(x) + x
        return x

class MLPPropagator(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.latent_channels = dim

        for d in range(depth):
            layer = nn.Sequential(
                nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
                nn.InstanceNorm2d(dim),
            )
            self.layers.append(layer)

    def forward(self, z):
        for layer, ffn in enumerate(self.layers):
            z = ffn(z) + z
        return z


class PointWiseMLPPropagator(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.latent_channels = dim

        for d in range(depth):
            if d == 0:
                layer = nn.Sequential(
                    nn.InstanceNorm1d(dim + 2),
                    nn.Linear(dim + 2, dim, bias=False),  # for position
                    nn.GELU(),
                    nn.Linear(dim, dim, bias=False),
                    nn.GELU(),
                    nn.Linear(dim, dim, bias=False),
                )
            else:
                layer = nn.Sequential(
                    nn.InstanceNorm1d(dim),
                    nn.Linear(dim, dim, bias=False),
                    nn.GELU(),
                    nn.Linear(dim, dim, bias=False),
                    nn.GELU(),
                    nn.Linear(dim, dim, bias=False),
                )
            self.layers.append(layer)

    def forward(self, z, pos):
        for layer, ffn in enumerate(self.layers):
            if layer == 0:
                z = ffn(torch.cat((z, pos), dim=-1)) + z
            else:
                z = ffn(z) + z
        return z


# code copied from: https://github.com/ndahlquist/pytorch-fourier-feature-networks
# author: Nic Dahlquist
class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, n, num_input_channels],
     returns a tensor of size [batches, n, mapping_size*2].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):

        batches, num_of_points, channels = x.shape

        # Make shape compatible for matmul with _B.
        # From [B, N, C] to [(B*N), C].
        x = rearrange(x, 'b n c -> (b n) c')

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = rearrange(x, '(b n) c -> b n c', b=batches)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class CrossFormer(nn.Module):
    def __init__(self,
                 dim,
                 attn_type,
                 heads,
                 dim_head,
                 mlp_dim,
                 residual=True,
                 use_ffn=True,
                 use_ln=False,
                 relative_emb=False,
                 scale=1.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 dropout=0.,
                 cat_pos=False,
                 ):
        super().__init__()

        self.cross_attn_module = CrossLinearAttention(dim, attn_type,
                                                       heads=heads, dim_head=dim_head, dropout=dropout,
                                                       relative_emb=relative_emb,
                                                       scale=scale,

                                                       relative_emb_dim=relative_emb_dim,
                                                       min_freq=min_freq,
                                                       init_method='orthogonal',
                                                       cat_pos=cat_pos,
                                                       pos_dim=relative_emb_dim,
                                                  )
        self.use_ln = use_ln
        self.residual = residual
        self.use_ffn = use_ffn

        if self.use_ln:
            self.ln1 = nn.LayerNorm(dim)
            self.ln2 = nn.LayerNorm(dim)

        if self.use_ffn:
            self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, z, x_pos=None, z_pos=None):
        # x in [b n1 c]
        # b, n1, c = x.shape   # coordinate encoding
        # b, n2, c = z.shape   # system encoding
        if self.use_ln:
            z = self.ln1(z)
            if self.residual:
                x = self.ln2(self.cross_attn_module(x, z, x_pos, z_pos)) + x
            else:
                x = self.ln2(self.cross_attn_module(x, z, x_pos, z_pos))
        else:
            if self.residual:
                x = self.cross_attn_module(x, z, x_pos, z_pos) + x
            else:
                x = self.cross_attn_module(x, z, x_pos, z_pos)

        if self.use_ffn:
            x = self.ffn(x) + x

        return x


class BranchTrunkNet(nn.Module):
    def __init__(self,
                 dim,
                 branch_size,
                 branchnet_dim,
                 ):
        super().__init__()
        self.proj = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear(branch_size, branchnet_dim),
            nn.ReLU(),
            nn.Linear(branchnet_dim//2, branchnet_dim//2),
            nn.ReLU(),
            nn.Linear(branchnet_dim//2, 1),

        )
        self.net = ProjDotProduct(dim, dim, dim)

    def forward(self, x, z):
        # x in [b n1 c]
        # b, n1, c = x.shape   # coordinate encoding
        # b, n2, c = z.shape   # system encoding
        z = self.proj(z).squeeze(-1)
        return self.net(x, z)


class Decoder(nn.Module):
    def __init__(self,
                 grid_size,                # 64 x 64
                 latent_channels,              # 256??
                 out_channels,                 # 1 or 2?
                 out_steps,                    # 10
                 decoding_depth,                        # 4?
                 propagator_depth,
                 pos_encoding_aug=False,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.out_steps = out_steps
        self.grid_size = grid_size
        self.latent_channels = latent_channels
        self.pos_encoding_aug = pos_encoding_aug

        self.propagator = MLPPropagator(self.latent_channels, propagator_depth)

        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.latent_channels + 2 if (l == 0 and pos_encoding_aug) else self.latent_channels,
                          self.latent_channels, 1, 1, 0, bias=False),
                nn.GELU(),
                nn.Conv2d(self.latent_channels, self.latent_channels, 1, 1, 0, bias=False),
                nn.GELU(),
                nn.Conv2d(self.latent_channels, self.latent_channels, 1, 1, 0, bias=False),
            )
            for l in range(decoding_depth)])

        self.to_out = nn.Conv2d(self.latent_channels, self.out_channels*self.out_steps, 1, 1, 0, bias=True)

        x0, y0 = np.meshgrid(np.linspace(0, 1, grid_size),
                             np.linspace(0, 1, grid_size))
        xs = np.concatenate((x0[None, ...], y0[None, ...]), axis=0)
        self.grid = nn.Parameter(torch.from_numpy(xs.reshape((1, 2, grid_size, grid_size))).float(), requires_grad=False)

    def decode(self, z):
        if self.pos_encoding_aug:
            z = torch.cat((z, repeat(self.grid, 'b c h w -> (repeat b) c h w', repeat=z.shape[0])), dim=1)
        for layer in self.decoder:
            z = layer(z)
        z = self.to_out(z)
        return z

    def forward(self, z, z_cls, forward_steps):
        assert len(z.shape) == 4  # [b, c, h, w]
        history = []
        z_cls = rearrange(z_cls, 'b c -> b c 1 1').repeat(1, 1, z.shape[2], z.shape[3])
        # forward the dynamics in the latent space
        for step in range(forward_steps):
            z = self.propagator(z + z_cls)
            u = self.decode(z)
            history.append(rearrange(u, 'b (c t) h w -> b c t h w', c=self.out_channels, t=self.out_steps))
        history = torch.cat(history, dim=2)
        return history    # [b, c, length_of_history, h, w]


class GraphDecoder(nn.Module):
    def __init__(self,
                 latent_channels,              # 256??
                 out_channels,                 # 1 or 2?
                 out_steps,                    # 10
                 decoding_depth,               # 4?
                 propagator_depth,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.out_steps = out_steps
        self.latent_channels = latent_channels

        # self.pivotal_to_query = SmoothConvDecoder(self.latent_channels, self.latent_channels, 3)

        self.propagator = PointWiseMLPPropagator(self.latent_channels, propagator_depth)

        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                nn.GELU(),
                nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                nn.GELU(),
                nn.Linear(self.latent_channels, self.latent_channels, bias=True),
            )
            for _ in range(decoding_depth)])

        self.to_out = nn.Linear(self.latent_channels, self.out_channels*self.out_steps, bias=True)

    def decode(self, z):
        for layer in self.decoder:
            z = layer(z)
        z = self.to_out(z)
        return z

    def forward(self,
                propagate_pos,      #  [sum n_p, 2]
                pivotal_pos,        #  [sum n_pivot, 2]
                pivotal2prop_graph,   #  [sum g_pivot, 2]
                pivotal2prop_cutoff,  # float
                z_pivotal,          # [b, c, num_of_pivot]
                z_cls,              # [b, c]
                forward_steps):
        assert len(z_pivotal.shape) == 3  # [b, n, c]
        batch_size = z_pivotal.shape[0]
        history = []
        num_of_prop = int(propagate_pos.shape[0] // batch_size)  # assuming each batch have same number of nodes
        z_cls = rearrange(z_cls, 'b c -> b 1 c').repeat(1, num_of_prop, 1)

        # get embedding for nodes we want to propagate dynamics
        # z in shape [b, n, c]
        # z = self.pivotal_to_query.forward(z_pivotal, pivotal_pos, propagate_pos, pivotal2prop_graph,
        #                                   pivotal2prop_cutoff)
        z = rearrange(z_pivotal, 'b c n -> b n c')
        pos = rearrange(propagate_pos, '(b n) c -> b n c', b=batch_size)
        # forward the dynamics in the latent space
        for step in range(forward_steps):
            z = self.propagator(z + z_cls, pos)
            u = self.decode(z)
            history.append(rearrange(u, 'b n (t c) -> b c t n', c=self.out_channels, t=self.out_steps))
        history = torch.cat(history, dim=2)  # concatenate in temporal dimension
        return history    # [b, c, length_of_history, n]


class DecoderNew(nn.Module):
    def __init__(self,
                 latent_channels,              # 256??
                 out_channels,                 # 1 or 2?
                 out_steps,                    # 10
                 decoding_depth,               # 4?
                 propagator_depth,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.out_steps = out_steps
        self.latent_channels = latent_channels

        self.propagator = PointWiseMLPPropagator(self.latent_channels, propagator_depth)

        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                nn.GELU(),
                nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                nn.GELU(),
                nn.Linear(self.latent_channels, self.latent_channels, bias=True),
            )
            for _ in range(decoding_depth)])

        self.to_out = nn.Linear(self.latent_channels, self.out_channels*self.out_steps, bias=True)

    def decode(self, z):
        for layer in self.decoder:
            z = layer(z)
        z = self.to_out(z)
        return z

    def forward(self,
                z,                  # [b, c, h, w]
                z_cls,              # [b, c]
                propagate_pos,      # [b, n, 2]
                forward_steps):
        history = []
        pos = propagate_pos
        z = rearrange(z, 'b c h w -> b (h w) c')
        z_cls = rearrange(z_cls, 'b c -> b 1 c').repeat(1, z.shape[1], 1)

        # forward the dynamics in the latent space
        for step in range(forward_steps):
            z = self.propagator(z + z_cls, pos)
            u = self.decode(z)
            history.append(rearrange(u, 'b n (t c) -> b c t n', c=self.out_channels, t=self.out_steps))
        history = torch.cat(history, dim=2)  # concatenate in temporal dimension
        return history    # [b, c, length_of_history, n]


class PointWiseDecoder(nn.Module):
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 out_steps,  # 10
                 decoding_depth,  # 4?
                 propagator_depth,
                 scale=8,
                 use_rope=False,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.out_steps = out_steps
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//4, scale=scale),
            nn.Linear(self.latent_channels // 2, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels // 2, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels // 2, self.latent_channels // 2, bias=False),
        )
        self.z_project = nn.Linear(self.latent_channels, self.latent_channels//2, bias=False)

        self.use_rope = use_rope
        if not use_rope:
            self.decoding_transformer = CrossFormer(self.latent_channels//2, 4, 64, self.latent_channels//2)
        else:
            self.decoding_transformer = CrossFormer(self.latent_channels//2, 4, 64, self.latent_channels//2,
                                                    relative_emb=True, scale=16.)

        self.project = nn.Sequential(
            nn.Linear(self.latent_channels//2, self.latent_channels, bias=False),
            nn.InstanceNorm1d(self.latent_channels))

        self.propagator = PointWiseMLPPropagator(self.latent_channels, propagator_depth)

        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.InstanceNorm1d(self.latent_channels),
                nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                nn.GELU(),
                nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                nn.GELU(),
                nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            )
            for _ in range(decoding_depth)])

        self.to_out = nn.Sequential(
            nn.Linear(self.latent_channels, self.out_channels * self.out_steps, bias=True))

    def decode(self, z):
        for layer in self.decoder:
            z = layer(z)
        z = self.to_out(z)
        return z

    def forward(self,
                z,  # [b, n c]
                z_cls,  # [b, c]
                propagate_pos,  # [b, n, 2]
                forward_steps,
                input_pos=None):
        history = []
        x = self.coordinate_projection.forward(propagate_pos)
        z_cls = z_cls.repeat(1, propagate_pos.shape[1], 1)
        z = self.z_project(z)  # c to c/2
        if not self.use_rope:
            z = self.decoding_transformer.forward(x, z)
        else:
            z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.project.forward(z)

        # forward the dynamics in the latent space
        for step in range(forward_steps):
            z = self.propagator(z + z_cls, propagate_pos)
            u = self.decode(z)
            history.append(rearrange(u, 'b n (t c) -> b c t n', c=self.out_channels, t=self.out_steps))
        history = torch.cat(history, dim=2)  # concatenate in temporal dimension
        return history  # [b, c, length_of_history, n]


class SimplerPointWiseDecoder(nn.Module):
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 out_steps,  # 10
                 decoding_depth,  # 4?
                 propagator_depth,
                 scale=8,
                 use_rope=False,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.out_steps = out_steps
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//4, scale=scale),
            nn.Linear(self.latent_channels // 2, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels // 2, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels // 2, self.latent_channels // 2, bias=False),
        )
        self.z_project = nn.Linear(self.latent_channels, self.latent_channels//2, bias=False)

        self.use_rope = use_rope
        if not use_rope:
            self.decoding_transformer = CrossFormer(self.latent_channels//2, 4, 64, self.latent_channels//2)
        else:
            self.decoding_transformer = CrossFormer(self.latent_channels//2, 4, 64, self.latent_channels//2,
                                                    relative_emb=True, scale=16.)

        self.project = nn.Sequential(
            nn.Linear(self.latent_channels//2, self.latent_channels, bias=False),
            nn.InstanceNorm1d(self.latent_channels))

        self.propagator = PointWiseMLPPropagator(self.latent_channels, propagator_depth)

        self.decoder = nn.Sequential(
                nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                nn.GELU(),
                nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                nn.GELU(),
                nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                nn.GELU()
            )

        self.to_out = nn.Linear(self.latent_channels, self.out_channels * self.out_steps, bias=True)

    def decode(self, z):
        z = self.decoder(z)
        z = self.to_out(z)
        return z

    def forward(self,
                z,  # [b, n c]
                z_cls,  # [b, c]
                propagate_pos,  # [b, n, 2]
                forward_steps,
                input_pos=None):
        history = []
        x = self.coordinate_projection.forward(propagate_pos)
        z_cls = z_cls.repeat(1, propagate_pos.shape[1], 1)
        z = self.z_project(z)  # c to c/2
        if not self.use_rope:
            z = self.decoding_transformer.forward(x, z)
        else:
            z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.project.forward(z)

        # forward the dynamics in the latent space
        for step in range(forward_steps):
            z = self.propagator(z + z_cls, propagate_pos)
            u = self.decode(z)
            history.append(rearrange(u, 'b n (t c) -> b c t n', c=self.out_channels, t=self.out_steps))
        history = torch.cat(history, dim=2)  # concatenate in temporal dimension
        return history  # [b, c, length_of_history, n]


class PointWiseDecoder2D(nn.Module):
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 out_steps,  # 10
                 propagator_depth,
                 scale=8,
                 dropout=0.,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.out_steps = out_steps
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels//2, 'galerkin', 4,
                                                self.latent_channels//2, self.latent_channels//2,
                                                relative_emb=True,
                                                scale=16.,
                                                relative_emb_dim=2,
                                                min_freq=1/64)

        self.expand_feat = nn.Linear(self.latent_channels//2, self.latent_channels)

        self.propagator = nn.ModuleList([
               nn.ModuleList([nn.LayerNorm(self.latent_channels),
               nn.Sequential(
                    nn.Linear(self.latent_channels + 2, self.latent_channels, bias=False),
                    nn.GELU(),
                    nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                    nn.GELU(),
                    nn.Linear(self.latent_channels, self.latent_channels, bias=False))])
            for _ in range(propagator_depth)])

        self.to_out = nn.Sequential(
            nn.LayerNorm(self.latent_channels),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels // 2, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels * self.out_steps, bias=True))

    def propagate(self, z, pos):
        for layer in self.propagator:
            norm_fn, ffn = layer
            z = ffn(torch.cat((norm_fn(z), pos), dim=-1)) + z
        return z

    def decode(self, z):
        z = self.to_out(z)
        return z

    def get_embedding(self,
                      z,  # [b, n c]
                      propagate_pos,  # [b, n, 2]
                      input_pos
                      ):
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.expand_feat(z)
        return z

    def forward(self,
                z,              # [b, n, c]
                propagate_pos   # [b, n, 2]
                ):
        z = self.propagate(z, propagate_pos)
        u = self.decode(z)
        u = rearrange(u, 'b n (t c) -> b (t c) n', c=self.out_channels, t=self.out_steps)
        return u, z                # [b c_out t n], [b c_latent t n]

    def rollout(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 2]
                forward_steps,
                input_pos):
        history = []
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.expand_feat(z)

        # forward the dynamics in the latent space
        for step in range(forward_steps//self.out_steps):
            z = self.propagate(z, propagate_pos)
            u = self.decode(z)
            history.append(rearrange(u, 'b n (t c) -> b (t c) n', c=self.out_channels, t=self.out_steps))
        history = torch.cat(history, dim=-2)  # concatenate in temporal dimension
        return history  # [b, length_of_history*c, n]


class PointWiseDecoder1D(nn.Module):
    # for Burgers equation
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 decoding_depth,  # 4?
                 scale=8,
                 res=2048,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(1, self.latent_channels, scale=scale),
            nn.GELU(),
            nn.Linear(self.latent_channels*2, self.latent_channels, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels, 'fourier', 8,
                                                self.latent_channels, self.latent_channels,
                                                relative_emb=True,
                                                scale=1,
                                                relative_emb_dim=1,
                                                min_freq=1/res)

        self.propagator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                nn.GELU(),
                nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                nn.GELU(),
                nn.Linear(self.latent_channels, self.latent_channels, bias=False),)
            for _ in range(decoding_depth)])

        self.init_propagator_params()
        self.to_out = nn.Sequential(
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))

    def propagate(self, z):
        for num_l, layer in enumerate(self.propagator):
            z = z + layer(z)
        return z

    def decode(self, z):
        z = self.to_out(z)
        return z

    def init_propagator_params(self):
        for block in self.propagator:
            for layers in block:
                    for param in layers.parameters():
                        if param.ndim > 1:
                            in_c = param.size(-1)
                            orthogonal_(param[:in_c], gain=1/in_c)
                            param.data[:in_c] += 1/in_c * torch.diag(torch.ones(param.size(-1), dtype=torch.float32))
                            if param.size(-2) != param.size(-1):
                                orthogonal_(param[in_c:], gain=1/in_c)
                                param.data[in_c:] += 1/in_c * torch.diag(torch.ones(param.size(-1), dtype=torch.float32))

    def forward(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 1]
                input_pos=None,
                ):

        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)

        z = self.propagate(z)
        z = self.decode(z)
        return z  # [b, n, c]


class PointWiseDecoder2DSimple(nn.Module):
    # for Darcy equation
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 res=211,
                 scale=0.5,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            # nn.Linear(2, self.latent_channels, bias=False),
            # nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            # nn.Dropout(0.05),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 4,
                                                self.latent_channels, self.latent_channels,
                                                use_ln=False,
                                                residual=True,
                                                relative_emb=True,
                                                scale=16,
                                                relative_emb_dim=2,
                                                min_freq=1/res)

        # self.init_propagator_params()
        self.to_out = nn.Sequential(
            nn.Linear(self.latent_channels+2, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))

    def decode(self, z):
        z = self.to_out(z)
        return z

    def forward(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 1]
                input_pos=None,
                ):

        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)

        z = self.decode(torch.cat((z, propagate_pos), dim=-1))
        return z  # [b, n, c]


class STPointWiseDecoder2D(nn.Module):
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 out_steps,
                 scale=8,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.out_steps = out_steps
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(3, self.latent_channels//2, scale=scale),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 1,
                                                self.latent_channels, self.latent_channels,
                                                residual=False,
                                                use_ffn=False,
                                                relative_emb=True,
                                                scale=1.,
                                                relative_emb_dim=2,
                                                min_freq=1/64)

        self.to_out = nn.Sequential(
            nn.LayerNorm(self.latent_channels),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels // 2, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))

    def decode(self, z):
        z = self.to_out(z)
        return z

    def forward(self,
                z,              # [b, n, c]
                propagate_pos,  # [b, tn, 3]
                input_pos,      # [b, n, 2]
                ):
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos[:, :, :-1], input_pos)
        z = self.decode(z)
        z = rearrange(z, 'b (t n) c -> b (t c) n', c=self.out_channels, t=self.out_steps)
        return z


class BCDecoder1D(nn.Module):
    # for Burgers equation, using DeepONet formulation
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 decoding_depth,  # 4?
                 scale=8,
                 res=2048,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(1, self.latent_channels, scale=scale),
            nn.GELU(),
            nn.Linear(self.latent_channels*2, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
        )

        self.decoding_transformer = BranchTrunkNet(latent_channels,
                                                   res)

    def forward(self,
                z,  # [b, n, c]
                propagate_pos,  # [b, n, 1]
                ):
        propagate_pos = propagate_pos[0]
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z)

        return z  # [b, n, c]


class PieceWiseDecoder2DSimple(nn.Module):
    # for Darcy flow inverse problem
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 res=141,
                 scale=0.5,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            # nn.Linear(2, self.latent_channels, bias=False),
            # nn.GELU(),
            # nn.Linear(self.latent_channels*2, self.latent_channels, bias=False),
            # nn.GELU(),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.Dropout(0.05),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 4,
                                                self.latent_channels, self.latent_channels,
                                                use_ln=False,
                                                residual=True,
                                                use_ffn=False,
                                                relative_emb=True,
                                                scale=16,
                                                relative_emb_dim=2,
                                                min_freq=1/res)

        # self.init_propagator_params()
        self.to_out = nn.Sequential(
            nn.Linear(self.latent_channels+2, self.latent_channels, bias=False),
            nn.ReLU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.ReLU(),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.ReLU(),
            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))

    def decode(self, z):
        z = self.to_out(z)
        return z

    def forward(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 1]
                input_pos=None,
                ):

        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)

        z = self.decode(torch.cat((z, propagate_pos), dim=-1))
        return z  # [b, n, c]


class NoRelPointWiseDecoder2D(nn.Module):
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 out_steps,  # 10
                 propagator_depth,
                 scale=8,
                 dropout=0.,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.out_steps = out_steps
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            nn.Linear(self.latent_channels, self.latent_channels),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels//2, 'galerkin', 4,
                                                self.latent_channels//2, self.latent_channels//2,
                                                relative_emb=False,
                                                cat_pos=True,
                                                relative_emb_dim=2,
                                                min_freq=1/64)

        self.expand_feat = nn.Linear(self.latent_channels//2, self.latent_channels)

        self.propagator = nn.ModuleList([
               nn.ModuleList([nn.LayerNorm(self.latent_channels),
               nn.Sequential(
                    nn.Linear(self.latent_channels + 2, self.latent_channels, bias=False),
                    nn.GELU(),
                    nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                    nn.GELU(),
                    nn.Linear(self.latent_channels, self.latent_channels, bias=False))])
            for _ in range(propagator_depth)])

        self.to_out = nn.Sequential(
            nn.LayerNorm(self.latent_channels),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels // 2, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels * self.out_steps, bias=True))

    def propagate(self, z, pos):
        for layer in self.propagator:
            norm_fn, ffn = layer
            z = ffn(torch.cat((norm_fn(z), pos), dim=-1)) + z
        return z

    def decode(self, z):
        z = self.to_out(z)
        return z

    def get_embedding(self,
                      z,  # [b, n c]
                      propagate_pos,  # [b, n, 2]
                      input_pos
                      ):
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.expand_feat(z)
        return z

    def forward(self,
                z,              # [b, n, c]
                propagate_pos   # [b, n, 2]
                ):
        z = self.propagate(z, propagate_pos)
        u = self.decode(z)
        u = rearrange(u, 'b n (t c) -> b (t c) n', c=self.out_channels, t=self.out_steps)
        return u, z                # [b c_out t n], [b c_latent t n]

    def rollout(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 2]
                forward_steps,
                input_pos):
        history = []
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.expand_feat(z)

        # forward the dynamics in the latent space
        for step in range(forward_steps//self.out_steps):
            z = self.propagate(z, propagate_pos)
            u = self.decode(z)
            history.append(rearrange(u, 'b n (t c) -> b (t c) n', c=self.out_channels, t=self.out_steps))
        history = torch.cat(history, dim=-2)  # concatenate in temporal dimension
        return history  # [b, length_of_history*c, n]


@dataclass
class OFormer2DOutput(BaseOutput):
    """
    The output of [`OFormer2D`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`
    """

    sample: "torch.Tensor"  # noqa: F821


class OFormer2D(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
            self,
            in_channels = 1,
            width = 64,
            height = 64,
            in_emb_dim =96,
            out_seq_emb_dim = 192,
            heads=1,
            depth=5,
            propagator_depth=1,
            **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels
        self.width = width
        self.height = height
        x = np.linspace(0, 1, self.width)
        y = np.linspace(0, 1,  self.height)
        x, y = np.meshgrid(x, y)
        grid = np.stack([x, y], axis=-1)
        pos = np.c_[x.ravel(), y.ravel()]

        self.register_buffer('grid', torch.tensor(grid, dtype=torch.float))
        self.register_buffer('pos', torch.tensor(pos, dtype=torch.float))

        self.encoder = SpatialTemporalEncoder2D(
            input_channels=self.in_channels + 2,
            in_emb_dim=in_emb_dim,
            out_seq_emb_dim=out_seq_emb_dim,
            heads=heads,
            depth=depth,
        )

        self.decoder = PointWiseDecoder2D(
            latent_channels=out_seq_emb_dim*2,
            out_channels=self.in_channels,
            out_steps=1,
            propagator_depth=propagator_depth,
            scale=8,
            dropout=0.0,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: Optional[torch.LongTensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            return_dict: bool = True,
    ):

        if hidden_states.dim() < 4:
            hidden_states = hidden_states.unsqueeze(0)

        # (batch_dim, channels, height, width) -> (batch_dim, height*width, channels)
        batch_dim = hidden_states.size(0)
        hidden_states = hidden_states.view(batch_dim, -1, hidden_states.size(1))

        # add batch size to pos
        pos = self.pos.unsqueeze(0).expand(batch_dim, -1, -1)
        #grid = self.grid.unsqueeze(0).expand(batch_dim, -1, -1, -1)

        input = torch.cat([hidden_states, pos], dim=-1)

        z = self.encoder.forward(input, pos)

        x = self.decoder.rollout(z, pos, 1, pos) # (batch_size, 1 * in_channels, n_grid*n_grid)

        # (batch_dim, 1 * channels, height * width) -> (batch_dim, channels, height, width)

        output = x.view(batch_dim, self.in_channels, self.width, self.height)

        if not return_dict:
            return (output,)

        return OFormer2DOutput(sample=output)