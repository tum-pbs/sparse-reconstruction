from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import torch.nn.functional as F
import torch.nn as nn
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.models.swinv2.modeling_swinv2 import Swinv2Attention, Swinv2DropPath, Swinv2Intermediate, Swinv2Output
import math
import collections.abc

import numpy as np
from timm.models.layers import trunc_normal_, DropPath
import torch
from transformers.pytorch_utils import meshgrid, find_pruneable_heads_and_indices, prune_linear_layer

from core.models.fastervit import AdaLayerNormZero, Mlp
from core.models.udit import FinalLayer, precompute_freqs_cis_2d, apply_rotary_emb


###############################
# We need to create subclass of Swinv2PreTrainedModel because it sets use_mask_token=True
# This will create a parameter swinv2.embeddings.mask_token that will receive no gradient if bool_masked_pos is None
# This then triggers https://github.com/Lightning-AI/pytorch-lightning/issues/17212
###############################

# Copied from transformers.models.swin.modeling_swin.window_partition
def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape

    input_feature = input_feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    return windows


# Copied from transformers.models.swin.modeling_swin.window_reverse
def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    num_channels = windows.shape[-1]
    windows = windows.view(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, num_channels)
    return windows

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class MaskEmbedder(nn.Module):
    """
    Embeds mask tokens into vector representations.
    """

    def __init__(self, hidden_size: int, mask_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(mask_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, mask):
        return self.mlp(mask)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class SimplePatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, patch_size=4, bias=True):
        super(SimplePatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c: int = 3, embed_dim: int = 48, patch_size: int = 4, overlap_size: int = 1,
                 bias:bool = False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=(patch_size+2*overlap_size, patch_size+2*overlap_size),
                              stride=patch_size, padding=overlap_size, bias=bias)

        self.patch_size = patch_size

    def forward(self, x, periodic_x: bool = False, periodic_y: bool = False):

        # x shape = (B, C, H, W)

        if periodic_x:

            x1 = x[:, :, :, :self.patch_size]
            x2 = x[:, :, :, -self.patch_size:]
            x = torch.cat((x2, x, x1), dim=-1)

        if periodic_y:

            x1 = x[:, :, :self.patch_size, :]
            x2 = x[:, :, -self.patch_size:, :]
            x = torch.cat((x2, x, x1), dim=-2)

        x = self.proj(x)

        if periodic_x:

            x = x[:, :, :, 1:-1]

        if periodic_y:

            x = x[:, :, 1:-1, :]

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class DownsampleV2(nn.Module):
    def __init__(self, n_feat):
        super(DownsampleV2, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class UpsampleV2(nn.Module):
    def __init__(self, n_feat):
        super(UpsampleV2, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
    def forward(self, x):
        return self.body(x)

class TokenInitializer(nn.Module):
    """
    Carrier token Initializer based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """
    def __init__(self,
                 dim,
                 # input_resolution,
                 window_size):
                 # ct_size=1):
        """
        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            window_size: window size.
            ct_size: spatial dimension of carrier token local window
        """
        super().__init__()

        # output_size1 = int(np.ceil(ct_size * input_resolution[0] /window_size))
        # stride_size1 = int(np.ceil(input_resolution[0]/output_size1))
        # kernel_size1 = input_resolution[0] - (output_size1 - 1) * stride_size1
        # output_size2 = int((ct_size) * input_resolution[1]/window_size)
        # stride_size2 = int(input_resolution[1]/output_size2)
        # kernel_size2 = input_resolution[1] - (output_size2 - 1) * stride_size2

        # self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        # to_global_feature = nn.Sequential()
        # to_global_feature.add_module("pos", self.pos_embed)
        # to_global_feature.add_module("pool", nn.AvgPool2d(kernel_size=(kernel_size1, kernel_size2),
        #                                                   stride=(stride_size1, stride_size2)))
        self.window_size = window_size
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        to_global_feature = nn.Sequential()
        to_global_feature.add_module("pos", self.pos_embed)
        # to_global_feature.add_module("pool", nn.AvgPool2d(kernel_size=(window_size, window_size),
        #                                                   stride=(window_size, window_size)))

        self.to_global_feature = to_global_feature
        self.window_size = window_size

    def forward(self, x):

        x = x.permute(0, 3, 1, 2)

        x = self.to_global_feature(x)

        B, C, H, W = x.shape

        pad_right = (self.window_size - W % self.window_size) % self.window_size
        pad_bottom = (self.window_size - H % self.window_size) % self.window_size

        x = F.pad(x, (0, pad_right, 0, pad_bottom, 0, 0, 0, 0), mode='constant', value=0)

        x = torch.nn.functional.avg_pool2d(x, kernel_size=(self.window_size, self.window_size), stride=(self.window_size, self.window_size),
                                           divisor_override=(self.window_size - pad_right) * (self.window_size - pad_bottom), padding=0)

        x = x.permute(0, 2, 3, 1)

        return x

class FasterDiTStage(nn.Module):
    def __init__(
        self, dim: int, depth: int,
            num_heads: int, window_size: int, drop_path,
            qkv_bias=True, periodic=False, carrier_token_active: bool = True,
    ):
        super().__init__()

        self.dim = dim
        blocks = []
        for i in range(depth):
            block = HATDiTBlock(
                dim=dim,
                num_heads=num_heads,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                window_size=window_size,
                periodic=periodic,
                carrier_token_active=carrier_token_active,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.periodic = periodic
        self.window_size = window_size

        self.carrier_token_active = carrier_token_active

        if self.carrier_token_active:
            self.global_tokenizer = TokenInitializer(dim,
                                                     window_size)


    def maybe_pad(self, hidden_states, height, width):
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def forward(self,
                hidden_states: torch.Tensor,
                cond: Optional[torch.Tensor] = None,
                timestep: Optional[torch.LongTensor] = None,
                class_labels: Optional[torch.LongTensor] = None, ):


        # TODO channel needs to be last here; can we change the code so that channel is always last in FasterDiTStage?
        B, C, H, W = hidden_states.shape

        for n, block in enumerate(self.blocks):

            shift_size = 0 if n % 2 == 0 else self.window_size // 2

            # channels last
            hidden_states = torch.permute(hidden_states, (0, 2, 3, 1))

            if shift_size > 0:
                shifted_hidden_states = torch.roll(hidden_states, shifts=(-shift_size, -shift_size),
                                                   dims=(1, 2))
            else:
                shifted_hidden_states = hidden_states

            shifted_hidden_states, pad_values = self.maybe_pad(shifted_hidden_states, H, W)
            _, height_pad, width_pad, _ = shifted_hidden_states.shape

            if self.carrier_token_active:
                ct = self.global_tokenizer(hidden_states)
            else:
                ct = None

            hidden_states = window_partition(shifted_hidden_states, self.window_size)

            hidden_states, ct = block(hidden_states, ct, timestep=timestep, class_labels=class_labels, emb=cond)

            hidden_states = window_reverse(hidden_states, self.window_size, height_pad, width_pad)

            if height_pad > 0 or width_pad > 0:
                hidden_states = hidden_states[:, :H, :W, :].contiguous()

            if shift_size > 0:
                hidden_states = torch.roll(hidden_states, shifts=(shift_size, shift_size),
                                                   dims=(1, 2))

            hidden_states = torch.permute(hidden_states, (0, 3, 1, 2))

        return hidden_states


class SwinDiTSelfOutput(nn.Module):
    def __init__(self, dim: int, attention_probs_dropout_prob: float = 0.0):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

class SwinDiTAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int,
                 pretrained_window_size=0):

        super().__init__()
        self.self = SwinDiTSelfAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            pretrained_window_size=pretrained_window_size
            if isinstance(pretrained_window_size, collections.abc.Iterable)
            else [pretrained_window_size, pretrained_window_size],
        )
        self.output = SwinDiTSelfOutput(dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class SwinDiTIntermediate(nn.Module):
    def __init__(self, dim: int, mlp_ratio = 4.0, hidden_act: str = "gelu"):
        super().__init__()
        self.dense = nn.Linear(dim, int(mlp_ratio * dim))
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class SwinDiTBlockOutput(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, hidden_dropout_prob: float = 0.0):
        super().__init__()
        self.dense = nn.Linear(int(mlp_ratio * dim), dim)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class PosEmbMLPSwinv2D(nn.Module):
    def __init__(self,
                 window_size: list[int],
                 pretrained_window_size: list[int],
                 num_heads: int,
                 no_log=False):
        super().__init__()

        self.window_size = [int(ws) for ws in window_size]
        self.num_heads = num_heads

        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)

        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)

        if not no_log:
            relative_coords_table *= 8  # normalize to -8, 8
            relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).int()

        self.register_buffer("relative_position_index", relative_position_index)

        self.pos_emb = None


    def forward(self, input_tensor, local_window_size):

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
            -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        n_global_feature = input_tensor.shape[2] - local_window_size

        relative_position_bias = torch.nn.functional.pad(relative_position_bias, (n_global_feature,
                                                                                  0,
                                                                                  n_global_feature,
                                                                                  0)).contiguous()

        self.pos_emb = relative_position_bias.unsqueeze(0)

        input_tensor += self.pos_emb
        return input_tensor

class CarrierTokenAttention2DTimestep(nn.Module):


    def __init__(self, dim, num_heads,
                 bias=False, posemb_type='rope2d', attn_type='v2', down_shortcut=False, **kwargs):

        super(CarrierTokenAttention2DTimestep, self).__init__()
        if kwargs != dict():  # is not empty
            print(f'Kwargs: {kwargs}')

        self.dim = dim
        self.heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.to_out = nn.Conv2d(dim, dim, 1)

        # v2
        self.attn_type = attn_type
        if attn_type == 'v2':
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)


        # downsample
        self.dh = 1
        self.dw = 1

        # posemb
        self.posemb_type = posemb_type

        # posemb type
        if self.posemb_type == 'rope2d':
            self.freqs_cis = None

    def forward(self, x):
        # b, h, w, c = x.size()

        # x = rearrange(x, 'b h w c -> b 1 (h w) c')

        b, n, c = x.size()
        x = torch.unsqueeze(x, 1)

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if self.posemb_type == 'rope2d':

            if self.freqs_cis is None or self.freqs_cis.shape[0] != n:

                self.freqs_cis = precompute_freqs_cis_2d(self.dim // self.heads, n).to(x.device)

            # q, k input shape: B N H Hc
            q, k = map(lambda t: rearrange(t, 'b p n (h d) -> (b p) n h d', h=self.heads), qkv[:-1])

            v = rearrange(qkv[2], 'b p n (h d) -> b p h n d', h=self.heads)

            q, k = apply_rotary_emb(q, k, freqs_cis=self.freqs_cis)
            # reshape back

            q = rearrange(q, '(b p) n h d -> b p h n d', b=b)
            k = rearrange(k, '(b p) n h d -> b p h n d', b=b)

        else:

            q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        if self.attn_type is None:  # v1 attention

            attn = (q @ k.transpose(-2, -1))
            attn = attn * self.scale

        elif self.attn_type == 'v2':  # v2 attention

            attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
            logit_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
            attn = attn * logit_scale

        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        # upsample
        # x = rearrange(x, 'b (dh dw) he (h w) d -> b (h dh) (w dw) (he d)',
        #               h=h // self.dh,
        #               w=w // self.dw,
        #               dh=self.dh,
        #               dw=self.dw)

        x = rearrange(x, 'b p h n d -> b p n (h d)')

        # x = self.to_out(x)

        x = x[:, 0]

        return x

class WindowAttention2DTime(nn.Module):
    """
    Window attention based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        resolution: int = 0,
        attn_type='v2',
    ):
        super().__init__()
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
            resolution: feature resolution.
            seq_length: sequence length.
        """
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

        # attention positional bias
        self.pos_emb_funct = PosEmbMLPSwinv2D(
            window_size=[resolution, resolution],
            pretrained_window_size=[resolution, resolution],
            num_heads=num_heads,
        )
        self.attn_type = attn_type

        if attn_type == 'v2':
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, -1, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.attn_type == 'v1':
            attn = (q @ k.transpose(-2, -1)) * self.scale

        elif self.attn_type == 'v2':
            attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
            logit_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
            attn = attn * logit_scale

        attn = self.pos_emb_funct(attn, self.resolution ** 2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)

        return x

def ct_window_simple(ct, W, H, window_size):
    bs = ct.shape[0]
    N = ct.shape[2]
    ct = ct.view(
        bs,
        H,
        W,
        N,
    )
    return ct

def ct_dewindow_simple(ct, W, H, window_size):
    bs = ct.shape[0]
    N = ct.shape[2]
    ct2 = ct.view(
        -1,
        H,
        W,
        N,
    ).permute(0, 3, 1, 2)
    ct2 = ct2.reshape(bs, N, W * H).transpose(1, 2)
    return ct2

def ct_window(ct, W, H, window_size):
    bs = ct.shape[0]
    N = ct.shape[2]
    ct = ct.view(
        bs,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        N,
    )
    ct = ct.permute(0, 1, 3, 2, 4, 5)
    return ct

def ct_dewindow(ct, W, H, window_size):
    bs = ct.shape[0]
    N = ct.shape[2]
    ct2 = ct.view(
        -1,
        W // window_size,
        H // window_size,
        window_size,
        window_size,
        N,
    ).permute(0, 5, 1, 3, 2, 4)
    ct2 = ct2.reshape(bs, N, W * H).transpose(1, 2)
    return ct2

class PosEmbMLPSwinv1D(nn.Module):
    def __init__(self,
                 dim,
                 rank=2,
                 conv=False):
        super().__init__()
        self.rank = rank
        if not conv:
            self.cpb_mlp = nn.Sequential(nn.Linear(self.rank, 512, bias=True),
                                         nn.ReLU(),
                                         nn.Linear(512, dim, bias=False))
        else:
            self.cpb_mlp = nn.Sequential(nn.Conv1d(self.rank, 512, 1,bias=True),
                                         nn.ReLU(),
                                         nn.Conv1d(512, dim, 1,bias=False))
        self.grid_exists = False
        self.pos_emb = None
        # self.deploy = False
        # relative_bias = torch.zeros(1, seq_length, dim)
        # self.register_buffer("relative_bias", relative_bias)
        self.conv = conv

    # def switch_to_deploy(self):
    #     self.deploy = True

    def forward(self, input_tensor):
        # seq_length = input_tensor.shape[1] if not self.conv else input_tensor.shape[2]

        # height = int(math.sqrt(seq_length))
        # width = height

        # if self.deploy:
        #     return input_tensor + self.relative_bias
        # else:
        #     self.grid_exists = False
        # if not self.grid_exists:
        #     self.grid_exists = True

        if self.rank == 1:

            seq_length = input_tensor.shape[1] if not self.conv else input_tensor.shape[2]

            relative_coords_h = torch.arange(0, seq_length, device=input_tensor.device, dtype = input_tensor.dtype)
            relative_coords_h -= seq_length//2
            relative_coords_h /= (seq_length//2)
            relative_coords_table = relative_coords_h
            self.pos_emb = self.cpb_mlp(relative_coords_table.unsqueeze(0).unsqueeze(2))
            # self.relative_bias = self.pos_emb

        else:

            height = input_tensor.shape[1]
            width = input_tensor.shape[2]

            relative_coords_h = torch.arange(0, height, device=input_tensor.device, dtype = input_tensor.dtype)
            relative_coords_w = torch.arange(0, width, device=input_tensor.device, dtype = input_tensor.dtype)
            relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w])).contiguous().unsqueeze(0)
            relative_coords_table[:,0] -= height // 2
            relative_coords_table[:,1] -= width // 2
            relative_coords_table[:,0] /= max((height // 2), 1.0) # special case for 1x1
            relative_coords_table[:,1] /= max((width // 2), 1.0) # special case for 1x1
            if not self.conv:
                # self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2).transpose(1,2))
                self.pos_emb = self.cpb_mlp(relative_coords_table.permute(0, 2, 3, 1))
            else:
                self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2))

            # self.relative_bias = self.pos_emb

        input_tensor = input_tensor + self.pos_emb
        return input_tensor

class HATDiTBlock(nn.Module):
    """
    Hierarchical attention (HAT) based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention

    Modifications:
        - 2D + time
        - AdaIN for diffusion conditioning
        - Conditioning via context tokens
        - Sliding window
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        num_embeds_ada_norm: Optional[int] = 1000,
        window_size=7,
        last=False,
        layer_scale=None,
        ct_size=1,
        do_propagation=False,
        carrier_token_active=True,
        shift_size=0,
        periodic=False,
    ):
        super().__init__()
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            mlp_ratio: MLP ratio.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
            act_layer: activation function.
            norm_layer: normalization layer.
            num_embeds_ada_norm: number of embeddings for AdaLayerNormZero.
            sr_ratio: input to window size ratio.
            window_size: window size.
            last: last layer flag.
            layer_scale: layer scale coefficient.
            ct_size: spatial dimension of carrier token local window.
            do_propagation: enable carrier token propagation.
        """
        # positional encoding for windowed attention tokens
        self.pos_embed = PosEmbMLPSwinv1D(dim, rank=2)
        self.norm1 = norm_layer(dim)

        self.carrier_token_active = carrier_token_active


        self.cr_window = 1
        self.attn = WindowAttention2DTime(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            resolution=window_size,
        )

        # self.attn_swin = SwinDiTAttention(
        #     dim=dim,
        #     num_heads=num_heads,
        #     window_size=window_size,
        #     pretrained_window_size=[window_size, window_size],
        # )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.window_size = window_size

        self.adain_2 = AdaLayerNormZero(dim, num_embeddings=None, norm_type="layer_norm")

        if self.carrier_token_active:

            # if do hierarchical attention, this part is for carrier tokens
            self.hat_norm1 = norm_layer(dim)
            self.hat_norm2 = norm_layer(dim)
            self.hat_attn = CarrierTokenAttention2DTimestep(
                dim=dim,
                num_heads=num_heads,
            )

            self.hat_mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
            self.hat_drop_path = (
                DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            )
            self.hat_pos_embed = PosEmbMLPSwinv1D(
                dim, rank=2,
            )

            self.adain_1 = AdaLayerNormZero(dim, num_embeddings=None, norm_type="layer_norm")
            self.upsampler = nn.Upsample(size=window_size, mode="nearest")

        # keep track for the last block to explicitly add carrier tokens to feature maps
        self.last = last
        self.do_propagation = do_propagation

        layer_norm_eps = 1e-5
        self.layernorm_before = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.intermediate = SwinDiTIntermediate(dim)
        self.output = SwinDiTBlockOutput(dim)
        self.layernorm_after = nn.LayerNorm(dim, eps=layer_norm_eps)

    def forward(self, x, carrier_tokens,
                timestep: Optional[torch.LongTensor] = None,
                class_labels: Optional[torch.LongTensor] = None,
                emb: Optional[torch.LongTensor] = None):

        B, H, W, N = x.shape
        ct = carrier_tokens

        # Do we need pos_embed here?
        # x = self.pos_embed(x)

        x = x.view(B, H * W, N)

        Bc = emb.shape[0]

        if self.carrier_token_active:

            Bc, Hc, Wc, Nc = ct.shape

            # positional bias for carrier tokens
            ct = self.hat_pos_embed(ct)
            ct = ct.reshape(Bc, Hc * Wc, Nc)


            ######## DiT block with MSA, MLP, and AdaIN ########
            msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate = self.adain_1(timestep=timestep,
                                                                                          class_labels=class_labels,                                                                      emb=emb)
            ct_msa = self.hat_norm1(ct)
            ct_msa = ct_msa * (1 + msa_scale[:, None]) + msa_shift[:, None]
            # attention plus mlp

            s = int(math.sqrt(ct.shape[1]))
            # ct_msa = rearrange(ct_msa, 'b (sa sb) d -> b d sa sb', sa=s, sb=s)
            ct_msa = self.hat_attn(ct_msa)
            # ct_msa = rearrange(ct_msa, 'b d sa sb -> b (sa sb) d', sa=s, sb=s)
            ct_msa = ct_msa * (1 + msa_gate[:, None])
            ct = ct + self.hat_drop_path(ct_msa)

            ct_mlp = self.hat_norm2(ct)
            ct_mlp = ct_mlp * (1 + mlp_scale[:, None]) + mlp_shift[:, None]
            ct_mlp = self.hat_mlp(ct_mlp)
            ct_mlp = ct_mlp * (1 + mlp_gate[:, None])

            ct = ct + self.hat_drop_path(ct_mlp)
            ct = ct.reshape(x.shape[0], -1, N)

            # concatenate carrier_tokens to the windowed tokens
            x = torch.cat((ct, x), dim=1)


        ########### DiT block with MSA, MLP, and AdaIN ############
        msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate = self.adain_2(timestep=timestep,
                                                                                      class_labels=class_labels,
                                                                                      emb=emb)

        num_windows_total = int(B // Bc)

        msa_shift = msa_shift.repeat_interleave(num_windows_total, dim=0)
        msa_scale = msa_scale.repeat_interleave(num_windows_total, dim=0)
        msa_gate = msa_gate.repeat_interleave(num_windows_total, dim=0)
        mlp_shift = mlp_shift.repeat_interleave(num_windows_total, dim=0)
        mlp_scale = mlp_scale.repeat_interleave(num_windows_total, dim=0)
        mlp_gate = mlp_gate.repeat_interleave(num_windows_total, dim=0)

        x_msa = self.norm1(x)

        x_msa = x_msa * (1 + msa_scale[:, None]) + msa_shift[:, None]

        x_msa = self.attn(x_msa)
        x_msa = x_msa * (1 + msa_gate[:, None])

        x = x + self.drop_path(x_msa)

        x_mlp = self.norm2(x)

        x_mlp = x_mlp * (1 + mlp_scale[:, None]) + mlp_shift[:, None]
        x_mlp = self.mlp(x_mlp)
        x_mlp = x_mlp * (1 + mlp_gate[:, None])
        x = x + self.drop_path(x_mlp)

        #############################################################
        ################ Split carrier tokens and normal tokens #####

        if self.carrier_token_active:

            # for hierarchical attention we need to split carrier tokens and window tokens back
            ctr, x = x.split(
                [
                    x.shape[1] - self.window_size * self.window_size,
                    self.window_size * self.window_size,
                ],
                dim=1,
            )

            ct = ctr.reshape(Bc, Hc * Wc, Nc)  # reshape carrier tokens.

            if self.last and self.do_propagation:
                # propagate carrier token information into the image
                ctr_image_space = ctr.transpose(1, 2).reshape(
                    B, N, self.cr_window, self.cr_window
                )
                x = x + self.gamma1 * self.upsampler(
                    ctr_image_space.to(dtype=torch.float32)
                ).flatten(2).transpose(1, 2).to(dtype=x.dtype)

        return x, ct


class FasterDiTImpl(nn.Module):
    """
    Diffusion UNet model with a Transformer backbone.
    """

    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            window_size: int = 8,
            patch_size: int = 4,
            overlap_size: int = 0,
            hidden_size: int = 1152,
            depth: Tuple[int] =(2, 4, 4, 6, 4, 4, 2),
            num_heads: int = 16,
            mlp_ratio: int = 4,
            class_dropout_prob = 0.4,
            num_classes: int = 1000,
            learn_sigma: bool = True,
            periodic: bool = False,
            use_carrier_tokens: bool = True,
            apply_shifts: bool = True,
            **kwargs
    ):
        super().__init__()

        assert len(depth) % 2 == 1, "Encoder and decoder depths must be equal."
        self.num_encoder_layers = len(depth) // 2

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else out_channels

        self.num_heads = num_heads
        self.periodic = periodic

        dit_stage_args = {
            "drop_path": None,
            "periodic": periodic,
            'use_carrier_tokens': use_carrier_tokens,
            'mlp_ratio': mlp_ratio,
            'apply_shifts': apply_shifts,
        }

        self.x_embedder = OverlapPatchEmbed(in_c=in_channels, embed_dim=hidden_size,
                                            patch_size=patch_size, overlap_size=overlap_size,
                                            bias=True)
        self.patch_size = patch_size

        # timestep and label embedders
        for i in range(self.num_encoder_layers + 1):
            self.__setattr__(f"t_embedder_{i}", TimestepEmbedder(hidden_size * 2 ** i))
            self.__setattr__(f"y_embedder_{i}", LabelEmbedder(num_classes, hidden_size * 2 ** i, class_dropout_prob))

        # encoder
        for i in range(self.num_encoder_layers):
            self.__setattr__(f"encoder_level_{i}", FasterDiTStage(dim=hidden_size * 2 ** i, num_heads=num_heads,
                                            window_size=window_size, depth=depth[i], **dit_stage_args))
            self.__setattr__(f"down{i}_{i+1}", Downsample(hidden_size * 2 ** i))

        # latent
        self.latent = FasterDiTStage(dim=hidden_size * 2 ** self.num_encoder_layers, num_heads=num_heads,
                                            window_size=window_size, depth=depth[self.num_encoder_layers], **dit_stage_args)


        # double hidden size for last decoder layer 0
        self.__setattr__("up1_0", Upsample(hidden_size * 2))
        self.__setattr__("reduce_chan_level0", nn.Conv2d(hidden_size * 2, hidden_size * 2, kernel_size=1, bias=True))
        self.__setattr__("decoder_level_0", FasterDiTStage(dim=hidden_size * 2, num_heads=num_heads,
                                        window_size=window_size, depth=depth[self.num_encoder_layers + 1], **dit_stage_args))

        # decoder layers 1 - num_encoder_layers
        for i in range(1, self.num_encoder_layers):
            self.__setattr__(f"up{i+1}_{i}", Upsample(hidden_size * 2 ** (i+1)))
            self.__setattr__(f"reduce_chan_level{i}", nn.Conv2d(hidden_size * 2 ** (i+1), hidden_size * 2 ** i, kernel_size=1, bias=True))
            self.__setattr__(f"decoder_level_{i}", FasterDiTStage(dim=hidden_size * 2 ** i, num_heads=num_heads,
                                            window_size=window_size, depth=depth[self.num_encoder_layers + i + 1], **dit_stage_args))

        self.output = nn.Conv2d(int(hidden_size * 2), int(hidden_size * 2), kernel_size=3, stride=1, padding=1,
                                bias=True)

        self.final_layer = FinalLayer(hidden_size * 2, self.out_channels * self.patch_size * self.patch_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        for i in range(self.num_encoder_layers):
            # Initialize label embedding table:
            nn.init.normal_(self.__getattr__(f"y_embedder_{i}").embedding_table.weight, std=0.02)

            # Initialize timestep embedding MLP:
            nn.init.normal_(self.__getattr__(f"t_embedder_{i}").mlp[0].weight, std=0.02)
            nn.init.normal_(self.__getattr__(f"t_embedder_{i}").mlp[2].weight, std=0.02)

        # # Zero-out adaLN modulation layers in IPT blocks:
        # blocks = [self.encoder_level_1, self.encoder_level_2,
        #           self.latent, self.decoder_level_2,
        #           self.decoder_level_1]

        # TODO set adaLN modulation layers to zero
        # for block in blocks:
        #     nn.init.constant_(block.adaLN_modulation1[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation1[-1].bias, 0)
        #     nn.init.constant_(block.adaLN_modulation2[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation2[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.out_proj.weight, 0)
        nn.init.constant_(self.final_layer.out_proj.bias, 0)

    def forward(self,
                x: torch.Tensor,  # list of tensors with shape (B, C, T, H, W)
                simulation_time: torch.Tensor, # shape (B, C)
                channel_type: torch.Tensor,  # shape (B, C)
                diffusion_time: torch.Tensor, # shape (B,)
                pde_type: torch.Tensor, # shape (B,)
                simulation_dt: torch.Tensor, # shape (B,)
                diffusion_mask: torch.Tensor, # shape (B, C, T)
               ):
        """
        Forward pass of SwinDiT.
        x (torch.Tensor): input tensor of shape (B, C, T, H, W).
        simulation_time (torch.Tensor): simulation time tensor of shape (B, C).
        channel_type (torch.Tensor): channel type tensor of shape (B, C).
        diffusion_time (torch.Tensor): diffusion time tensor of shape (B,).
        pde_type (torch.Tensor): PDE type tensor of shape (B,).
        simulation_dt (torch.Tensor): simulation time tensor of shape (B,).
        diffusion_mask (torch.Tensor): diffusion mask tensor of shape (B, C, D).
        """

        # Treat the channels separately
        x = x.unsqueeze(2)  # -> (B, C, 1, T, H, W)

        x = self.x_embedder(x)  # -> (N, C, H // patch_size, W // patch_size)

        emb_list = []
        for i in range(self.num_encoder_layers + 1):
            t_emb = self.__getattr__(f"t_embedder_{i}")(t)
            y_emb = self.__getattr__(f"y_embedder_{i}")(y, self.training)


            c = t_emb + y_emb
            emb_list.append(c)

        residuals_list = []
        for i, c in enumerate(emb_list[:-1]):
            # encoder
            out_enc_level = self.__getattr__(f"encoder_level_{i}")(x, c)
            residuals_list.append(out_enc_level)
            x = self.__getattr__(f"down{i}_{i+1}")(out_enc_level)

        c = emb_list[-1]
        x = self.latent(x, c)

        for i, (residual, emb) in enumerate(zip(residuals_list[1:][::-1], emb_list[1:-1][::-1])):
            # decoder
            x = self.__getattr__(f"up{self.num_encoder_layers - i}_{self.num_encoder_layers - i - 1}")(x)
            x = torch.cat([x, residual], 1)
            x = self.__getattr__(f"reduce_chan_level{self.num_encoder_layers - i - 1}")(x)
            x = self.__getattr__(f"decoder_level_{self.num_encoder_layers - i - 1}")(x, emb)

        x = self.__getattr__(f"up1_0")(x)
        x = torch.cat([x, residuals_list[0]], 1)
        x = self.__getattr__(f"reduce_chan_level0")(x)
        x = self.__getattr__(f"decoder_level_0")(x, emb_list[1])

        # output
        x = self.output(x)
        x = self.final_layer(x, emb_list[1])  # (N, T, patch_size ** 2 * out_channels)

        # unpatchify
        x = x.permute(0, 2, 3, 1)

        x = x.reshape(
            shape=x.shape[:3] + (self.patch_size, self.patch_size, self.out_channels)
        )

        height = x.shape[1]
        width = x.shape[2]

        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(
            shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        )

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of IPT, but also batches the unconIPTional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

@dataclass
class SwinDiTOutput(BaseOutput):
    """
    The output of [`SwinDiT`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.Tensor

class FasterDiT(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
            self,
            sample_size: int,
            in_channels: int,
            out_channels: int,
            type: str,
            periodic: bool = False,
            carrier_token_active: bool = True,
            patch_size: Optional[int] = None,
    ):
        super(FasterDiT, self).__init__()
        args = {'in_channels': in_channels, 'out_channels': out_channels, 'patch_size': patch_size, 'learn_sigma': False,
                'periodic': periodic, 'carrier_token_active': carrier_token_active}

        self.model: FasterDiTImpl = FasterDiT_models[type](**args)
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: Optional[torch.LongTensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            return_dict: bool = True,
    ):

        if timestep is None:
            timestep = torch.Tensor([0]).to(hidden_states.device)

        output = self.model.forward(hidden_states, timestep, class_labels)

        if not return_dict:
            return (output,)

        return SwinDiTOutput(sample=output)


#################################################################################
#                                   FasterDITs Configs                          #
#################################################################################


def FasterDiT_custom(**kwargs):
    return FasterDiTImpl(**kwargs)


def FasterDiT_S(**kwargs):
    # TODO, find best config for depth
    return FasterDiTImpl(down_factor=2, hidden_size=96, num_heads=4, depth=[2, 5, 8, 5, 2], ffn_type='rep', rep=1, mlp_ratio=2,
                 attn_type='v2', posemb_type='none', downsampler='dwconv5', down_shortcut=1, **kwargs)


def FasterDiT_B(**kwargs):
    return FasterDiTImpl(down_factor=2, hidden_size=192, num_heads=8, depth=[2, 5, 8, 5, 2], ffn_type='rep', rep=1, mlp_ratio=2,
                 attn_type='v2', posemb_type='rope2d', downsampler='dwconv5', down_shortcut=1, **kwargs)


def FasterDiT_L(**kwargs):
    return FasterDiTImpl(down_factor=2, hidden_size=384, num_heads=16, depth=[2, 5, 8, 5, 2], ffn_type='rep', rep=1,
                 mlp_ratio=2, attn_type='v2', posemb_type='rope2d', downsampler='dwconv5', down_shortcut=1, **kwargs)

FasterDiT_models = {
    'FasterDiT-custom': FasterDiT_custom,
    'FasterDiT-S': FasterDiT_S,  # U-DiT-S
    'FasterDiT-B': FasterDiT_B,  # U-DiT-B
    'FasterDiT-L': FasterDiT_L,  # U-DiT-L
}