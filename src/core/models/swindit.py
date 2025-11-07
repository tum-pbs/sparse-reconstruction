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

from src.core.models.fastervit import AdaLayerNormZero, Mlp
from src.core.models.udit import FinalLayer, precompute_freqs_cis_2d, apply_rotary_emb


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

class SwinDiTStage(nn.Module):
    def __init__(
        self, dim: int, input_resolution: list[int], depth: int,
            num_heads: int, window_size: int, drop_path, downsample,
            pretrained_window_size=0, qkv_bias=True, periodic=False
    ):
        super().__init__()

        self.dim = dim
        blocks = []
        for i in range(depth):
            block = SwinDiTBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                window_size=window_size,
                pretrained_window_size=pretrained_window_size,
                periodic=periodic,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        self.pointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        conditioning: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        hidden_states = torch.permute(hidden_states, (0, 2, 3, 1))
        batch_size, height, width, _ = hidden_states.shape
        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                layer_head_mask,
                output_attentions,
                conditioning=conditioning,
            )

            hidden_states = layer_outputs[0]

        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(hidden_states_before_downsampling, [height, width])
        else:
            output_dimensions = (height, width, height, width)

        hidden_states = torch.permute(hidden_states, (0, 3, 1, 2))

        return hidden_states

        # stage_outputs = (hidden_states, hidden_states_before_downsampling, output_dimensions)
        #
        # if output_attentions:
        #     stage_outputs += layer_outputs[1:]
        # return stage_outputs

class TokenInitializer(nn.Module):
    """
    Carrier token Initializer based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """
    def __init__(self,
                 dim,
                 window_size):

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
        # x = x.permute(0, 3, 1, 2)
        B, C, H, W = x.shape

        pad_right = (self.window_size - W % self.window_size) % self.window_size
        pad_bottom = (self.window_size - H % self.window_size) % self.window_size

        x = F.pad(x, (0, pad_right, 0, pad_bottom, 0, 0, 0, 0), mode='constant', value=0)

        x = torch.nn.functional.avg_pool2d(x, kernel_size=(self.window_size, self.window_size), stride=(self.window_size, self.window_size),
                                           divisor_override=(self.window_size - pad_right) * (self.window_size - pad_bottom), padding=0)

        x = x.permute(0, 2, 3, 1)

        # TODO : Check if this is correct
        # ct = x.view(B, C, int(H // self.window_size), self.window_size, int(W // self.window_size), self.window_size)
        # ct = ct.permute(0, 2, 4, 3, 5, 1).reshape(-1, H*W, C)
        # return ct

        return x

class SwinDiTSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, pretrained_window_size=[0, 0],
                 qkv_bias: bool=True, attention_probs_dropout_prob: float=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        )
        self.pretrained_window_size = pretrained_window_size
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        # mlp to generate continuous relative position bias
        self.continuous_position_bias_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False)
        )

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.int64).float()
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.int64).float()
        relative_coords_table = (
            torch.stack(meshgrid([relative_coords_h, relative_coords_w], indexing="ij"))
            .permute(1, 2, 0)
            .contiguous()
            .unsqueeze(0)
        )  # [1, 2*window_height - 1, 2*window_width - 1, 2]
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
        elif window_size > 1:
            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / math.log2(8)
        )
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=qkv_bias)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        batch_size, dim, num_channels = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # cosine attention
        attention_scores = nn.functional.normalize(query_layer, dim=-1) @ nn.functional.normalize(
            key_layer, dim=-1
        ).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attention_scores = attention_scores * logit_scale
        relative_position_bias_table = self.continuous_position_bias_mlp(self.relative_coords_table).view(
            -1, self.num_attention_heads
        )
        # [window_height*window_width,window_height*window_width,num_attention_heads]
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        # [num_attention_heads,window_height*window_width,window_height*window_width]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in Swinv2Model forward() function)
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            ) + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

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
    def __init__(self, dim: int, num_heads: int, window_size: int, pretrained_window_size=0):
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
                 ct_correct: bool = False,
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

        self.grid_exists = False
        self.pos_emb = None
        # self.deploy = False
        # relative_bias = torch.zeros(1, num_heads, seq_length, seq_length)
        # self.seq_length = seq_length
        # self.register_buffer("relative_bias", relative_bias)
        self.ct_correct=ct_correct

    # def switch_to_deploy(self):
    #     self.deploy = True

    def forward(self, input_tensor, local_window_size):
        # if self.deploy:
        #     input_tensor += self.relative_bias
        #     return input_tensor
        # else:
        #     self.grid_exists = False
        #
        # if not self.grid_exists:
        #     self.grid_exists = True

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
            -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        n_global_feature = input_tensor.shape[2] - local_window_size

        if n_global_feature > 0 and self.ct_correct:

            step_for_ct= self.window_size[0] / (n_global_feature**0.5+1)
            seq_length = int(n_global_feature ** 0.5)
            indices = []
            for i in range(seq_length):
                for j in range(seq_length):
                    ind = (i+1)*step_for_ct*self.window_size[0] + (j+1)*step_for_ct
                    indices.append(int(ind))

            top_part = relative_position_bias[:, indices, :]
            lefttop_part = relative_position_bias[:, indices, :][:, :, indices]
            left_part = relative_position_bias[:, :, indices]

        relative_position_bias = torch.nn.functional.pad(relative_position_bias, (n_global_feature,
                                                                                  0,
                                                                                  n_global_feature,
                                                                                  0)).contiguous()
        if n_global_feature > 0 and self.ct_correct:
            relative_position_bias = relative_position_bias*0.0
            relative_position_bias[:, :n_global_feature, :n_global_feature] = lefttop_part
            relative_position_bias[:, :n_global_feature, n_global_feature:] = top_part
            relative_position_bias[:, n_global_feature:, :n_global_feature] = left_part

        self.pos_emb = relative_position_bias.unsqueeze(0)

        # self.relative_bias = self.pos_emb

        input_tensor += self.pos_emb
        return input_tensor


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
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # attention positional bias
        self.pos_emb_funct = PosEmbMLPSwinv2D(
            window_size=[resolution, resolution],
            pretrained_window_size=[resolution, resolution],
            num_heads=num_heads,
        )

        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, -1, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.pos_emb_funct(attn, self.resolution**2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
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

class SwinDiTBlock(nn.Module):
    def __init__(self, dim: int, input_resolution: list[int], num_heads: int, window_size:int,
                 shift_size: int = 0, pretrained_window_size: int = 0, layer_norm_eps=1e-5, drop_path_rate=0.0,
                 periodic: bool = False):
        super().__init__()
        self.periodic = periodic
        self.input_resolution = input_resolution
        window_size, shift_size = self._compute_window_shift(
            (window_size, window_size), (shift_size, shift_size)
        )
        self.window_size = window_size[0]
        self.shift_size = shift_size[0]
        self.attention = SwinDiTAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            pretrained_window_size=pretrained_window_size
            if isinstance(pretrained_window_size, collections.abc.Iterable)
            else (pretrained_window_size, pretrained_window_size),
        )
        self.layernorm_before = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.drop_path = Swinv2DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.intermediate = SwinDiTIntermediate(dim) # TODO add mlp_ratio and hidden_activation
        self.output = SwinDiTBlockOutput(dim) # TODO add mlp_ratio and hidden_dropout_prob
        self.layernorm_after = nn.LayerNorm(dim, eps=layer_norm_eps)

    def _compute_window_shift(self, target_window_size, target_shift_size) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        window_size = [r if r <= w else w for r, w in zip(self.input_resolution, target_window_size)]
        shift_size = [0 if r <= w else s for r, w, s in zip(self.input_resolution, window_size, target_shift_size)]
        return window_size, shift_size

    def get_attn_mask(self, height, width, dtype):
        if self.shift_size > 0 and (not self.periodic): # no attention mask, if periodic
            # calculate attention mask for shifted window multihead self attention
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def maybe_pad(self, hidden_states, height, width):
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        conditioning: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, height, width, channels = hidden_states.shape
        shortcut = hidden_states
        shortcut = shortcut.view(batch_size, height * width, channels)

        # pad hidden_states to multiples of window size
        hidden_states = hidden_states.view(batch_size, height, width, channels)

        # shift before padding -> better solution for periodic boundaries
        # cyclic shift
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        shifted_hidden_states, pad_values = self.maybe_pad(shifted_hidden_states, height, width)
        _, height_pad, width_pad, _ = shifted_hidden_states.shape

        # partition windows
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)

        attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
        )

        attention_output = attention_outputs[0]

        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
        shifted_windows = window_reverse(attention_windows, self.window_size, height_pad, width_pad)

        # reverse cyclic shift
        if self.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attention_windows = shifted_windows

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        attention_windows = attention_windows.view(batch_size, height * width, channels)
        hidden_states = self.layernorm_before(attention_windows)

        hidden_states = shortcut + self.drop_path(hidden_states)

        layer_output = self.intermediate(hidden_states)
        layer_output = self.output(layer_output)
        layer_output = hidden_states + self.drop_path(self.layernorm_after(layer_output))

        layer_output = layer_output.view(batch_size, height, width, channels)

        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs

class SwinDiTImpl(nn.Module):
    """
    Diffusion UNet model with a Transformer backbone.
    """

    def __init__(
            self,
            input_size=32,
            down_factor=2,
            in_channels=4,
            out_channels=4,
            window_size=8,
            patch_size=None,
            hidden_size=1152,
            depth=[2, 5, 8, 5, 2],
            num_heads=16,
            mlp_ratio=4,
            class_dropout_prob=0.1,
            num_classes=1000,
            learn_sigma=True,
            periodic=False,
            rep=1,
            ffn_type='rep',
            **kwargs
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else out_channels
        self.num_heads = num_heads
        self.periodic = periodic

        down_factor = down_factor if isinstance(down_factor, list) else [down_factor] * 5

        if patch_size is not None:
            self.x_embedder = SimplePatchEmbed(in_channels, hidden_size, patch_size, bias=True)
            self.patch_size = patch_size
        else:
            self.x_embedder = OverlapPatchEmbed(in_channels, hidden_size, bias=True)
            self.patch_size = 1

        # self.t_embedder_1 = TimestepEmbedder(hidden_size)
        # self.y_embedder_1 = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        self.t_embedder_2 = TimestepEmbedder(hidden_size * 2)
        self.y_embedder_2 = LabelEmbedder(num_classes, hidden_size * 2, class_dropout_prob)

        # self.t_embedder_3 = TimestepEmbedder(hidden_size * 4)
        # self.y_embedder_3 = LabelEmbedder(num_classes, hidden_size * 4, class_dropout_prob)

        # encoder-1
        self.encoder_level_1 = SwinDiTStage(dim=hidden_size, input_resolution=[input_size, input_size], num_heads=num_heads,
                                            window_size=window_size, depth=depth[0], downsample=None, drop_path=None, periodic=self.periodic)
        # self.encoder_level_1 = nn.ModuleList([
        #     U_DiTBlock(input_size, hidden_size, num_heads, mlp_ratio=mlp_ratio, down_factor=down_factor[0], rep=rep,
        #                ffn_type=ffn_type, **kwargs) for _ in range(depth[0])
        # ])
        self.down1_2 = Downsample(hidden_size)

        # encoder-2
        self.encoder_level_2 = SwinDiTStage(dim=hidden_size * 2, input_resolution=[input_size // 2, input_size // 2], num_heads=num_heads,
                                            window_size=window_size, depth=depth[1], downsample=None, drop_path=None, periodic=self.periodic)
        # self.encoder_level_2 = nn.ModuleList([
        #     SwinDiTBlock(input_size // 2, hidden_size * 2, num_heads, mlp_ratio=mlp_ratio, down_factor=down_factor[1],
        #                rep=rep, ffn_type=ffn_type, **kwargs) for _ in range(depth[1])
        # ])
        self.down2_3 = Downsample(hidden_size * 2)

        # latent
        self.latent = SwinDiTStage(dim=hidden_size * 4, input_resolution=[input_size // 4, input_size // 4], num_heads=num_heads,
                                            window_size=window_size, depth=depth[2], downsample=None, drop_path=None, periodic=self.periodic)

        # self.latent = nn.ModuleList([
        #     SwinDiTBlock(input_size // 4, hidden_size * 4, num_heads, mlp_ratio=mlp_ratio, down_factor=down_factor[2],
        #                rep=rep, ffn_type=ffn_type, **kwargs) for _ in range(depth[2])
        # ])


        # decoder-2
        self.up3_2 = Upsample(int(hidden_size * 4))  ## From Level 4 to Level 3
        self.reduce_chan_level2 = nn.Conv2d(int(hidden_size * 4), int(hidden_size * 2), kernel_size=1, bias=True)

        self.decoder_level_2 = SwinDiTStage(dim=hidden_size * 2, input_resolution=[input_size // 2, input_size // 2], num_heads=num_heads,
                                            window_size=window_size, depth=depth[3], downsample=None, drop_path=None, periodic=self.periodic)

        # self.decoder_level_2 = nn.ModuleList([
        #     SwinDiTBlock(input_size // 2, hidden_size * 2, num_heads, mlp_ratio=mlp_ratio, down_factor=down_factor[3],
        #                rep=rep, ffn_type=ffn_type, **kwargs) for _ in range(depth[3])
        # ])

        # decoder-1
        self.up2_1 = Upsample(int(hidden_size * 2))  ## From Level 4 to Level 3
        self.reduce_chan_level1 = nn.Conv2d(int(hidden_size * 2), int(hidden_size * 2), kernel_size=1, bias=True)

        self.decoder_level_1 = SwinDiTStage(dim=hidden_size * 2, input_resolution=[input_size, input_size], num_heads=num_heads,
                                            window_size=window_size, depth=depth[4], downsample=None, drop_path=None, periodic=self.periodic)

        # self.decoder_level_1 = nn.ModuleList([
        #     SwinDiTBlock(input_size, hidden_size * 2, num_heads, mlp_ratio=mlp_ratio, down_factor=down_factor[4], rep=rep,
        #                ffn_type=ffn_type, **kwargs) for _ in range(depth[4])
        # ])

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

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder_1.embedding_table.weight, std=0.02)
        nn.init.normal_(self.y_embedder_2.embedding_table.weight, std=0.02)
        # nn.init.normal_(self.y_embedder_3.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        # nn.init.normal_(self.t_embedder_1.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.t_embedder_1.mlp[2].weight, std=0.02)
        #
        nn.init.normal_(self.t_embedder_2.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder_2.mlp[2].weight, std=0.02)
        #
        # nn.init.normal_(self.t_embedder_3.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.t_embedder_3.mlp[2].weight, std=0.02)

        # # Zero-out adaLN modulation layers in IPT blocks:
        # blocks = self.encoder_level_1 + self.encoder_level_2 + self.latent + self.decoder_level_2 + self.decoder_level_1
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

    def forward(self, x, t, y):
        """
        Forward pass of U-DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x)  # (N, C, H, W)

        # t1 = self.t_embedder_1(t)  # (N, C, 1, 1)
        # y1 = self.y_embedder_1(y, self.training)  # (N, C, 1, 1)
        # c1 = t1 + y1  # (N, D, 1, 1)
        #
        t2 = self.t_embedder_2(t)  # (N, C, 1, 1)
        y2 = self.y_embedder_2(y, self.training)  # (N, C, 1, 1)
        c2 = t2 + y2  # (N, D, 1, 1)
        #
        # t3 = self.t_embedder_3(t)  # (N, C, 1, 1)
        # y3 = self.y_embedder_3(y, self.training)  # (N, C, 1, 1)
        # c3 = t3 + y3  # (N, D, 1, 1)

        # encoder_1
        out_enc_level1 = x
        out_enc_level1 = self.encoder_level_1(out_enc_level1, ) # c1) TODO add time/label conditioning c1
        # for block in self.encoder_level_1:
        #     out_enc_level1 = block(out_enc_level1, c1)
        inp_enc_level2 = self.down1_2(out_enc_level1)

        # encoder_2
        out_enc_level2 = inp_enc_level2
        out_enc_level2 = self.encoder_level_2(out_enc_level2, ) # c1)
        # for block in self.encoder_level_2:
        #     out_enc_level2 = block(out_enc_level2, c2)
        inp_enc_level3 = self.down2_3(out_enc_level2)

        # latent
        latent = inp_enc_level3
        latent = self.latent(latent) # , c3)
        # for block in self.latent:
        #     latent = block(latent, c3)

        # decoder_2
        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = inp_dec_level2
        out_dec_level2 = self.decoder_level_2(out_dec_level2) # , c2)
        # for block in self.decoder_level_2:
        #     out_dec_level2 = block(out_dec_level2, c2)

        # decoder_1
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = inp_dec_level1
        out_dec_level1 = self.decoder_level_1(out_dec_level1) # , c2)
        # for block in self.decoder_level_1:
        #     out_dec_level1 = block(out_dec_level1, c2)

        # output
        x = self.output(out_dec_level1)

        x = self.final_layer(x, c2)  # (N, T, patch_size ** 2 * out_channels)

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

class SwinDiT(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
            self,
            sample_size: int,
            in_channels: int,
            out_channels: int,
            type: str,
            periodic: bool = False,
            patch_size: Optional[int] = None,
    ):
        super(SwinDiT, self).__init__()
        args = {'in_channels': in_channels, 'out_channels': out_channels, 'patch_size': patch_size, 'learn_sigma': False,
                'periodic': periodic, 'input_size': sample_size}

        self.model: SwinDiTImpl = SwinDiT_models[type](**args)
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

        if len(timestep.shape) == 0:
            timestep = timestep.unsqueeze(0)
            timestep = timestep.repeat(hidden_states.shape[0])
            timestep = timestep.to(hidden_states.device)

        # timestep scaling (from 0 - 1 to 0 - 1000)
        timestep = timestep * 1000.0

        output = self.model.forward(hidden_states, timestep, class_labels)

        if not return_dict:
            return (output,)

        return SwinDiTOutput(sample=output)

#################################################################################
#                                   U-DITs Configs                                  #
#################################################################################

def SwinDiT_custom(**kwargs):
    return SwinDiTImpl(**kwargs)


def SwinDiT_S(**kwargs):
    return SwinDiTImpl(down_factor=2, hidden_size=96, num_heads=4, depth=[2, 5, 8, 5, 2], ffn_type='rep', rep=1, mlp_ratio=2,
                 attn_type='v2', posemb_type='rope2d', downsampler='dwconv5', down_shortcut=1, **kwargs)


def SwinDiT_B(**kwargs):
    return SwinDiTImpl(down_factor=2, hidden_size=192, num_heads=8, depth=[2, 5, 8, 5, 2], ffn_type='rep', rep=1, mlp_ratio=2,
                 attn_type='v2', posemb_type='rope2d', downsampler='dwconv5', down_shortcut=1, **kwargs)


def SwinDiT_L(**kwargs):
    return SwinDiTImpl(down_factor=2, hidden_size=384, num_heads=16, depth=[2, 5, 8, 5, 2], ffn_type='rep', rep=1,
                 mlp_ratio=2, attn_type='v2', posemb_type='rope2d', downsampler='dwconv5', down_shortcut=1, **kwargs)


SwinDiT_models = {
    'SwinDiT-custom': SwinDiT_custom,
    'SwinDiT-S': SwinDiT_S,  # U-DiT-S
    'SwinDiT-B': SwinDiT_B,  # U-DiT-B
    'SwinDiT-L': SwinDiT_L,  # U-DiT-L
}
