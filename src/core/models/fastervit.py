from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import math
import torch
import decimal
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings
# from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils import BaseOutput
from timm.models.layers import trunc_normal_, DropPath

from src.utils import instantiate_from_config


class PixelShuffle3d(nn.Module):
    def __init__(self, upscale_factor=None):
        super().__init__()

        if upscale_factor is None:
            raise TypeError(
                "__init__() missing 1 required positional argument: 'upscale_factor'"
            )

        self.upscale_factor = upscale_factor

    def forward(self, x):
        if x.ndim < 3:
            raise RuntimeError(
                f"pixel_shuffle expects input to have at least 3 dimensions, but got input with {x.ndim} dimension(s)"
            )
        elif x.shape[-4] % self.upscale_factor**3 != 0:
            raise RuntimeError(
                f"pixel_shuffle expects its input's 'channel' dimension to be divisible by the cube of upscale_factor, but input.size(-4)={x.shape[-4]} is not divisible by {self.upscale_factor**3}"
            )

        channels, in_depth, in_height, in_width = x.shape[-4:]
        nOut = channels // self.upscale_factor**3

        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        input_view = x.contiguous().view(
            *x.shape[:-4],
            nOut,
            self.upscale_factor,
            self.upscale_factor,
            self.upscale_factor,
            in_depth,
            in_height,
            in_width,
        )

        axes = torch.arange(input_view.ndim)[:-6].tolist() + [-3, -6, -2, -5, -1, -4]
        output = input_view.permute(axes).contiguous()

        return output.view(*x.shape[:-4], nOut, out_depth, out_height, out_width)

def cube_root(n):
    # Convert n to a PyTorch tensor for high-precision calculations
    n_tensor = torch.tensor(n, dtype=torch.float64)
    result = torch.pow(n_tensor, 1 / 3)
    return int(torch.round(result).item())


def window_partition(x, window_size):
    B, C, H, W, D = x.shape
    x = x.view(
        B,
        C,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        D // window_size,
        window_size,
    )
    windows = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(
        -1, window_size * window_size * window_size, C
    )
    return windows


def window_reverse(windows, window_size, H, W, D, B):
    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        D // window_size,
        window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, windows.shape[2], H, W, D)
    return x


def ct_dewindow(ct, W, H, D, window_size):
    bs = ct.shape[0]
    N = ct.shape[2]
    ct2 = ct.view(
        -1,
        W // window_size,
        H // window_size,
        D // window_size,
        window_size,
        window_size,
        window_size,
        N,
    ).permute(0, 7, 1, 3, 5, 2, 4, 6)
    ct2 = ct2.reshape(bs, N, W * H * D).transpose(1, 2)
    return ct2


def ct_window(ct, W, H, D, window_size):
    bs = ct.shape[0]
    N = ct.shape[2]
    ct = ct.view(
        bs,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        D // window_size,
        window_size,
        N,
    )
    ct = ct.permute(0, 1, 3, 5, 2, 4, 6, 7)
    return ct


def _load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            all_missing_keys,
            unexpected_keys,
            err_msg,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    load = None
    missing_keys = [key for key in all_missing_keys if "num_batches_tracked" not in key]

    if unexpected_keys:
        err_msg.append(
            "unexpected key in source " f'state_dict: {", ".join(unexpected_keys)}\n'
        )
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n'
        )

    if len(err_msg) > 0:
        err_msg.insert(0, "The model and loaded state dict do not match exactly\n")
        err_msg = "\n".join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def _load_checkpoint(model, filename, map_location="cpu", strict=False, logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = torch.load(filename, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    if sorted(list(state_dict.keys()))[0].startswith("encoder"):
        state_dict = {
            k.replace("encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }

    _load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        """
        Args:
            in_features: input features dimension.
            hidden_features: hidden features dimension.
            out_features: output features dimension.
            act_layer: activation function.
            drop: dropout rate.
        """

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_size = x.size()
        x = x.view(-1, x_size[-1])
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.view(x_size)
        return x


class LayerNorm3d(nn.LayerNorm):
    def __init__(self, norm_shape, eps=1e-6, affine=True):
        super().__init__(norm_shape, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 4, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class PosEmbMLPSwinv1D(nn.Module):
    def __init__(self, dim, rank=3, seq_length=4, conv=False):
        super().__init__()
        self.rank = rank
        if not conv:
            self.cpb_mlp = nn.Sequential(
                nn.Linear(self.rank, 512, bias=True),
                nn.ReLU(),
                nn.Linear(512, dim, bias=False),
            )
        else:
            self.cpb_mlp = nn.Sequential(
                nn.Conv1d(self.rank, 512, 1, bias=True),
                nn.ReLU(),
                nn.Conv1d(512, dim, 1, bias=False),
            )
        self.grid_exists = False
        self.pos_emb = None
        self.deploy = False
        relative_bias = torch.zeros(1, seq_length, dim)
        self.register_buffer("relative_bias", relative_bias)
        self.conv = conv

    def switch_to_deploy(self):
        self.deploy = True

    def forward(self, input_tensor):
        seq_length = input_tensor.shape[1] if not self.conv else input_tensor.shape[2]
        if self.deploy:
            return input_tensor + self.relative_bias
        else:
            self.grid_exists = False
        if not self.grid_exists:
            self.grid_exists = True
            if self.rank == 1:
                relative_coords_h = torch.arange(
                    0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype
                )
                relative_coords_h -= seq_length // 2
                relative_coords_h /= seq_length // 2
                relative_coords_table = relative_coords_h
                self.pos_emb = self.cpb_mlp(
                    relative_coords_table.unsqueeze(0).unsqueeze(2)
                )
                self.relative_bias = self.pos_emb
            else:
                seq_length = cube_root(seq_length)
                relative_coords_h = torch.arange(
                    0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype
                )
                relative_coords_w = torch.arange(
                    0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype
                )
                relative_coords_d = torch.arange(
                    0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype
                )
                relative_coords_table = (
                    torch.stack(
                        torch.meshgrid(
                            [relative_coords_h, relative_coords_w, relative_coords_d]
                        )
                    )
                    .contiguous()
                    .unsqueeze(0)
                )
                relative_coords_table -= seq_length // 2
                relative_coords_table /= seq_length // 2
                if not self.conv:
                    self.pos_emb = self.cpb_mlp(
                        relative_coords_table.flatten(2).transpose(1, 2)
                    )
                else:
                    self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2))
                self.relative_bias = self.pos_emb
        input_tensor = input_tensor + self.pos_emb
        return input_tensor


class PosEmbMLPSwinv3D(nn.Module):
    def __init__(
        self,
        window_size,
        pretrained_window_size,
        num_heads,
        seq_length,
        ct_correct=False,
        no_log=False,
    ):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.cpb_mlp = nn.Sequential(
            nn.Linear(3, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )
        relative_coords_h = torch.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32
        )
        relative_coords_w = torch.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32
        )
        relative_coords_d = torch.arange(
            -(self.window_size[2] - 1), self.window_size[2], dtype=torch.float32
        )
        relative_coords_table = (
            torch.stack(
                torch.meshgrid(
                    [relative_coords_h, relative_coords_w, relative_coords_d]
                )
            )
            .permute(1, 2, 3, 0)
            .contiguous()
            .unsqueeze(0)
        )  # 1, 2*Wh-1, 2*Ww-1, 2*Wd-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
            relative_coords_table[:, :, :, 2] /= pretrained_window_size[2] - 1
        else:
            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
            relative_coords_table[:, :, :, 2] /= self.window_size[2] - 1

        if not no_log:
            relative_coords_table *= 8  # normalize to -8, 8
            relative_coords_table = (
                torch.sign(relative_coords_table)
                * torch.log2(torch.abs(relative_coords_table) + 1.0)
                / np.log2(8)
            )

        self.register_buffer("relative_coords_table", relative_coords_table)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_d = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_d]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (
            2 * self.window_size[2] - 1
        )
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.grid_exists = False
        self.pos_emb = None
        self.deploy = False
        relative_bias = torch.zeros(1, num_heads, seq_length, seq_length)
        self.seq_length = seq_length
        self.register_buffer("relative_bias", relative_bias)
        self.ct_correct = ct_correct

    def switch_to_deploy(self):
        self.deploy = True

    def forward(self, input_tensor, local_window_size):
        if self.deploy:
            input_tensor += self.relative_bias
            return input_tensor
        else:
            self.grid_exists = False

        if not self.grid_exists:
            self.grid_exists = True

            num_positions = (
                self.window_size[0] * self.window_size[1] * self.window_size[2]
            )
            relative_position_bias_table = self.cpb_mlp(
                self.relative_coords_table
            ).view(-1, self.num_heads)
            relative_position_bias = relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(num_positions, num_positions, -1)
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)

            # Calculate global feature size difference between input and local window
            n_global_feature = input_tensor.shape[2] - local_window_size
            if n_global_feature > 0 and self.ct_correct:
                # Calculate steps for ct based on 3D window size
                seq_length = cube_root(n_global_feature)
                step_for_ct = self.window_size[0] / (seq_length + 1)  # Step for H, W, D

                indices = []
                for i in range(seq_length):
                    for j in range(seq_length):
                        for k in range(seq_length):
                            # 3D index computation, similar to 2D but with added depth
                            ind = int(
                                (i + 1) * step_for_ct * self.window_size[0]
                                + (j + 1) * step_for_ct * self.window_size[1]
                                + (k + 1) * step_for_ct
                            )
                            indices.append(ind)

                top_part = relative_position_bias[:, indices, :]
                lefttop_part = relative_position_bias[:, indices, :][:, :, indices]
                left_part = relative_position_bias[:, :, indices]

            # Apply 3D padding
            pad_dims = (n_global_feature, 0, n_global_feature, 0)
            relative_position_bias = torch.nn.functional.pad(
                relative_position_bias, pad_dims
            ).contiguous()

            if n_global_feature > 0 and self.ct_correct:
                # Zero out the newly padded bias
                relative_position_bias = relative_position_bias * 0.0
                # Fill in the padded areas with extracted parts from the smaller grid
                relative_position_bias[:, :n_global_feature, :n_global_feature] = (
                    lefttop_part
                )
                relative_position_bias[:, :n_global_feature, n_global_feature:] = (
                    top_part
                )
                relative_position_bias[:, n_global_feature:, :n_global_feature] = (
                    left_part
                )

            self.pos_emb = relative_position_bias.unsqueeze(0)
            self.relative_bias = self.pos_emb

        input_tensor += self.pos_emb
        return input_tensor


class Downsample3D(nn.Module):

    def __init__(
        self,
        dim,
        keep_dim=False,
    ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.norm = LayerNorm3d(dim)
        self.reduction = nn.Sequential(
            nn.Conv3d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm3d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv3d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm3d(dim, eps=1e-4),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class ConvBlock3D(nn.Module):

    def __init__(self, dim, drop_path=0.0, layer_scale=None, kernel_size=3):
        super().__init__()
        """
        Args:
            drop_path: drop path.
            layer_scale: layer scale coefficient.
            kernel_size: kernel size.
        """
        self.conv1 = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm3d(dim, eps=1e-5)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm3d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, global_feature=None):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x, global_feature


class WindowAttention3D(nn.Module):
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
        resolution=0,
        seq_length=0,
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
        self.pos_emb_funct = PosEmbMLPSwinv3D(
            window_size=[resolution, resolution, resolution],
            pretrained_window_size=[resolution, resolution, resolution],
            num_heads=num_heads,
            seq_length=seq_length,
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
        attn = self.pos_emb_funct(attn, self.resolution**3)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HAT3D(nn.Module):
    """
    Hierarchical attention (HAT) based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
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
        sr_ratio=1.0,
        window_size=7,
        last=False,
        layer_scale=None,
        ct_size=1,
        do_propagation=False,
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
            sr_ratio: input to window size ratio.
            window_size: window size.
            last: last layer flag.
            layer_scale: layer scale coefficient.
            ct_size: spatial dimension of carrier token local window.
            do_propagation: enable carrier token propagation.
        """
        # positional encoding for windowed attention tokens
        self.pos_embed = PosEmbMLPSwinv1D(dim, rank=3, seq_length=window_size**3)
        self.norm1 = norm_layer(dim)
        self.square = True if sr_ratio[0] == sr_ratio[1] == sr_ratio[2] else False
        self.do_sr_hat = (
            True
            if ((sr_ratio[0] > 1) or (sr_ratio[1] > 1) or (sr_ratio[2] > 1))
            else False
        )
        # number of carrier tokens per every window
        cr_tokens_per_window = ct_size**3 if self.do_sr_hat else 0
        # total number of carrier tokens
        cr_tokens_total = cr_tokens_per_window * sr_ratio[0] * sr_ratio[1] * sr_ratio[2]
        self.cr_window = ct_size
        self.attn = WindowAttention3D(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            resolution=window_size,
            seq_length=window_size**3 + cr_tokens_per_window,
        )

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

        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma3 = (
            nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        )
        self.gamma4 = (
            nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        )

        self.sr_ratio = sr_ratio
        if self.do_sr_hat:
            # if do hierarchical attention, this part is for carrier tokens
            self.hat_norm1 = norm_layer(dim)
            self.hat_norm2 = norm_layer(dim)
            self.hat_attn = WindowAttention3D(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                resolution=cube_root(cr_tokens_total),
                seq_length=cr_tokens_total,
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
            if self.square:
                self.hat_pos_embed = PosEmbMLPSwinv1D(
                    dim, rank=3, seq_length=cr_tokens_total
                )
            self.gamma1 = (
                nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
            )
            self.gamma2 = (
                nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
            )
            self.upsampler = nn.Upsample(size=window_size, mode="nearest")

        # keep track for the last block to explicitly add carrier tokens to feature maps
        self.last = last
        self.do_propagation = do_propagation

    def forward(self, x, carrier_tokens):
        B, T, N = x.shape
        ct = carrier_tokens
        x = self.pos_embed(x)

        if self.do_sr_hat:
            # do hierarchical attention via carrier tokens
            # first do attention for carrier tokens
            Bg, Ng, Hg = ct.shape

            # ct are located quite differently
            ct = ct_dewindow(
                ct,
                self.cr_window * self.sr_ratio[0],
                self.cr_window * self.sr_ratio[1],
                self.cr_window * self.sr_ratio[2],
                self.cr_window,
            )

            # positional bias for carrier tokens
            ct = self.hat_pos_embed(ct)

            # attention plus mlp
            ct = ct + self.hat_drop_path(
                self.gamma1 * self.hat_attn(self.hat_norm1(ct))
            )
            ct = ct + self.hat_drop_path(self.gamma2 * self.hat_mlp(self.hat_norm2(ct)))

            # ct are put back to windows
            ct = ct_window(
                ct,
                self.cr_window * self.sr_ratio[0],
                self.cr_window * self.sr_ratio[1],
                self.cr_window * self.sr_ratio[2],
                self.cr_window,
            )

            ct = ct.reshape(x.shape[0], -1, N)
            # concatenate carrier_tokens to the windowed tokens
            x = torch.cat((ct, x), dim=1)

        # window attention together with carrier tokens
        x = x + self.drop_path(self.gamma3 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma4 * self.mlp(self.norm2(x)))

        if self.do_sr_hat:
            # for hierarchical attention we need to split carrier tokens and window tokens back
            ctr, x = x.split(
                [
                    x.shape[1] - self.window_size * self.window_size * self.window_size,
                    self.window_size * self.window_size * self.window_size,
                ],
                dim=1,
            )
            ct = ctr.reshape(Bg, Ng, Hg)  # reshape carrier tokens.
            if self.last and self.do_propagation:
                # propagate carrier token information into the image
                ctr_image_space = ctr.transpose(1, 2).reshape(
                    B, N, self.cr_window, self.cr_window, self.cr_window
                )
                x = x + self.gamma1 * self.upsampler(
                    ctr_image_space.to(dtype=torch.float32)
                ).flatten(2).transpose(1, 2).to(dtype=x.dtype)
        return x, ct

class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embedding's dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None, norm_type="layer_norm", bias=True):
        super().__init__()
        if num_embeddings is not None:
            self.emb_impl = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        else:
            self.emb_impl = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        elif norm_type == "fp32_layer_norm":
            self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb_impl is not None:
            emb = self.emb_impl(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate = emb.chunk(6, dim=1)
        return msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate

class HAT3DCond(nn.Module):
    """
    Hierarchical attention (HAT) based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
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
        sr_ratio=1.0,
        window_size=7,
        last=False,
        layer_scale=None,
        ct_size=1,
        do_propagation=False,
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
        self.pos_embed = PosEmbMLPSwinv1D(dim, rank=3, seq_length=window_size**3)
        self.norm1 = norm_layer(dim)
        self.square = True if sr_ratio[0] == sr_ratio[1] == sr_ratio[2] else False
        self.do_sr_hat = (
            True
            if ((sr_ratio[0] > 1) or (sr_ratio[1] > 1) or (sr_ratio[2] > 1))
            else False
        )
        # number of carrier tokens per every window
        cr_tokens_per_window = ct_size**3 if self.do_sr_hat else 0
        # total number of carrier tokens
        cr_tokens_total = cr_tokens_per_window * sr_ratio[0] * sr_ratio[1] * sr_ratio[2]
        self.cr_window = ct_size
        self.attn = WindowAttention3D(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            resolution=window_size,
            seq_length=window_size**3 + cr_tokens_per_window,
        )

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

        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]

        # self.gamma3 = (
        #     nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        # )
        # self.gamma4 = (
        #     nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        # )

        self.adain_2 = AdaLayerNormZero(dim, num_embeddings=num_embeds_ada_norm, norm_type="layer_norm")

        self.sr_ratio = sr_ratio
        if self.do_sr_hat:
            # if do hierarchical attention, this part is for carrier tokens
            self.hat_norm1 = norm_layer(dim)
            self.hat_norm2 = norm_layer(dim)
            self.hat_attn = WindowAttention3D(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                resolution=cube_root(cr_tokens_total),
                seq_length=cr_tokens_total,
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
            if self.square:
                self.hat_pos_embed = PosEmbMLPSwinv1D(
                    dim, rank=3, seq_length=cr_tokens_total
                )

            self.adain_1 = AdaLayerNormZero(dim, num_embeddings=num_embeds_ada_norm, norm_type="layer_norm")

            # self.gamma1 = (
            #     nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
            # )
            # self.gamma2 = (
            #     nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
            # )
            self.upsampler = nn.Upsample(size=window_size, mode="nearest")

        # keep track for the last block to explicitly add carrier tokens to feature maps
        self.last = last
        self.do_propagation = do_propagation

    def forward(self, x, carrier_tokens,
                timestep: Optional[torch.LongTensor] = None,
                class_labels: Optional[torch.LongTensor] = None):

        B, T, N = x.shape
        ct = carrier_tokens
        x = self.pos_embed(x)

        if self.do_sr_hat:
            # do hierarchical attention via carrier tokens
            # first do attention for carrier tokens
            Bg, Ng, Hg = ct.shape

            # ct are located quite differently
            ct = ct_dewindow(
                ct,
                self.cr_window * self.sr_ratio[0],
                self.cr_window * self.sr_ratio[1],
                self.cr_window * self.sr_ratio[2],
                self.cr_window,
            )

            # positional bias for carrier tokens
            ct = self.hat_pos_embed(ct)


            ######## DiT block with MSA, MLP, and AdaIN ########
            msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate = self.adain_1(timestep=timestep, class_labels=class_labels)
            ct_msa = self.hat_norm1(ct)
            ct_msa = ct_msa * (1 + msa_scale[:, None]) + msa_shift[:, None]
            # attention plus mlp
            ct_msa = self.hat_attn(ct_msa)
            ct_msa = ct_msa * (1 + msa_gate[:, None])
            ct = ct + self.hat_drop_path(ct_msa)

            ct_mlp = self.hat_norm2(ct)
            ct_mlp = ct_mlp * (1 + mlp_scale[:, None]) + mlp_shift[:, None]
            ct_mlp = self.hat_mlp(ct_mlp)
            ct_mlp = ct_mlp * (1 + mlp_gate[:, None])

            ct = ct + self.hat_drop_path(ct_mlp)
            ###################################################

            # ct are put back to windows
            ct = ct_window(
                ct,
                self.cr_window * self.sr_ratio[0],
                self.cr_window * self.sr_ratio[1],
                self.cr_window * self.sr_ratio[2],
                self.cr_window,
            )

            ct = ct.reshape(x.shape[0], -1, N)

            # concatenate carrier_tokens to the windowed tokens
            x = torch.cat((ct, x), dim=1)


        ########### DiT block with MSA, MLP, and AdaIN ############
        msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate = self.adain_2(timestep=timestep, class_labels=class_labels)
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
        ##########################################################


        if self.do_sr_hat:
            # for hierarchical attention we need to split carrier tokens and window tokens back
            ctr, x = x.split(
                [
                    x.shape[1] - self.window_size * self.window_size * self.window_size,
                    self.window_size * self.window_size * self.window_size,
                ],
                dim=1,
            )
            ct = ctr.reshape(Bg, Ng, Hg)  # reshape carrier tokens.
            if self.last and self.do_propagation:
                # propagate carrier token information into the image
                ctr_image_space = ctr.transpose(1, 2).reshape(
                    B, N, self.cr_window, self.cr_window, self.cr_window
                )
                x = x + self.gamma1 * self.upsampler(
                    ctr_image_space.to(dtype=torch.float32)
                ).flatten(2).transpose(1, 2).to(dtype=x.dtype)
        return x, ct

class TokenInitializer3D(nn.Module):
    """
    Carrier token Initializer based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(self, dim, input_resolution, window_size, ct_size=1):
        """
        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            window_size: window size.
            ct_size: spatial dimension of carrier token local window
        """
        super().__init__()
        output_size1 = int((ct_size) * input_resolution[0] / window_size)
        stride_size1 = int(input_resolution[0] / output_size1)
        kernel_size1 = input_resolution[0] - (output_size1 - 1) * stride_size1
        output_size2 = int((ct_size) * input_resolution[1] / window_size)
        stride_size2 = int(input_resolution[1] / output_size2)
        kernel_size2 = input_resolution[1] - (output_size2 - 1) * stride_size2
        output_size3 = int((ct_size) * input_resolution[2] / window_size)
        stride_size3 = int(input_resolution[2] / output_size3)
        kernel_size3 = input_resolution[2] - (output_size3 - 1) * stride_size3

        self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        to_global_feature = nn.Sequential()
        to_global_feature.add_module("pos", self.pos_embed)
        to_global_feature.add_module(
            "pool",
            nn.AvgPool3d(
                kernel_size=(kernel_size1, kernel_size2, kernel_size3),
                stride=(stride_size1, stride_size2, stride_size3),
            ),
        )
        self.to_global_feature = to_global_feature
        self.window_size = ct_size

    def forward(self, x):
        x = self.to_global_feature(x)
        B, C, H, W, D = x.shape
        ct = x.view(
            B,
            C,
            H // self.window_size,
            self.window_size,
            W // self.window_size,
            self.window_size,
            D // self.window_size,
            self.window_size,
        )
        ct = ct.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(-1, H * W * D, C)
        return ct

class FasterDiTLayer3D(nn.Module):
    """
    FasterDiT layer based on: "Hatamizadeh et al.,
    FasterDiT: Fast Vision Transformers with Hierarchical Attention"
    """

    def __init__(
        self,
        dim,
        depth,
        input_resolution,
        num_heads,
        window_size,
        ct_size=1,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        layer_scale_conv=None,
        only_local=False,
        hierarchy=True,
        do_propagation=False,
    ):
        """
        Args:
            dim: feature size dimension.
            depth: layer depth.
            input_resolution: input resolution.
            num_heads: number of attention head.
            window_size: window size.
            ct_size: spatial dimension of carrier token local window.
            conv: conv_based stage flag.
            downsample: downsample flag.
            mlp_ratio: MLP ratio.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            layer_scale: layer scale coefficient.
            layer_scale_conv: conv layer scale coefficient.
            only_local: local attention flag.
            hierarchy: hierarchical attention flag.
            do_propagation: enable carrier token propagation.
        """
        super().__init__()
        self.transformer_block = False

        H = (
            input_resolution[0]
            + (window_size - input_resolution[0] % window_size) % window_size
        )
        W = (
            input_resolution[1]
            + (window_size - input_resolution[1] % window_size) % window_size
        )
        D = (
            input_resolution[2]
            + (window_size - input_resolution[2] % window_size) % window_size
        )
        input_resolution = [H, W, D]

        sr_ratio = [
            input_resolution[0] // window_size if not only_local else 1,
            input_resolution[1] // window_size if not only_local else 1,
            input_resolution[2] // window_size if not only_local else 1,
        ]
        self.blocks = nn.ModuleList(
            [
                HAT3DCond(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    sr_ratio=sr_ratio,
                    window_size=window_size,
                    last=(i == depth - 1),
                    layer_scale=layer_scale,
                    ct_size=ct_size,
                    do_propagation=do_propagation,
                )
                for i in range(depth)
            ]
        )

        self.transformer_block = True
        self.downsample = None if not downsample else Downsample3D(dim=dim)

        if (
            len(self.blocks)
            and not only_local
            and sr_ratio
            and hierarchy
        ):
            self.global_tokenizer = TokenInitializer3D(
                dim, input_resolution, window_size, ct_size=ct_size
            )
            self.do_gt = True
        else:
            self.do_gt = False

        self.window_size = window_size

    def forward(self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,):

        B, C, H, W, D = hidden_states.shape

        if self.transformer_block:
            pad_d = (self.window_size - D % self.window_size) % self.window_size
            pad_h = (self.window_size - H % self.window_size) % self.window_size
            pad_w = (self.window_size - W % self.window_size) % self.window_size
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                hidden_states = torch.nn.functional.pad(
                    hidden_states, (0, pad_d, 0, pad_w, 0, pad_h)
                )  # Pad in 3D (W, H, D)
                _, _, Hp, Wp, Dp = hidden_states.shape
            else:
                Hp, Wp, Dp = H, W, D

        ct = self.global_tokenizer(hidden_states) if self.do_gt else None

        if self.transformer_block:
            hidden_states = window_partition(hidden_states, self.window_size)
        for bn, blk in enumerate(self.blocks):
            hidden_states, ct = blk(hidden_states, ct, timestep=timestep, class_labels=class_labels)

        if self.transformer_block:
            hidden_states = window_reverse(hidden_states, self.window_size, Hp, Wp, Dp, B)
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                hidden_states = hidden_states[:, :, :H, :W, :D].contiguous()

        if self.downsample is None:
            return hidden_states, hidden_states

        return self.downsample(hidden_states), hidden_states

class FasterViTLayer3D(nn.Module):
    """
    FasterViT layer based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention"
    """

    def __init__(
        self,
        dim,
        depth,
        input_resolution,
        num_heads,
        window_size,
        ct_size=1,
        conv=False,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        layer_scale_conv=None,
        only_local=False,
        hierarchy=True,
        do_propagation=False,
    ):
        """
        Args:
            dim: feature size dimension.
            depth: layer depth.
            input_resolution: input resolution.
            num_heads: number of attention head.
            window_size: window size.
            ct_size: spatial dimension of carrier token local window.
            conv: conv_based stage flag.
            downsample: downsample flag.
            mlp_ratio: MLP ratio.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            layer_scale: layer scale coefficient.
            layer_scale_conv: conv layer scale coefficient.
            only_local: local attention flag.
            hierarchy: hierarchical attention flag.
            do_propagation: enable carrier token propagation.
        """
        super().__init__()
        self.conv = conv
        self.transformer_block = False

        H = (
            input_resolution[0]
            + (window_size - input_resolution[0] % window_size) % window_size
        )
        W = (
            input_resolution[1]
            + (window_size - input_resolution[1] % window_size) % window_size
        )
        D = (
            input_resolution[2]
            + (window_size - input_resolution[2] % window_size) % window_size
        )
        input_resolution = [H, W, D]

        if conv:
            self.blocks = nn.ModuleList(
                [
                    ConvBlock3D(
                        dim=dim,
                        drop_path=(
                            drop_path[i] if isinstance(drop_path, list) else drop_path
                        ),
                        layer_scale=layer_scale_conv,
                    )
                    for i in range(depth)
                ]
            )
            self.transformer_block = False
        else:
            sr_ratio = [
                input_resolution[0] // window_size if not only_local else 1,
                input_resolution[1] // window_size if not only_local else 1,
                input_resolution[2] // window_size if not only_local else 1,
            ]
            self.blocks = nn.ModuleList(
                [
                    HAT3D(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=(
                            drop_path[i] if isinstance(drop_path, list) else drop_path
                        ),
                        sr_ratio=sr_ratio,
                        window_size=window_size,
                        last=(i == depth - 1),
                        layer_scale=layer_scale,
                        ct_size=ct_size,
                        do_propagation=do_propagation,
                    )
                    for i in range(depth)
                ]
            )
            self.transformer_block = True
        self.downsample = None if not downsample else Downsample3D(dim=dim)
        if (
            len(self.blocks)
            and not only_local
            and sr_ratio
            and hierarchy
            and not self.conv
        ):
            self.global_tokenizer = TokenInitializer3D(
                dim, input_resolution, window_size, ct_size=ct_size
            )
            self.do_gt = True
        else:
            self.do_gt = False

        self.window_size = window_size

    def forward(self, x):
        B, C, H, W, D = x.shape
        if self.transformer_block:
            pad_d = (self.window_size - D % self.window_size) % self.window_size
            pad_h = (self.window_size - H % self.window_size) % self.window_size
            pad_w = (self.window_size - W % self.window_size) % self.window_size
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                x = torch.nn.functional.pad(
                    x, (0, pad_d, 0, pad_w, 0, pad_h)
                )  # Pad in 3D (W, H, D)
                _, _, Hp, Wp, Dp = x.shape
            else:
                Hp, Wp, Dp = H, W, D
        ct = self.global_tokenizer(x) if self.do_gt else None
        if self.transformer_block:
            x = window_partition(x, self.window_size)
        for bn, blk in enumerate(self.blocks):
            x, ct = blk(x, ct)
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp, Dp, B)
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                x = x[:, :, :H, :W, :D].contiguous()
        if self.downsample is None:
            return x, x
        return self.downsample(x), x

@dataclass
class FasterViT3DOutput(BaseOutput):
    """
    The output of [`FasterViT3D`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width, depth)`
    """

    sample: "torch.Tensor"  # noqa: F821

@dataclass
class WrapperModel3DOutput(BaseOutput):
    """
    The output of [`WrapperModel3D`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width, depth)`
    """

    sample: "torch.Tensor"  # noqa: F821

DEFAULT_DTYPE = torch.float32
@torch.jit.script
def get_timestep_embedding(timesteps, embed_dim: int, dtype: torch.dtype = DEFAULT_DTYPE):
    """
    Adapted from fairseq/fairseq/modules/sinusoidal_positional_embedding.py
    The implementation is slightly different from the decription in Section 3.5 of [1]
    [1] Vaswani, Ashish, et al. "Attention is all you need."
     Advances in neural information processing systems 30 (2017).
    """
    half_dim = embed_dim // 2
    embed = math.log(10000) / (half_dim - 1)
    embed = torch.exp(-torch.arange(half_dim, dtype=dtype, device=timesteps.device) * embed)
    embed = torch.outer(timesteps.ravel().to(dtype), embed)
    embed = torch.cat([torch.sin(embed), torch.cos(embed)], dim=1)
    if embed_dim % 2 == 1:
        embed = F.pad(embed, [0, 1])  # padding the last dimension
    assert embed.dtype == dtype
    return embed

class WrapperModel3D(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self, model: Dict,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 sample_size: int = 320,
                 feature_embedding_dim: int = 64,
                 num_downsampling_layers: int = 3,
                 time_embedding_dim: int = 64,
                 num_groups: int = 32):
        super().__init__()

        self.sample_size = sample_size
        self.model_impl = instantiate_from_config(model)

        self.feature_embedding_dim = feature_embedding_dim
        self.time_embedding_dim = time_embedding_dim

        self.encoder = ConditionedEncoder3D(
            in_channels=in_channels,
            feature_embedding_dim=feature_embedding_dim,
            num_downsampling_layers=num_downsampling_layers,
            embedding_dim=time_embedding_dim,
            num_groups=num_groups,
        )

        self.decoder = ConditionedDecoder3D(
            out_channels=out_channels,
            feature_embedding_dim=feature_embedding_dim,
            num_upsampling_layers=num_downsampling_layers,
            embedding_dim=time_embedding_dim,
            features_first_layer=self.model_impl.num_features_out,
            num_groups=num_groups,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: Optional[torch.LongTensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            return_dict: bool = True,
    ):
        """
        Args:
            hidden_states: input tensor of shape `(batch_size, num_channels, height, width, depth)`
            timestep: timestep tensor of shape `(batch_size, 1)`
            class_labels: class label tensor of shape `(batch_size, 1)`
            cross_attention_kwargs: cross attention kwargs
            return_dict: return dict flag
        """

        if timestep is None:
            timestep = torch.Tensor([0]).to(hidden_states.device)

        if len(timestep.shape) == 0:
            timestep = timestep.unsqueeze(0)
            timestep = timestep.repeat(hidden_states.shape[0])
            timestep = timestep.to(hidden_states.device)

        if class_labels is None:
            class_labels = torch.Tensor([0]).to(hidden_states.device)

        if len(class_labels.shape) == 0:
            class_labels = class_labels.unsqueeze(0)
            class_labels = class_labels.repeat(hidden_states.shape[0])
            class_labels = class_labels.to(hidden_states.device)

        timestep_embedding = get_timestep_embedding(timestep, self.time_embedding_dim)
        residuals = self.encoder(hidden_states, timestep_embedding)
        state = residuals[-1]

        state, state_dash = self.model_impl(state, timestep, class_labels)

        out = self.decoder(state_dash, timestep_embedding, residuals)

        return FasterViT3DOutput(sample=out)

class FasterDiT3D(ModelMixin, ConfigMixin):
    """
    FasterDiT based on: "Hatamizadeh et al., and Peebles et al.
    FasterDiT: Fast Vision Transformers with Hierarchical Attention + DiT conditioning
    """

    @register_to_config
    def __init__(
        self,
        dim,
        depths,
        window_size,
        ct_size,
        mlp_ratio,
        num_heads,
        resolution=224,
        drop_path_rate=0.2,
        num_classes=1000,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        layer_scale=None,
        layer_scale_conv=None,
        layer_norm_last=True,
        hat=None,
        do_propagation=False,
        only_local=False,
        **kwargs,
    ):
        """
        Args:
            dim: feature size dimension.
            in_dim: inner-plane feature size dimension.
            depths: layer depth.
            window_size: window size.
            ct_size: spatial dimension of carrier token local window.
            mlp_ratio: MLP ratio.
            num_heads: number of attention head.
            resolution: image resolution.
            drop_path_rate: drop path rate.
            in_chans: input channel dimension.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            layer_scale: layer scale coefficient.
            layer_scale_conv: conv layer scale coefficient.
            layer_norm_last: last stage layer norm flag.
            hat: hierarchical attention flag.
            do_propagation: enable carrier token propagation.
        """
        super().__init__()
        if type(resolution) != tuple and type(resolution) != list:
            resolution = [resolution, resolution, resolution]

        num_features = int(dim * 2 ** (len(depths) - 1))
        num_features_dash = 0
        for i in range(len(depths)):
            num_features_dash += int(dim / (4 ** i))

        self.num_features_out = num_features_dash

        self.num_classes = num_classes

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.levels = nn.ModuleList()
        self.upsample = nn.ModuleList()

        if hat is None:
            hat = [
                not only_local,
            ] * len(depths)

        for i in range(len(depths)):
            level = FasterDiTLayer3D(
                dim=int(dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                ct_size=ct_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])], # noqa
                downsample= (i != len(depths) - 1),
                layer_scale=layer_scale,
                layer_scale_conv=layer_scale_conv,
                input_resolution=[
                    int(2 ** (-i) * resolution[0]),
                    int(2 ** (-i) * resolution[1]),
                    int(2 ** (-i) * resolution[2]),
                ],
                only_local=not hat[i],
                do_propagation=do_propagation,
            )

            upsample = PixelShuffle3d(upscale_factor=2)

            self.levels.append(level)
            self.upsample.append(upsample)

        # self.norm = (
        #     LayerNorm3d(num_features)
        #     if layer_norm_last
        #     else nn.BatchNorm3d(num_features)
        # )

        self.norm_dash = (
            LayerNorm3d(num_features_dash)
            if layer_norm_last
            else nn.BatchNorm3d(num_features_dash)
        )

        self.avgpool = nn.AdaptiveAvgPool3d(1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"rpb"}

    def forward(self, x: torch.Tensor,
                timestep: Optional[torch.LongTensor] = None,
                class_labels: Optional[torch.LongTensor] = None):
        outs = []
        for level in self.levels:
            x, xo = level(x, timestep, class_labels)
            outs.append(xo.contiguous())

        x_dash = self.upsample[0](x)
        for i in range(len(outs)-1):
            y = outs[-i-2]
            x_dash = torch.concatenate([x_dash, y], dim=1)
            if i < len(outs) - 2:
                x_dash = self.upsample[i+1](x_dash)

        # x = self.norm(x)
        x_dash = self.norm_dash(x_dash)

        return x, x_dash

    def _load_state_dict(self, pretrained, strict: bool = False):
        _load_checkpoint(self, pretrained, strict=strict)

class FasterViT3D(ModelMixin, ConfigMixin):
    """
    FasterViT based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    @register_to_config
    def __init__(
        self,
        dim,
        in_dim,
        depths,
        window_size,
        ct_size,
        mlp_ratio,
        num_heads,
        resolution=224,
        drop_path_rate=0.2,
        in_chans=3,
        num_classes=1000,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        layer_scale=None,
        layer_scale_conv=None,
        layer_norm_last=False,
        hat=[False, False, True, False],
        do_propagation=False,
        head_type="transposed_conv",
        head_attention=False,
        head_num_heads=6,
        **kwargs,
    ):
        """
        Args:
            dim: feature size dimension.
            in_dim: inner-plane feature size dimension.
            depths: layer depth.
            window_size: window size.
            ct_size: spatial dimension of carrier token local window.
            mlp_ratio: MLP ratio.
            num_heads: number of attention head.
            resolution: image resolution.
            drop_path_rate: drop path rate.
            in_chans: input channel dimension.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            layer_scale: layer scale coefficient.
            layer_scale_conv: conv layer scale coefficient.
            layer_norm_last: last stage layer norm flag.
            hat: hierarchical attention flag.
            do_propagation: enable carrier token propagation.
        """
        super().__init__()
        if type(resolution) != tuple and type(resolution) != list:
            resolution = [resolution, resolution, resolution]
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed3D(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        if hat is None:
            hat = [
                True,
            ] * len(depths)
        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False
            level = FasterViTLayer3D(
                dim=int(dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                ct_size=ct_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                conv=conv,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                downsample=(i < 3),
                layer_scale=layer_scale,
                layer_scale_conv=layer_scale_conv,
                input_resolution=[
                    int(2 ** (-2 - i) * resolution[0]),
                    int(2 ** (-2 - i) * resolution[1]),
                    int(2 ** (-2 - i) * resolution[2]),
                ],
                only_local=not hat[i],
                do_propagation=do_propagation,
            )
            self.levels.append(level)
        self.norm = (
            LayerNorm3d(num_features)
            if layer_norm_last
            else nn.BatchNorm3d(num_features)
        )
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        upsample_scales = [2, 2, 2, 2, 2]
        out_features = [20, 16, 12, 8, 4]

        if head_type == "transposed_conv":
            self.head = UNetTransposedConvHead(
                in_features=num_features,
                out_features=out_features,
                upsample_scales=upsample_scales,
                resolution=resolution,
                final_channels=3,
                with_attention=head_attention,
                num_heads=head_num_heads,
            )
        elif head_type == "pixelshuffle":
            self.head = PixelShuffleHead(num_features, num_channels=in_chans)
        else:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(1),
                (
                    nn.Linear(num_features, num_classes)
                    if num_classes > 0
                    else nn.Identity()
                ),
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"rpb"}

    def forward_features(self, x):
        x = self.patch_embed(x)
        outs = []
        for level in self.levels:
            x, xo = level(x)
            outs.append(xo.contiguous())
        x = self.norm(x)
        return x

    def forward_head(self, x):
        x = self.head(x)
        return x

    def forward(self, x, **kwargs):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return FasterViT3DOutput(sample=x)

    def _load_state_dict(self, pretrained, strict: bool = False):
        _load_checkpoint(self, pretrained, strict=strict)


class UNetTransposedConvHead(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        upsample_scales,
        resolution,
        final_channels=3,
        with_attention=False,
        num_heads=0,
    ):
        super().__init__()
        self.upsample_layers = nn.ModuleList()
        self.activation = nn.ReLU()
        self.current_resolution = resolution

        # Ensure out_features matches the number of upsampling scales
        assert len(out_features) == len(
            upsample_scales
        ), "out_features and upsample_scales must have the same length"

        self.with_attention = with_attention
        if with_attention:
            self.attention = nn.MultiheadAttention(in_features, num_heads)

        # Create multiple upsampling layers with activation functions
        for i, scale in enumerate(upsample_scales):
            self.upsample_layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        in_features,
                        out_features[i],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm3d(out_features[i]),
                    self.activation,
                )
            )
            in_features = out_features[i]
            self.current_resolution = self.current_resolution * 2  # Update resolution

        # Final output layer to match the required number of channels (e.g., 3 for RGB or segmentation masks)
        self.final_layer = nn.Conv3d(out_features[-1], final_channels, kernel_size=1)

    def forward(self, x):
        if self.with_attention:
            # Flatten and apply attention
            B, C, H, W, D = x.shape
            x = x.flatten(2)  # [B, C, HWD]
            x = x.permute(2, 0, 1)  # [HWD, B, C]

            # Attention mechanism
            x, _ = self.attention(x, x, x)

            # Reshape back to [B, C, H, W, D]
            x = x.permute(1, 2, 0).view(B, C, H, W, D)

        for layer in self.upsample_layers:
            x = layer(x)

        x = self.final_layer(x)
        return x

class ConditionedEncoder3DBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 num_groups: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.gn_1 = nn.GroupNorm(num_groups, in_channels)
        self.activation_1 = nn.GELU()
        self.conv_1 = nn.Conv3d(in_channels, in_channels, 3, 1, 1)

        self.mlp_scale_bias = nn.Linear(embed_dim, 2 * in_channels)
        self.gn_2 = nn.GroupNorm(num_groups, in_channels)
        self.activation_2 = nn.GELU()
        self.conv_2 = nn.Conv3d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x, embedding):

        scale_and_shift = self.mlp_scale_bias(embedding)
        scale, shift = scale_and_shift.chunk(2, dim=-1)

        x_res = x

        x = self.gn_1(x)
        x = self.activation_1(x)
        x = self.conv_1(x)
        x = self.gn_2(x)
        x = x * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        x = self.activation_2(x)
        x = self.conv_2(x)

        x = x + x_res

        return x

class ConditionedEncoder3D(nn.Module):

    def __init__(self,
                 in_channels: int,
                 feature_embedding_dim: int,
                 num_downsampling_layers: int,
                 embedding_dim: int,
                 num_groups: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.feature_embedding_dim = feature_embedding_dim
        self.num_downsampling_layers = num_downsampling_layers
        self.embedding_dim = embedding_dim

        self.feature_embed = nn.Conv3d(in_channels, feature_embedding_dim, 3, 1, 1)
        self.downsampling_layers = nn.ModuleList()
        for i in range(num_downsampling_layers):
            self.downsampling_layers.append(
                nn.Conv3d(feature_embedding_dim * 2 ** i, feature_embedding_dim * 2 ** (i+1), 3, 2, 1)
            )
        self.blocks = nn.ModuleList()
        for i in range(num_downsampling_layers - 1):
            self.blocks.append(
                ConditionedEncoder3DBlock(feature_embedding_dim * 2 ** (i+1), embedding_dim,
                                          num_groups=num_groups)
            )


    def forward(self, x, embedding):

        x = self.feature_embed(x)

        res_list = [x]

        x = self.downsampling_layers[0](x)

        for i in range(self.num_downsampling_layers - 1):
            x = self.blocks[i](x, embedding)
            res_list.append(x)
            x = self.downsampling_layers[i+1](x)

        res_list.append(x)

        return res_list

ConditionedDecoder3DBlock = ConditionedEncoder3DBlock

class DecoderUpsamplingBlock(nn.Module):

        def __init__(self,
                    in_channels: int,
                    out_channels: int):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels

            self.linear_conv = nn.Conv3d(in_channels, out_channels * 4, 1)
            self.shuffle = PixelShuffle3d(2)

        def forward(self, x):
            x = self.linear_conv(x)
            x = self.shuffle(x)
            return x

class ConditionedDecoder3D(nn.Module):

        def __init__(self,
                    out_channels: int,
                    feature_embedding_dim: int,
                    num_upsampling_layers: int,
                    embedding_dim: int,
                    features_first_layer: int = None,
                    num_groups: int = 32):
            super().__init__()
            self.out_channels = out_channels
            self.feature_embedding_dim = feature_embedding_dim
            self.num_upsampling_layers = num_upsampling_layers
            self.embedding_dim = embedding_dim

            self.decompress = nn.Conv3d(feature_embedding_dim, out_channels, 3, 1, 1)

            self.blocks = nn.ModuleList()
            for i in range(num_upsampling_layers - 1):
                self.blocks.append(
                    ConditionedDecoder3DBlock(feature_embedding_dim * 2 ** (num_upsampling_layers - i - 1), embedding_dim,
                                              num_groups=num_groups)
                )


            if features_first_layer is None:
                features_first_layer = feature_embedding_dim

            self.upsampling_layers = nn.ModuleList()

            local_feature_dim = feature_embedding_dim * 2 ** num_upsampling_layers
            self.upsampling_layers.append(
                DecoderUpsamplingBlock(features_first_layer, local_feature_dim)
            )
            for i in range(num_upsampling_layers-1):
                local_feature_dim = feature_embedding_dim * 2 ** (num_upsampling_layers - i - 1)
                self.upsampling_layers.append(
                    DecoderUpsamplingBlock(local_feature_dim, local_feature_dim)
                )

        def forward(self, x, embedding, encoder_outputs):

            x = self.upsampling_layers[0](x)
            x += encoder_outputs[::-1][1]

            for i in range(self.num_upsampling_layers - 1):
                x = self.blocks[i](x, embedding)
                x = self.upsampling_layers[i+1](x)
                x += encoder_outputs[::-1][i+2]

            x = self.decompress(x)

            return x

class PixelShuffleHead(nn.Module):
    def __init__(
        self,
        num_features,
        num_channels=3,
        encoder_stride=2,
        use_single_shuffle=False,
    ):
        super().__init__()

        if use_single_shuffle:
            encoder_stride = 32
            self.decoder = nn.Sequential(
                nn.Conv3d(num_features, encoder_stride**3 * num_channels, 1),
                PixelShuffle3d(encoder_stride),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Conv3d(num_features, encoder_stride**3 * num_channels, 1),
                PixelShuffle3d(encoder_stride),
                nn.Conv3d(num_channels, encoder_stride**3 * num_channels, 1),
                PixelShuffle3d(encoder_stride),
                nn.Conv3d(num_channels, encoder_stride**3 * num_channels, 1),
                PixelShuffle3d(encoder_stride),
                nn.Conv3d(num_channels, encoder_stride**3 * num_channels, 1),
                PixelShuffle3d(encoder_stride),
                nn.Conv3d(num_channels, encoder_stride**3 * num_channels, 1),
                PixelShuffle3d(encoder_stride),
            )

    def forward(self, x):
        out = self.decoder(x)
        return out