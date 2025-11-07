from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Sequence, Union

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings, get_timestep_embedding, Timesteps, \
    TimestepEmbedding, LabelEmbedding
from diffusers.utils import BaseOutput
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath

from src.core.models.box.output3d import Output3D
from src.core.models.box.pixelshuffle import PixelShuffle3d
from src.core.models.box.unet3d import ConditionedEncoder3D, ConditionedDecoder3D
from src.core.models.fasterdit import TimestepEmbedder, LabelEmbedder
from src.utils import instantiate_from_config

def cube_root(n):
    # Convert n to a PyTorch tensor for high-precision calculations
    n_tensor = torch.tensor(n, dtype=torch.float64)
    result = torch.pow(n_tensor, 1 / 3)
    return int(torch.round(result).item())


def window_partition(x, window_size):
    B, H, W, D, C = x.shape
    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        D // window_size,
        window_size,
        C
    )
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(
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
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, H, W, D, -1)
    return x

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


class PosEmbMLPSwinv3D(nn.Module):
    def __init__(
        self,
        window_size,
        pretrained_window_size,
        num_heads,
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
        seq_length = self.window_size[0] * self.window_size[1] * self.window_size[2]
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

            self.pos_emb = relative_position_bias.unsqueeze(0)
            self.relative_bias = self.pos_emb

        input_tensor += self.pos_emb
        return input_tensor

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


@dataclass
class WrapperModel3DOutput(BaseOutput):
    """
    The output of [`WrapperModel3D`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width, depth)`
    """

    sample: "torch.Tensor"  # noqa: F821

class PlaceholderModel(nn.Module):

    def __init__(self, dim: int, *args, **kwargs):
        super().__init__()
        self.args = args
        self.latent_size = dim
        self.kwargs = kwargs

    def encode(self, x, t, *args, **kwargs):
        return x, [t], [x]

    def decode(self, x, *args, **kwargs):
        return x

    def forward(self, x, *args, **kwargs):
        return x

def Wrapper3D_S(channel_size: int,
              channel_size_out: Optional[int] = None,
              drop_class_labels: bool = False,
              partition_size: Optional[Union[Tuple[int, int, int], int]] = 1,
              mending: bool = False,
              ):

    if channel_size_out is None:
        channel_size_out = channel_size

    model = {
        'target': 'src.core.models.box.pdedit.PlaceholderModel',
        'params': {
            'dim': 64,
        }
    }

    return WrapperModel3D(model, in_channels=channel_size, out_channels=channel_size_out,
                          feature_embedding_dim=[32, 32, 64], num_downsampling_layers=2,
                          time_embedding_dim=64, num_groups=16, drop_class_labels=drop_class_labels,
                          partition_size=partition_size, mending=mending)


def Wrapper3D_B(channel_size: int,
              channel_size_out: Optional[int] = None,
              drop_class_labels: bool = False,
              partition_size: Optional[Union[Tuple[int, int, int], int]] = 1,
              mending: bool = False,
               ):

    if channel_size_out is None:
        channel_size_out = channel_size

    model = {
        'target': 'src.core.models.box.pdedit.PlaceholderModel',
        'params': {
            'dim': 128,
        }
    }

    return WrapperModel3D(model, in_channels=channel_size, out_channels=channel_size_out,
                          feature_embedding_dim=[64, 128, 128], num_downsampling_layers=2,
                          time_embedding_dim=64, num_groups=32, drop_class_labels=drop_class_labels,
                          partition_size=partition_size, mending=mending)


def Wrapper3D_L(channel_size: int,
              channel_size_out: Optional[int] = None,
              drop_class_labels: bool = False,
              partition_size: Optional[Union[Tuple[int, int, int], int]] = 1,
              mending: bool = False,
                ):

    if channel_size_out is None:
        channel_size_out = channel_size

    model = {
        'target': 'src.core.models.box.pdedit.PlaceholderModel',
        'params': {
            'dim': 256,
        }
    }

    return WrapperModel3D(model, in_channels=channel_size, out_channels=channel_size_out,
                          feature_embedding_dim=[128, 256, 256], num_downsampling_layers=2,
                          time_embedding_dim=64, num_groups=64, drop_class_labels=drop_class_labels,
                          partition_size=partition_size, mending=mending)


def PDEDiT3D_S(channel_size: int,
               channel_size_out: Optional[int] = None,
               drop_class_labels: bool = False,
               partition_size: Optional[Union[Tuple[int, int, int], int]] = 1,
               mending: bool = False,
               **kwargs):

    if channel_size_out is None:
        channel_size_out = channel_size

    model = {
        'target': 'src.core.models.box.PDEDiT3DBackbone',
        'params': {
            'hidden_size': 64,
            'window_size': [4, 4, 4, 4, 4],
            'num_heads': [4, 4, 4, 4, 4],
            'depth': [2, 2, 2, 2, 2],
        }
    }

    return WrapperModel3D(model, in_channels=channel_size, out_channels=channel_size_out,
                          feature_embedding_dim=[32, 32, 64], num_downsampling_layers=2,
                          time_embedding_dim=64, num_groups=16,
                          repetitions=2, drop_class_labels=drop_class_labels,
                          partition_size=partition_size, mending=mending)

def PDEDiT3D_B(channel_size, 
               channel_size_out: Optional[int] = None,
               drop_class_labels: bool = False,
               partition_size: Optional[Union[Tuple[int, int, int], int]] = 1,
               mending: bool = False,
               **kwargs):

    if channel_size_out is None:
        channel_size_out = channel_size

    model = {
        'target': 'src.core.models.box.PDEDiT3DBackbone',
        'params': {
            'hidden_size': 128,
            'window_size': [4, 4, 4, 4, 4],
            'num_heads': [4, 4, 4, 4, 4],
            'depth': [2, 2, 2, 2, 2],
        }
    }

    return WrapperModel3D(model, in_channels=channel_size, out_channels=channel_size_out,
                          feature_embedding_dim=[64, 128, 128], num_downsampling_layers=2,
                          time_embedding_dim=64, num_groups=32,
                          repetitions=2, drop_class_labels=drop_class_labels,
                          partition_size=partition_size, mending=mending)

def PDEDiT3D_L(channel_size,
               channel_size_out: Optional[int] = None,
               drop_class_labels: bool = False,
               partition_size: Optional[Union[Tuple[int, int, int], int]] = 1,
               mending: bool = False,
               **kwargs):

    if channel_size_out is None:
        channel_size_out = channel_size

    model = {
        'target': 'src.core.models.box.PDEDiT3DBackbone',
        'params': {
            'hidden_size': 256,
            'window_size': [4, 4, 4, 4, 4],
            'num_heads': [8, 8, 8, 8, 8],
            'depth': [2, 2, 2, 2, 2],
        }
    }

    return WrapperModel3D(model, in_channels=channel_size, out_channels=channel_size_out,
                          feature_embedding_dim=[128, 256, 256], num_downsampling_layers=2,
                          time_embedding_dim=64, num_groups=32,
                          repetitions=2, drop_class_labels=drop_class_labels,
                          partition_size=partition_size, mending=mending)

def PDEDiT3D_XL(channel_size,
               channel_size_out: Optional[int] = None,
               drop_class_labels: bool = False,
               partition_size: Optional[Union[Tuple[int, int, int], int]] = 1,
               mending: bool = False,
               **kwargs):

    if channel_size_out is None:
        channel_size_out = channel_size

    model = {
        'target': 'src.core.models.box.PDEDiT3DBackbone',
        'params': {
            'hidden_size': 256,
            'window_size': [4, 4, 4, 4, 4],
            'num_heads': [8, 8, 8, 8, 8],
            'depth': [2, 2, 2, 2, 2],
        }
    }

    return WrapperModel3D(model, in_channels=channel_size, out_channels=channel_size_out,
                          feature_embedding_dim=[256, 256, 256], num_downsampling_layers=2,
                          time_embedding_dim=64, num_groups=64,
                          repetitions=4, drop_class_labels=drop_class_labels,
                          partition_size=partition_size, mending=mending)

class CombinedTimestepLabelParameterEmbeddings(nn.Module):
    def __init__(self, num_classes, embedding_dim, class_dropout_prob=0.1):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.class_embedder = LabelEmbedding(num_classes, embedding_dim, class_dropout_prob)
        self.parameter_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.parameter_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep, class_labels, pde_parameter, hidden_dtype=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        class_labels = self.class_embedder(class_labels)  # (N, D)

        pde_parameter_proj = self.parameter_proj(pde_parameter)
        pde_parameter_emb = self.parameter_embedder(pde_parameter_proj.to(dtype=hidden_dtype))  # (N, D)

        conditioning = timesteps_emb + class_labels + pde_parameter_emb  # (N, D)

        return conditioning

class WrapperModel3D(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self, model: Dict,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 feature_embedding_dim: Union[int, Sequence[int]] = 64,
                 num_downsampling_layers: int = 3,
                 time_embedding_dim: int = 64,
                 num_groups: int = 32,
                 num_classes: int = 1000,
                 drop_class_labels: bool = False,
                 repetitions: int = 1,
                 class_dropout_prob: float = 0.1,
                 partition_size: Optional[Union[Tuple[int, int, int], int]] = 1,
                 mending: bool = False):
        super().__init__()

        self.model_impl = instantiate_from_config(model)

        self.num_downsampling_layers = num_downsampling_layers
        self.drop_class_labels = drop_class_labels

        if not isinstance(feature_embedding_dim, Sequence):
            self.feature_embedding_dim = [feature_embedding_dim * (2 ** i)
                                          for i in range(num_downsampling_layers)]

        else:
            self.feature_embedding_dim = feature_embedding_dim

        if isinstance(partition_size, int):
            partition_size = (partition_size, partition_size, partition_size)
        self.partition_size = partition_size

        self.mending = mending

        self.time_embedding_dim = time_embedding_dim

        self.class_embedding = CombinedTimestepLabelParameterEmbeddings(
            num_classes=num_classes, embedding_dim=time_embedding_dim, class_dropout_prob=class_dropout_prob)

        self.encoder = ConditionedEncoder3D(
            in_channels=in_channels,
            feature_embedding_dim=feature_embedding_dim,
            num_downsampling_layers=num_downsampling_layers,
            embedding_dim=time_embedding_dim,
            num_groups=num_groups,
            repetitions=repetitions,
        )

        self.latent_size = self.model_impl.latent_size

        self.decoder = ConditionedDecoder3D(
            out_channels=out_channels,
            feature_embedding_dim=feature_embedding_dim[::-1],
            num_upsampling_layers=num_downsampling_layers,
            embedding_dim=time_embedding_dim,
            features_first_layer=feature_embedding_dim[-1],
            num_groups=num_groups,
            repetitions=repetitions,
        )

    def region_partition(self, x, emb=None):

        split_x, split_y, split_z = self.partition_size

        x = rearrange(
            x,
            "N C (HP HC) (WP WC) (DP DC) -> (N HP WP DP) C HC WC DC ",
            HP=split_x,
            WP=split_y,
            DP=split_z,
            HC=int(x.shape[2] // split_x),
            WC=int(x.shape[3] // split_y),
            DC=int(x.shape[4] // split_z),
        )

        if not (emb is None):
            emb = emb.repeat_interleave(split_x * split_y * split_z, dim=0)

        return x, emb


    def region_assemble_backbone(self, residuals,
                                 emb, latent,
                                 emb_backbone, residuals_backbone):

        split_x, split_y, split_z = self.partition_size

        residuals_ = []

        for res in residuals:
            res = rearrange(
                res,
                "(N HP WP DP) C HC WC DC -> N C (HP HC) (WP WC) (DP DC)",
                HP=split_x,
                WP=split_y,
                DP=split_z,
            )
            residuals_.append(res)

        emb_ = emb[::split_x * split_y * split_z]

        latent_ = rearrange(
            latent,
            "(N HP WP DP) C HC WC DC -> N C (HP HC) (WP WC) (DP DC)",
            HP=split_x,
            WP=split_y,
            DP=split_z,
        )

        emb_backbone_ = []
        for e in emb_backbone:
            e = e[::split_x * split_y * split_z]
            emb_backbone_.append(e)

        residuals_backbone_ = []
        for res in residuals_backbone:
            res = rearrange(
                res,
                "(N HP WP DP) C HC WC DC -> N C (HP HC) (WP WC) (DP DC)",
                HP=split_x,
                WP=split_y,
                DP=split_z,
            )
            residuals_backbone_.append(res)

        return residuals_, emb_, latent_, emb_backbone_, residuals_backbone_


    def region_assemble(self, x):

        split_x, split_y, split_z = self.partition_size

        x = rearrange(
            x,
            "(N HP WP DP) C HC WC DC -> N C (HP HC) (WP WC) (DP DC)",
            HP=split_x,
            WP=split_y,
            DP=split_z,
        )

        return x

    def encode(self,
                x: torch.Tensor,
                timestep: Optional[torch.Tensor] = None,
                class_labels: Optional[torch.LongTensor] = None,
                pde_parameters: Optional[torch.Tensor] = None,
                ):

        if timestep is None:
            timestep = torch.Tensor([0]).to(x.device).repeat(x.shape[0])

        if len(timestep.shape) == 0:
            timestep = timestep.unsqueeze(0)
            timestep = timestep.repeat(x.shape[0])
            timestep = timestep.to(x.device)

        if class_labels is None or self.drop_class_labels:
            class_labels = torch.Tensor([0]).to(x.device).long().repeat(x.shape[0])

        if len(class_labels.shape) == 0:
            class_labels = class_labels.unsqueeze(0)
            class_labels = class_labels.repeat(x.shape[0])
            class_labels = class_labels.to(x.device)

        if pde_parameters is None:
            pde_parameters = torch.Tensor([0]).to(x.device).repeat(x.shape[0])

        if len(pde_parameters.shape) == 0:
            pde_parameters = pde_parameters.unsqueeze(0)
            pde_parameters = pde_parameters.repeat(x.shape[0])
            pde_parameters = pde_parameters.to(x.device)

        emb = self.class_embedding(timestep, class_labels, pde_parameters)

        ## region partition
        x, emb = self.region_partition(x, emb)
        timestep = timestep.repeat_interleave(self.partition_size[0] * self.partition_size[1] * self.partition_size[2], dim=0)
        class_labels = class_labels.repeat_interleave(self.partition_size[0] * self.partition_size[1] * self.partition_size[2], dim=0)
        pde_parameters = pde_parameters.repeat_interleave(self.partition_size[0] * self.partition_size[1] * self.partition_size[2], dim=0)

        residuals = self.encoder(x, emb)
        state = residuals[-1]

        latent, emb_backbone, residuals_backbone = (
            self.model_impl.encode(state, timestep, class_labels, pde_parameters))

        ## region assemble
        residuals, emb, latent, emb_backbone, residuals_backbone = (
            self.region_assemble_backbone(residuals, emb, latent, emb_backbone, residuals_backbone))

        return Output3D(hidden_states=residuals+residuals_backbone+[latent],
                        embedding=[emb]+emb_backbone)

    def region_partition_decode(self, x, emb, residuals):
        split_x, split_y, split_z = self.partition_size

        x = rearrange(
            x,
            "N C (HP HC) (WP WC) (DP DC) -> (N HP WP DP) C HC WC DC ",
            HP=split_x,
            WP=split_y,
            DP=split_z,
            HC=int(x.shape[2] // split_x),
            WC=int(x.shape[3] // split_y),
            DC=int(x.shape[4] // split_z),
        )

        emb_ = []
        for e in emb:
            emb_.append(e.repeat_interleave(split_x * split_y * split_z, dim=0))

        residuals_ = []
        for res in residuals:

            res = rearrange(
                res,
                "N C (HP HC) (WP WC) (DP DC) -> (N HP WP DP) C HC WC DC ",
                HP=split_x,
                WP=split_y,
                DP=split_z,
                HC=int(res.shape[2] // split_x),
                WC=int(res.shape[3] // split_y),
                DC=int(res.shape[4] // split_z),
            )

            residuals_.append(res)

        return x, emb_, residuals_

    def region_assemble_final(self, reconstructed):

        split_x, split_y, split_z = self.partition_size

        reconstructed = rearrange(
            reconstructed,
            "(N HP WP DP) C HC WC DC -> N C (HP HC) (WP WC) (DP DC)",
            HP=split_x,
            WP=split_y,
            DP=split_z,
        )

        return reconstructed

    def region_assemble_mending(self, state, embedding_wrapper, residuals_wrapper):
        split_x, split_y, split_z = self.partition_size

        state = rearrange(
            state,
            "(N HP WP DP) C HC WC DC -> N C (HP HC) (WP WC) (DP DC)",
            HP=split_x,
            WP=split_y,
            DP=split_z,
        )

        embedding_wrapper_ = embedding_wrapper[::split_x * split_y * split_z]

        residuals_wrapper_ = []
        for res in residuals_wrapper:
            res = rearrange(
                res,
                "(N HP WP DP) C HC WC DC -> N C (HP HC) (WP WC) (DP DC)",
                HP=split_x,
                WP=split_y,
                DP=split_z,
            )
            residuals_wrapper_.append(res)

        return state, embedding_wrapper_, residuals_wrapper_

    def decode(self, x, embedding, residuals):

        # region partition
        x, embedding, residuals = self.region_partition_decode(x, embedding, residuals)

        residuals_wrapper = residuals[:self.num_downsampling_layers+1]
        residuals_backbone = residuals[self.num_downsampling_layers+1:-1]

        embedding_wrapper = embedding[0]
        embedding_backbone = embedding[1:]

        state = self.model_impl.decode(x, embedding_backbone, residuals_backbone)

        residuals_wrapper[-1] += state

        if self.mending:
            # region assemble for mending
            state, embedding_wrapper, residuals_wrapper = self.region_assemble_mending(state, embedding_wrapper, residuals_wrapper)

        reconstructed = self.decoder(state, embedding_wrapper, residuals_wrapper)

        if not self.mending:

            # region assemble (no mending)
            reconstructed = self.region_assemble_final(reconstructed)

        return Output3D(reconstructed=reconstructed, sample=reconstructed)

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: Optional[torch.LongTensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            pde_parameters: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_states: input tensor of shape `(batch_size, num_channels, height, width, depth)`
            timestep: timestep tensor of shape `(batch_size, 1)`
            class_labels: class label tensor of shape `(batch_size, 1)`
            pde_parameters: PDE parameters tensor of shape `(batch_size, 1)`
        """

        encoded = self.encode(hidden_states, timestep, class_labels, pde_parameters)
        state = encoded.hidden_states[-1]

        return self.decode(state, encoded.embedding, encoded.hidden_states)


def modulate(x, shift, scale):
    return (x * (1 + scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
            + shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv3d(n_feat, n_feat * 2, kernel_size=3, stride=2, padding=1, bias=False),
                                  )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv3d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  PixelShuffle3d(2))

    def forward(self, x):
        return self.body(x)

class FinalLayer(nn.Module):
    """
    The final layer of IPT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = LayerNorm3d(hidden_size, affine=False, eps=1e-6)
        self.out_proj = nn.Conv3d(hidden_size, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.out_proj(x)
        return x


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
        resolution: int = 0,
        attn_type:str = 'v2',
        seq_length: int = 0,
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

        # attention positional bias
        self.pos_emb_funct = PosEmbMLPSwinv3D(
            window_size=[resolution, resolution, resolution],
            pretrained_window_size=[resolution, resolution, resolution],
            num_heads=num_heads,
        )

        self.attn_type = attn_type

        assert self.attn_type in ['v1', 'v2'], f"attn_type {self.attn_type} not supported. Use 'v1' or 'v2'."

        if attn_type == 'v2':
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        self.resolution = resolution

    def forward(self, x, attn_mask=None):

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

        if attn_mask is not None:
            # Apply the attention mask is (precomputed for all layers in PDEDiT3D forward() function)
            mask_shape = attn_mask.shape[0]
            attn = attn.view(
                B // mask_shape, mask_shape, self.num_heads, N, N
            ) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        return x

class PDEDiTBlock(nn.Module):
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
        window_size=8,
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
        """

        self.norm1 = norm_layer(dim)

        self.cr_window = 1
        self.attn = WindowAttention3D(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            resolution=window_size,
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

        self.adain_2 = AdaLayerNormZero(dim, num_embeddings=None, norm_type="layer_norm")

    def forward(self, x,
                timestep: Optional[torch.LongTensor] = None,
                class_labels: Optional[torch.LongTensor] = None,
                emb: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None):

        B, W, N = x.shape
        Bc = emb.shape[0]

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

        x_msa = self.attn(x_msa, attn_mask=attn_mask)
        x_msa = x_msa * (1 + msa_gate[:, None])

        x = x + self.drop_path(x_msa)

        x_mlp = self.norm2(x)

        x_mlp = x_mlp * (1 + mlp_scale[:, None]) + mlp_shift[:, None]
        x_mlp = self.mlp(x_mlp)
        x_mlp = x_mlp * (1 + mlp_gate[:, None])
        x = x + self.drop_path(x_mlp)

        return x

class PDEDiT3DStage(nn.Module):
    def __init__(
        self, dim: int, depth: int,
            num_heads: int,
            window_size: int,
            periodic: bool = False,
            mlp_ratio: float = 4.0,
            shift: bool = False
    ):
        super().__init__()

        self.dim = dim
        blocks = []
        for i in range(depth):
            block = PDEDiTBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.periodic = periodic
        self.window_size = window_size
        self.shift = shift

        self.shift_size = window_size // 2 if shift else 0

    def maybe_pad(self, hidden_states, height, width, depth):
        pad_depth = (self.window_size - depth % self.window_size) % self.window_size
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_depth, 0, pad_right, 0, pad_bottom)
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def get_attn_mask(self, height, width, depth, dtype, device):

        if height < self.window_size or width < self.window_size or depth < self.window_size:
            return None

        if self.shift_size > 0 and not self.periodic:
            # calculate attention mask for shifted window multihead self attention

            padded_height = height + (self.window_size - height % self.window_size) % self.window_size
            padded_width = width + (self.window_size - width % self.window_size) % self.window_size
            padded_depth = depth + (self.window_size - depth % self.window_size) % self.window_size

            img_mask = torch.zeros((1, padded_height, padded_width, padded_depth, 1), dtype=dtype, device=device)
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
            depth_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    for depth_slice in depth_slices:
                        img_mask[:, height_slice, width_slice, depth_slice, :] = count
                        count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def forward(self,
                hidden_states: torch.Tensor,
                cond: Optional[torch.Tensor] = None,
                timestep: Optional[torch.LongTensor] = None,
                class_labels: Optional[torch.LongTensor] = None, ):

        B, C, H, W, D = hidden_states.shape

        # precompute attention mask
        attn_mask_precomputed = self.get_attn_mask(H, W, D, hidden_states.dtype,
                                                   hidden_states.device)

        for n, block in enumerate(self.blocks):

            shift_size = 0 if n % 2 == 0 else self.shift_size

            # channels last
            hidden_states = torch.permute(hidden_states, (0, 2, 3, 4, 1))

            if shift_size > 0:
                attn_mask = attn_mask_precomputed
                shifted_hidden_states = torch.roll(hidden_states, shifts=(-shift_size, -shift_size, -shift_size),
                                                   dims=(1, 2, 3))
            else:
                attn_mask = None
                shifted_hidden_states = hidden_states

            shifted_hidden_states, pad_values = self.maybe_pad(shifted_hidden_states, H, W, D)
            _, height_pad, width_pad, depth_pad, _ = shifted_hidden_states.shape


            hidden_states = window_partition(shifted_hidden_states, self.window_size)

            hidden_states = block(hidden_states, timestep=timestep, class_labels=class_labels, emb=cond,
                                      attn_mask=attn_mask)

            hidden_states = window_reverse(hidden_states, self.window_size, height_pad, width_pad, depth_pad, B)

            if height_pad > 0 or width_pad > 0 or depth_pad > 0:
                hidden_states = hidden_states[:, :H, :W, :D, :].contiguous()

            if shift_size > 0:
                hidden_states = torch.roll(hidden_states, shifts=(shift_size, shift_size, shift_size),
                                                   dims=(1, 2, 3))

            hidden_states = torch.permute(hidden_states, (0, 4, 1, 2, 3))

        return hidden_states

class PDEDiT3DBackbone(nn.Module):
    """
    Diffusion UNet model with a Transformer backbone.
    """

    def __init__(
            self,
            window_size: Union[int, Sequence[int]] = 8,
            hidden_size: int = 1152,
            max_hidden_size: int = 2048,
            depth=(2, 4, 4, 6, 4, 4, 2),
            num_heads: Union[int, Sequence[int]] = 16,
            mlp_ratio: float = 4.0,
            class_dropout_prob: float = 0.1,
            num_classes:int = 1000,
            periodic:bool = False,
            shift: bool = False,
            **kwargs
    ):
        super().__init__()

        assert len(depth) % 2 == 1, "Encoder and decoder depths must be equal."

        if num_heads is not None and isinstance(num_heads, int):
            num_heads = [num_heads] * len(depth)
        if window_size is not None and isinstance(window_size, int):
            window_size = [window_size] * len(depth)

        self.hidden_size = hidden_size

        self.latent_size = hidden_size * 2 ** (len(depth) // 2)

        self.shift = shift
        self.num_encoder_layers = len(depth) // 2

        self.num_heads = num_heads
        self.periodic = periodic

        self.max_hidden_size = max_hidden_size

        assert self.max_hidden_size >= hidden_size, f"max_hidden_size {max_hidden_size} must be greater than or equal to hidden_size {hidden_size}."

        dit_stage_args = {
            "periodic": periodic,
            'mlp_ratio': mlp_ratio,
            'shift': shift,
        }

        # timestep and label embedders
        for i in range(self.num_encoder_layers + 1):

            hidden_size_layer = min(hidden_size * 2 ** i, max_hidden_size)
            self.__setattr__(f"t_embedder_{i}", TimestepEmbedder(hidden_size_layer))
            self.__setattr__(f"param_embedder_{i}", TimestepEmbedder(hidden_size_layer))
            self.__setattr__(f"y_embedder_{i}", LabelEmbedder(num_classes, hidden_size_layer, class_dropout_prob))

        # encoder
        for i in range(self.num_encoder_layers):
            hidden_size_layer = min(hidden_size * 2 ** i, max_hidden_size)
            self.__setattr__(f"encoder_level_{i}", PDEDiT3DStage(dim=hidden_size_layer, num_heads=num_heads[i],
                                            window_size=window_size[i], depth=depth[i], **dit_stage_args))
            self.__setattr__(f"down{i}_{i+1}", Downsample(hidden_size_layer))

        # latent
        hidden_size_latent = min(hidden_size * 2 ** self.num_encoder_layers, max_hidden_size)
        self.latent = PDEDiT3DStage(dim=hidden_size_latent, num_heads=num_heads[self.num_encoder_layers],
                                            window_size=window_size[self.num_encoder_layers], depth=depth[self.num_encoder_layers], **dit_stage_args)

        hidden_size_layer0 = min(hidden_size * 2, max_hidden_size)
        # double hidden size for last decoder layer 0
        self.__setattr__("up1_0", Upsample(hidden_size_layer0))
        self.__setattr__("reduce_chan_level0", nn.Conv3d(int(1.5 * min(hidden_size, max_hidden_size)), hidden_size_layer0, kernel_size=1, bias=True))
        self.__setattr__("decoder_level_0", PDEDiT3DStage(dim=hidden_size_layer0, num_heads=num_heads[self.num_encoder_layers + 1],
                                        window_size=window_size[self.num_encoder_layers + 1], depth=depth[self.num_encoder_layers + 1], **dit_stage_args))

        # decoder layers 1 - num_encoder_layers
        for i in range(1, self.num_encoder_layers):

            hidden_size_layer = min(hidden_size * 2 ** i, max_hidden_size)
            if 2 * hidden_size_layer >= max_hidden_size:
                keep_dim = True
                hidden_size_upsample = max_hidden_size
            else:
                keep_dim = False
                hidden_size_upsample = 2 * hidden_size_layer

            self.__setattr__(f"up{i+1}_{i}", Upsample(hidden_size_upsample))
            self.__setattr__(f"reduce_chan_level{i}", nn.Conv3d(int(hidden_size_layer * 1.5), hidden_size_layer, kernel_size=1, bias=True))
            self.__setattr__(f"decoder_level_{i}", PDEDiT3DStage(dim=hidden_size_layer, num_heads=num_heads[self.num_encoder_layers + i + 1],
                                            window_size=window_size[self.num_encoder_layers + i + 1], depth=depth[self.num_encoder_layers + i + 1], **dit_stage_args))

        self.final_layer = FinalLayer(2 * self.hidden_size, self.hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        for i in range(self.num_encoder_layers):
            # Initialize label embedding table:
            nn.init.normal_(self.__getattr__(f"y_embedder_{i}").embedding_table.weight, std=0.02)

            # Initialize timestep embedding MLP:
            nn.init.normal_(self.__getattr__(f"t_embedder_{i}").mlp[0].weight, std=0.02)
            nn.init.normal_(self.__getattr__(f"t_embedder_{i}").mlp[2].weight, std=0.02)

            nn.init.normal_(self.__getattr__(f"param_embedder_{i}").mlp[0].weight, std=0.02)
            nn.init.normal_(self.__getattr__(f"param_embedder_{i}").mlp[2].weight, std=0.02)


        blocks = [self.__getattr__(f"encoder_level_{i}") for i in range(self.num_encoder_layers)]
        blocks += [self.latent]
        blocks += [self.__getattr__(f"decoder_level_{i}") for i in range(self.num_encoder_layers)]

        for block in blocks:

            for blc in block.blocks:

                nn.init.constant_(blc.adain_2.linear.weight, 0)
                nn.init.constant_(blc.adain_2.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.out_proj.weight, 0)
        nn.init.constant_(self.final_layer.out_proj.bias, 0)

    def encode(self, x, t, class_labels, pde_parameters):
        """
        Encoding of PDEDiT3D
        x: (N, C, H, W, D) tensor of spatial inputs (images or latent representations of images)
        t: (N, ) tensor of diffusion timesteps
        class_labels: (N, ) tensor of class labels
        pde_parameters: (N, ) tensor of PDE parameters
        """

        emb_list = []
        for i in range(self.num_encoder_layers + 1):
            t_emb = self.__getattr__(f"t_embedder_{i}")(t)
            pde_emb = self.__getattr__(f"param_embedder_{i}")(pde_parameters)
            y_emb = self.__getattr__(f"y_embedder_{i}")(class_labels, self.training)
            c = t_emb + y_emb + pde_emb
            emb_list.append(c)

        residuals_list = []
        for i, c in enumerate(emb_list[:-1]):
            # encoder
            out_enc_level = self.__getattr__(f"encoder_level_{i}")(x, c)
            residuals_list.append(out_enc_level)
            x = self.__getattr__(f"down{i}_{i + 1}")(out_enc_level)

        c = emb_list[-1]
        x = self.latent(x, c)

        return x, emb_list, residuals_list

    def decode(self, x, embeddings, residuals):

        for i, (residual, emb) in enumerate(zip(residuals[1:][::-1], embeddings[1:-1][::-1])):
            # decoder
            x = self.__getattr__(f"up{self.num_encoder_layers - i}_{self.num_encoder_layers - i - 1}")(x)
            x = torch.cat([x, residual], 1)
            x = self.__getattr__(f"reduce_chan_level{self.num_encoder_layers - i - 1}")(x)
            x = self.__getattr__(f"decoder_level_{self.num_encoder_layers - i - 1}")(x, emb)

        x = self.__getattr__(f"up1_0")(x)
        x = torch.cat([x, residuals[0]], 1)
        x = self.__getattr__(f"reduce_chan_level0")(x)
        x = self.__getattr__(f"decoder_level_0")(x, embeddings[1])

        # output
        x = self.final_layer(x, embeddings[1])

        return x

    def forward(self, x, t, pde_parameters, class_labels):
        """
        Forward pass of PDEDiT3D
        x: (N, C, H, W, D) tensor of spatial inputs (images or latent representations of images)
        t: (N, ) tensor of diffusion timesteps
        pde_parameters: (N, ) tensor of PDE parameters
        class_labels: (N, ) tensor of class labels
        """

        x, emb_list, residuals_list = self.encode(x, t, pde_parameters, class_labels)

        x = self.decode(x, emb_list, residuals_list)

        return x
