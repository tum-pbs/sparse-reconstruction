# Copyright 2024 The swirl_dynamics Authors.
# Modifications made by the CAM Lab at ETH Zurich.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""3D U-Net denoiser models.

Intended for inputs with dimensions (batch, time, x, y, channels). The U-Net
stacks successively apply 2D downsampling/upsampling in space only. At each
resolution, an axial attention block (involving space and/or time) is applied.
"""

from collections.abc import Sequence
from typing import Any, Callable, Literal, Optional
import functools
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config

from src.core.models.box.output3d import Output3D

Tensor = torch.Tensor


def default_init(scale: float = 1e-10):
    """Initialization of weights and biases with scaling"""

    def initializer(tensor: Tensor):
        """We need to differentiate between biases and weights"""

        if tensor.ndim == 1:  # if bias
            bound = torch.sqrt(torch.tensor(3.0)) * scale
            with torch.no_grad():
                return tensor.uniform_(-bound, bound)

        else:  # if weights
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
            std = torch.sqrt(torch.tensor(scale / ((fan_in + fan_out) / 2.0)))
            bound = torch.sqrt(torch.tensor(3.0)) * std  # uniform dist. scaling factor
            with torch.no_grad():
                return tensor.uniform_(-bound, bound)

    return initializer


class ConvLocal2d(nn.Module):
    """Customized locally connected 2D convolution (ConvLocal) for PyTorch"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride: int = 1,
        padding: int = 0,
        padding_mode: str = "constant",
        use_bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        super(ConvLocal2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.padding_mode = padding_mode
        self.use_bias = use_bias
        self.device = device
        self.dtype = dtype

        # Weights for each spatial location (out_height x out_width)
        self.weights = None

        if self.use_bias:
            self.bias = nn.Parameter(
                torch.zeros(out_channels, dtype=self.dtype, device=self.device)
            )
        else:
            self.bias = None

    def forward(self, x):
        if len(x.shape) < 4:
            raise ValueError(
                f"Local 2D Convolution with shape length of 4 instead of {len(x.shape)}"
            )

        # Input dim: (batch_size, in_channels, height, width)
        # width, height = lat, lon
        batch_size, in_channels, height, width = x.shape

        if self.padding > 0:
            x = F.pad(
                x,
                [self.padding, self.padding, self.padding, self.padding],
                mode=self.padding_mode,
                value=0,
            )

        out_height = (height - self.kernel_size[0] + 2 * self.padding) // self.stride[
            0
        ] + 1
        out_width = (width - self.kernel_size[1] + 2 * self.padding) // self.stride[
            1
        ] + 1

        # Initialize weights
        if self.weights is None:
            self.weights = nn.Parameter(
                torch.empty(
                    out_height,
                    out_width,
                    self.out_channels,
                    in_channels,
                    self.kernel_size[0],
                    self.kernel_size[1],
                    device=self.device,  # x.device
                    dtype=self.dtype,
                )
            )
            torch.nn.init.xavier_uniform_(self.weights)

        output = torch.zeros(
            (batch_size, self.out_channels, out_height, out_width),
            dtype=self.dtype,
            device=self.device,
        )

        # manually scripted convolution.
        for i in range(out_height):
            for j in range(out_width):
                patch = x[
                    :,
                    :,
                    i * self.stride[0] : i * self.stride[0] + self.kernel_size[0],
                    j * self.stride[1] : j * self.stride[1] + self.kernel_size[1],
                ]
                # Sums of the product based on Einstein's summation convention
                output[:, :, i, j] = torch.einsum(
                    "bchw, ocwh->bo", patch, self.weights[i, j]
                )

        if self.use_bias:
            bias_shape = [1] * len(x.shape)
            bias_shape[1] = -1
            output += self.bias.view(bias_shape)

        return output


class LatLonConv(nn.Module):
    """2D convolutional layer adapted to inputs a lot-lon grid"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        order: Literal["latlon", "lonlat"] = "latlon",
        use_bias: bool = True,
        strides: tuple[int, int] = (1, 1),
        use_local: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Any | None = None,
        **kwargs,
    ):
        super(LatLonConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.order = order
        self.use_bias = use_bias
        self.strides = strides
        self.use_local = use_local
        self.dtype = dtype
        self.device = device

        if self.use_local:
            self.conv = ConvLocal2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=strides,
                bias=use_bias,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=strides,
                bias=use_bias,
                padding=0,
                device=self.device,
                dtype=self.dtype,
            )

    def forward(self, inputs):
        """Applies lat-lon and lon-lat convolution with edge and circular padding"""
        if len(inputs.shape) < 4:
            raise ValueError(f"Input must be 4D or higher: {inputs.shape}.")

        if self.kernel_size[0] % 2 == 0 or self.kernel_size[1] % 2 == 0:
            raise ValueError(f"Current kernel size {self.kernel_size} must be odd.")

        if self.order == "latlon":
            lon_axis, lat_axis = (-3, -2)
            lat_pad, lon_pad = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        elif self.order == "lonlat":
            lon_axis, lat_axis = (-3, -2)
            lon_pad, lat_pad = self.kernel_size[1] // 2, self.kernel_size[0] // 2
            # TODO: There is no difference lon_axis and lat_axis in "lonlat" should be switched?
        else:
            raise ValueError(
                f"Unrecogniized order {self.order} - 'loatlon' or 'lonlat expected."
            )

        # Circular padding to longitudinal (lon) axis
        padded_inputs = F.pad(inputs, [0, 0, lon_pad, lon_pad], mode="circular")
        # Edge padding to latitudinal (lat) axis
        padded_inputs = F.pad(padded_inputs, [lat_pad, lat_pad, 0, 0], mode="replicate")

        return self.conv(padded_inputs)


class DownsampleConv(nn.Module):
    """Downsampling layer through strided convolution.

    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
      ratios: downsampling ratio for the resolution, increase of the channel dimension
      case: dimensionality of the dataset, 1D, 2D and 3D (int: 1, 2, or 3)
      use_bias:  If True, adds a learnable bias to the output. Default: True
      kernel_init: initializations for the convolution weights
      bias_init: initializtations for the convolution bias values
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_resolution: Sequence[int],
        ratios: Sequence[int],
        case: int,
        use_bias: bool = True,
        kernel_init: Callable = torch.nn.init.kaiming_uniform_,
        bias_init: Callable = torch.nn.init.zeros_,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        super(DownsampleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_dim = len(spatial_resolution)  # Spatial dimension of dataset
        self.ratios = ratios
        self.bias_init = bias_init
        self.dtype = dtype
        self.device = device

        dataset_shape = self.kernel_dim + 2  # spatial resolution + channel + batch_size

        # Check if input has the correct shape and size to be downsampled!
        if dataset_shape <= len(self.ratios):
            raise ValueError(
                f"Inputs ({dataset_shape}) for downsampling must have at least 1 more dimension "
                f"than that of 'ratios' ({self.ratios})."
            )

        if not all(s % r == 0 for s, r in zip(spatial_resolution, self.ratios)):
            raise ValueError(
                f"Input dimensions (spatial) {spatial_resolution} must divide the "
                f"downsampling ratio {self.ratios}."
            )

        self.use_bias = use_bias
        if kernel_init is torch.nn.init.kaiming_uniform_:
            self.kernel_init = functools.partial(kernel_init, a=np.sqrt(5))
        else:
            self.kernel_init = kernel_init

        # For downsampling padding = 0 and stride > 1
        if case == 1:
            self.conv_layer = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=ratios,
                stride=ratios,
                bias=use_bias,
                padding=0,
                device=self.device,
                dtype=self.dtype,
            )

        elif case == 2:
            self.conv_layer = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=ratios,
                stride=ratios,
                bias=use_bias,
                padding=0,
                device=self.device,
                dtype=self.dtype,
            )

        elif case == 3:
            self.conv_layer = nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=ratios,
                stride=ratios,
                bias=use_bias,
                padding=0,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            raise ValueError(f"Dataset dimension should either be 1D, 2D or 3D")

        # Initialize with variance_scaling
        # Only use this if the activation function is ReLU or smth. similar
        self.kernel_init(self.conv_layer.weight)
        self.bias_init(self.conv_layer.bias)

    def forward(self, inputs):
        """Applies strided convolution for downsampling."""

        return self.conv_layer(inputs)

def ConvLayer(
    in_channels: int,
    out_channels: int,
    kernel_size: int | Sequence[int],
    padding_mode: str,
    padding: int = 0,
    stride: int = 1,
    use_bias: bool = True,
    use_local: bool = False,
    case: int = 2,
    kernel_init: Callable = None,
    bias_init: Callable = torch.nn.init.zeros_,
    dtype: torch.dtype = torch.float32,
    device: Any | None = None,
    **kwargs,
) -> nn.Module:
    """Factory for different types of convolution layers.

    Where the last part requires a case differentiation:
    case == 1: 1D (bs, c, width)
    case == 2: 2D (bs, c, height, width)
    case == 3: 3D (bs, c, depth, height, width)
    """
    if isinstance(padding_mode, str) and padding_mode.lower() in ["lonlat", "latlon"]:
        if not (isinstance(kernel_size, tuple) and len(kernel_size) == 2):
            raise ValueError(
                f"kernel size {kernel_size} must be a length-2 tuple "
                f"for convolution type {padding_mode}."
            )
        return LatLonConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            order=padding_mode.lower(),
            dtype=dtype,
            device=device,
            **kwargs,
        )

    elif use_local:
        return ConvLocal2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            use_bias=use_bias,
            device=device,
            dtype=dtype,
        )
    else:
        if case == 1:
            conv_layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding_mode=padding_mode.lower(),
                padding=padding,
                stride=stride,
                bias=use_bias,
                device=device,
                dtype=dtype,
            )
        elif case == 2:
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding_mode=padding_mode.lower(),
                padding=padding,
                stride=stride,
                bias=use_bias,
                device=device,
                dtype=dtype,
            )
        elif case == 3:
            conv_layer = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding_mode=padding_mode.lower(),
                padding=padding,
                stride=stride,
                bias=use_bias,
                device=device,
                dtype=dtype,
            )

    # Initialize weights and biases
    if kernel_init is not None:
        kernel_init(conv_layer.weight)
        bias_init(conv_layer.bias)

    return conv_layer

class FourierEmbedding(nn.Module):
    """Fourier embedding."""

    def __init__(
        self,
        dims: int = 64,
        max_freq: float = 2e4,
        projection: bool = True,
        act_fun: Callable[[Tensor], Tensor] = F.silu,
        dtype: torch.dtype = torch.float32,
        device: Any | None = None,
        max_val: float = 1e6,  # for numerical stability
    ):
        super(FourierEmbedding, self).__init__()

        self.dims = dims
        self.max_freq = max_freq
        self.projection = projection
        self.act_fun = act_fun
        self.dtype = dtype
        self.device = device
        self.max_val = max_val

        logfreqs = torch.linspace(
            0,
            torch.log(
                torch.tensor(self.max_freq, dtype=self.dtype, device=self.device)
            ),
            self.dims // 2,
            dtype=self.dtype,
            device=self.device,
        )

        # freqs are constant and scaled with pi!
        const_freqs = torch.pi * torch.exp(logfreqs)[None, :]  # Shape: (1, dims//2)

        # Store freqs as a non-trainable buffer also to ensure device and dtype transfers
        self.register_buffer("const_freqs", const_freqs)

        if self.projection:
            self.lin_layer1 = nn.Linear(
                self.dims, 2 * self.dims, dtype=self.dtype, device=self.device
            )
            self.lin_layer2 = nn.Linear(
                2 * self.dims, self.dims, dtype=self.dtype, device=self.device
            )

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 1, "Input tensor must be 1D"

        # Use the registered buffer const_freqs
        x_proj = self.const_freqs * x[:, None]
        # x_proj is now a 2D tensor
        x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        # clamping values to avoid running into numerical instability!
        x_proj = torch.clamp(x_proj, min=-self.max_val, max=self.max_val)

        if self.projection:
            x_proj = self.lin_layer1(x_proj)
            x_proj = self.act_fun(x_proj)
            x_proj = self.lin_layer2(x_proj)

        return x_proj

def permute_tensor(tensor: Tensor, kernel_dim: int) -> Tensor:
    if kernel_dim == 1:
        # Reshape for the 1D case
        return tensor.permute(0, 2, 1)
    elif kernel_dim == 2:
        # Reshape for the 2D case
        return tensor.permute(0, 3, 2, 1)
    elif kernel_dim == 3:
        # Reshape for the 3D case
        return tensor.permute(0, 4, 3, 2, 1)
    else:
        raise ValueError(
            f"Unsupported kernel_dim={kernel_dim}. Only 1D, 2D, and 3D data are valid."
        )

def reshape_jax_torch(tensor: Tensor, kernel_dim: int = None) -> Tensor:
    """
    A jax based dataloader is off shape (bs, width, height, depth, c),
    while a PyTorch based dataloader is off shape (bs, c, depth, height, width).

    It transforms a tensor for the 2D and 3D case as follows:
    - 2D: (bs, c, depth, height, width) <-> (bs, width, height, depth, c)
    - 3D: (bs, c, height, width) <-> (bs, width, height, c)

    Code can be used either dynamics or static.
    - dynamic: if kernel_dim is None
    - static: if kernel_dim
    """
    if kernel_dim is None:
        # Infer kernel_dim dynamically based on tensor.ndim
        kernel_dim = tensor.ndim - 2  # Extract batch_size and channel

    return permute_tensor(tensor, kernel_dim)

class AddAxialPositionEmbedding(nn.Module):
    """Adds trainable axial position embeddings to the inputs."""

    def __init__(
        self,
        position_axis: int,
        spatial_resolution: int,
        input_channels: int,
        initializer: nn.init = nn.init.normal_,
        std: float = 0.02,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        super(AddAxialPositionEmbedding, self).__init__()

        self.initializer = initializer
        self.position_axis = position_axis
        self.spatial_resolution = spatial_resolution
        self.input_channels = input_channels
        self.kernel_dim = len(spatial_resolution)
        self.input_dim = self.kernel_dim + 2  # channel and batch_size in addition
        self.std = std
        self.dtype = dtype
        self.device = device

        pos_axis = self.position_axis
        pos_axis = pos_axis if pos_axis >= 0 else pos_axis + self.input_dim

        if not 0 <= pos_axis < self.input_dim:
            raise ValueError(f"Invalid position ({self.position_axis}) or feature axis")

        self.feat_axis = self.input_dim - 1
        if pos_axis == self.feat_axis:
            raise ValueError(
                f"Position axis ({self.position_axis}) must not coincide with feature"
                f" axis ({self.feat_axis})!"
            )

        unsqueeze_axes = tuple(set(range(self.input_dim)) - {pos_axis, self.feat_axis})
        self.unsqueeze_axes = sorted(unsqueeze_axes)

        self.embedding = nn.Parameter(
            self.initializer(
                torch.empty(
                    (spatial_resolution[pos_axis - 1], input_channels),
                    dtype=self.dtype,
                    device=self.device,
                ),
                std=self.std,
            )
        )

    def forward(self, inputs: Tensor) -> Tensor:
        # Tensor should be off shape: (bs, width, height, depth, c)

        embedding = self.embedding

        if self.unsqueeze_axes:
            for axis in self.unsqueeze_axes:
                embedding = embedding.unsqueeze(dim=axis)

        return inputs + embedding


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_heads=None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        super().__init__()

        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.device = device
        self.dtype = dtype

        self.heads = heads
        self.to_q = nn.Linear(
            dim, dim_hidden, bias=False, device=self.device, dtype=self.dtype
        )
        self.to_kv = nn.Linear(
            dim, 2 * dim_hidden, bias=False, device=self.device, dtype=self.dtype
        )
        self.to_out = nn.Linear(dim_hidden, dim, device=self.device, dtype=self.dtype)

        # self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier uniform distribution."""
        torch.nn.init.xavier_uniform_(self.to_q.weight)
        torch.nn.init.xavier_uniform_(self.to_kv.weight)
        torch.nn.init.xavier_uniform_(self.to_out.weight)

    def forward(self, query, kv=None):
        kv = query if kv is None else kv
        q, k, v = (self.to_q(query), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads
        merge_heads = (
            lambda query: query.reshape(b, -1, h, e)
            .transpose(1, 2)
            .reshape(b * h, -1, e)
        )
        q, k, v = map(merge_heads, (q, k, v))
        dots = torch.einsum("bie,bje->bij", q, k) * (e**-0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum("bij,bje->bie", dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out


class MultiHeadDotProductAttention(nn.Module):
    """Mulit Head Dot Product Attention with querry and key normalization"""

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        normalize_qk: bool = False,
        dropout: float = 0.0,
        device: Any | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super(MultiHeadDotProductAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.normalize_qk = normalize_qk
        self.dropout = dropout
        self.device = device
        self.dtype = dtype

        if emb_dim % num_heads != 0:
            raise ValueError(
                "Embedding Dimension must be divisible through the number of heads"
            )
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.emb_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
            device=self.device,
            dtype=self.dtype,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.multihead_attention.in_proj_weight)
        nn.init.xavier_uniform_(self.multihead_attention.out_proj.weight)

    def forward(
        self, query: Tensor, key: Tensor = None, value: Tensor = None
    ) -> Tensor:
        """Required shape for multihead attention is:

        2D case: (bs, width*height, emb_dim)
        3D case: (bs, length, emb_dim)
        where the length is just the height, width or depth. Used for axial self attention
        """

        if key is None and value is None:
            key = value = query

        elif key is None:
            if value is not None:
                raise ValueError("value can not be not None if key is None")
            key = query

        if value is None:
            value = key

        if self.normalize_qk:
            # L2 normalization across the feature dimension
            query = F.normalize(query, p=2, dim=-1)
            key = F.normalize(key, p=2, dim=-1)

        out, _ = self.multihead_attention(query, key, value)

        return out

class AxialSelfAttention(nn.Module):
    """Axial self-attention for multidimensional inputs."""

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        attention_axis: int = -2,
        dropout: float = 0.0,
        normalize_qk: bool = False,
        use_simple_attention: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ):
        super(AxialSelfAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.attention_axis = attention_axis
        self.dropout = dropout
        self.normalize_qk = normalize_qk
        self.dtype = dtype
        self.device = device

        if use_simple_attention:
            self.attention = SelfAttention(
                dim=self.emb_dim,
                heads=self.num_heads,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self.attention = MultiHeadDotProductAttention(
                emb_dim=self.emb_dim,
                num_heads=self.num_heads,
                normalize_qk=self.normalize_qk,
                dropout=self.dropout,
                device=self.device,
                dtype=self.dtype,
            )

    def forward(self, inputs: Tensor) -> Tensor:
        """Applies axial self-attention to the inputs.

        inputs: Tensor should have the shape (bs, width, height, depth, c)
            where c here is the embedding dimension
        """

        if self.attention_axis == -1 or self.attention_axis == inputs.ndim - 1:
            raise ValueError(
                f"Attention axis ({self.attention_axis}) cannot be the last axis,"
                " which is treated as the features!"
            )

        inputs = torch.swapaxes(inputs, self.attention_axis, -2)
        query = inputs.reshape(-1, *inputs.shape[-2:])

        out = self.attention(query=query)

        out = out.reshape(*inputs.shape)
        out = torch.swapaxes(out, -2, self.attention_axis)

        return out

class CombineResidualWithSkip(nn.Module):
    """Combine residual and skip connections.

    Attributes:
      project_skip: Whether to add a linear projection layer to the skip
        connections. Mandatory if the number of channels are different between
        skip and residual values.
    """

    def __init__(
        self,
        residual_channels: int,
        skip_channels: int,
        kernel_dim: int = None,
        project_skip: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Any | None = None,
    ):
        super(CombineResidualWithSkip, self).__init__()

        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.kernel_dim = kernel_dim
        self.project_skip = project_skip
        self.dtype = dtype
        self.device = device

        if residual_channels != skip_channels and not project_skip:
            raise ValueError(
                f"Residual tensor has {residual_channels}, Skip tensor has {skip_channels}. "
                f"Set project_skip to True to resolve this mismatch."
            )

        if self.residual_channels and self.skip_channels and self.project_skip:
            self.skip_projection = nn.Linear(
                skip_channels, residual_channels, device=self.device, dtype=self.dtype
            )
            torch.nn.init.kaiming_uniform_(self.skip_projection.weight, a=np.sqrt(5))
            torch.nn.init.zeros_(self.skip_projection.bias)
        else:
            self.skip_projection = None

    def forward(self, residual: Tensor, skip: Tensor) -> Tensor:
        # residual, skip (bs, c, w, h, d)
        if self.project_skip:
            skip = reshape_jax_torch(
                self.skip_projection(reshape_jax_torch(skip, self.kernel_dim)),
                self.kernel_dim,
            )

        return (skip + residual) / math.sqrt(2)


class AdaptiveScale(nn.Module):
    """Adaptively scale the input based on embedding.

    Conditional information is projected to two vectors of length c where c is
    the number of channels of x, then x is scaled channel-wise by first vector
    and offset channel-wise by the second vector.

    This method is now standard practice for conditioning with diffusion models,
    see e.g. https://arxiv.org/abs/2105.05233, and for the
    more general FiLM technique see https://arxiv.org/abs/1709.07871.
    """

    def __init__(
        self,
        emb_channels: int,
        input_channels: int,
        input_dim: int,
        act_fun: Callable[[Tensor], Tensor] = F.silu,
        dtype: torch.dtype = torch.float32,
        device: Any | None = None,
    ):
        super(AdaptiveScale, self).__init__()

        self.emb_channels = emb_channels
        self.input_channels = input_channels
        self.input_dim = input_dim
        self.act_fun = act_fun
        self.dtype = dtype
        self.device = device

        # self.affine = None
        self.affine = nn.Linear(
            in_features=emb_channels,
            out_features=input_channels * 2,
            dtype=self.dtype,
            device=self.device,
        )
        default_init(1.0)(self.affine.weight)
        torch.nn.init.zeros_(self.affine.bias)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """Adaptive scaling applied to the channel dimension.

        Args:
          x: Tensor to be rescaled.
          emb: Embedding values that drives the rescaling.

        Returns:
          Rescaled tensor plus bias
        """
        assert (
            len(emb.shape) == 2
        ), "The dimension of the embedding needs to be two, instead it was : " + str(
            len(emb.shape)
        )

        scale_params = self.affine(self.act_fun(emb))  # (bs, c*2)

        scale, bias = torch.chunk(scale_params, 2, dim=-1)
        scale = scale[(...,) + (None,) * self.input_dim]
        bias = bias[(...,) + (None,) * self.input_dim]

        return x * (scale + 1) + bias

class ConvBlock(nn.Module):
    """A basic two-layer convolution block with adaptive scaling in between.

    main conv path:
    --> GroupNorm --> Swish --> Conv -->
        GroupNorm --> FiLM --> Swish --> Dropout --> Conv

    shortcut path:
    --> Linear

    Attributes:
      channels: The number of output channels.
      kernel_sizes: Kernel size for both conv layers.
      padding: The type of convolution padding to use.
      dropout: The rate of dropout applied in between the conv layers.
      film_act_fun: Activation function for the FilM layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        kernel_size: tuple[int, ...],
        padding_mode: str = "circular",
        padding: int = 0,
        stride: int = 1,
        use_bias: bool = True,
        case: int = 2,
        dropout: float = 0.0,
        film_act_fun: Callable[[Tensor], Tensor] = F.silu,
        act_fun: Callable[[Tensor], Tensor] = F.silu,
        dtype: torch.dtype = torch.float32,
        device: Any | None = None,
        **kwargs,
    ):
        super(ConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.dropout = dropout
        self.film_act_fun = film_act_fun
        self.act_fun = act_fun
        self.dtype = dtype
        self.device = device
        self.padding = padding
        self.stride = stride
        self.use_bias = use_bias
        self.case = case

        self.norm1 = nn.GroupNorm(
            min(max(self.in_channels // 4, 1), 32),
            self.in_channels,
            device=self.device,
            dtype=self.dtype,
        )

        self.conv1 = ConvLayer(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding_mode=self.padding_mode,
            padding=self.padding,
            stride=self.stride,
            use_bias=self.use_bias,
            case=self.case,
            kernel_init=default_init(1.0),
            dtype=self.dtype,
            device=self.device,
            **kwargs,
        )

        self.norm2 = nn.GroupNorm(
            min(max(self.out_channels // 4, 1), 32),
            self.out_channels,
            device=self.device,
            dtype=self.dtype,
        )

        self.film = AdaptiveScale(
            emb_channels=self.emb_channels,
            input_channels=self.out_channels,
            input_dim=self.case,
            act_fun=self.film_act_fun,
            dtype=self.dtype,
            device=self.device,
        )

        self.dropout_layer = nn.Dropout(dropout)

        self.conv2 = ConvLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding_mode=self.padding_mode,
            padding=self.padding,
            stride=self.stride,
            use_bias=self.use_bias,
            case=self.case,
            kernel_init=default_init(1.0),
            dtype=self.dtype,
            device=self.device,
        )
        self.res_layer = CombineResidualWithSkip(
            residual_channels=self.out_channels,
            skip_channels=self.in_channels,
            kernel_dim=self.case,
            project_skip=True,
            dtype=self.dtype,
            device=self.device,
        )

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        # ConvBlock per level in the UNet doesn't change it's number of
        # channels or resolution.
        h = x.clone()
        # First block
        h = self.norm1(h)
        h = self.act_fun(h)
        h = self.conv1(h)
        # second block
        h = self.norm2(h)
        h = self.film(h, emb)
        h = self.act_fun(h)
        # For dropout use the following logic: set UNet to .train() or .eval()
        h = self.dropout_layer(h)
        h = self.conv2(h)
        # residual connection
        return self.res_layer(residual=h, skip=x)


class AxialSelfAttentionBlock(nn.Module):
    """Block consisting of (potentially multiple) axial attention layers."""

    def __init__(
        self,
        in_channels: int,
        spatial_resolution: Sequence[int],
        attention_axes: int | Sequence[int] = -2,
        add_position_embedding: bool | Sequence[bool] = True,
        num_heads: int | Sequence[int] = 1,
        normalize_qk: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        super(AxialSelfAttentionBlock, self).__init__()

        self.in_channels = in_channels
        self.dtype = dtype
        self.device = device
        self.kernel_dim = len(spatial_resolution)
        self.normalize_qk = normalize_qk

        # permute spatial resolution since the following transformation in the 3D case is being done:
        # (bs, c, w, h, d) -> (bs, d, h, w, c) thus the resolution changes (w, h, d) -> (d, h, w)
        spatial_resolution = spatial_resolution[::-1]

        if isinstance(attention_axes, int):
            attention_axes = (attention_axes,)
        self.attention_axes = attention_axes
        num_axes = len(attention_axes)

        if isinstance(add_position_embedding, bool):
            add_position_embedding = (add_position_embedding,) * num_axes
        self.add_position_embedding = add_position_embedding

        if isinstance(num_heads, int):
            num_heads = (num_heads,) * num_axes
        self.num_heads = num_heads

        self.attention_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        self.dense_layers = nn.ModuleList()
        self.pos_emb_layers = nn.ModuleList()

        for level, (axis, add_emb, num_head) in enumerate(
            zip(self.attention_axes, self.add_position_embedding, self.num_heads)
        ):
            if add_emb:
                self.pos_emb_layers.append(
                    AddAxialPositionEmbedding(
                        position_axis=axis,
                        spatial_resolution=spatial_resolution,
                        input_channels=self.in_channels,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )

            self.norm_layers_1.append(
                nn.GroupNorm(
                    min(max(self.in_channels // 4, 1), 32),
                    self.in_channels,
                    device=self.device,
                    dtype=self.dtype,
                )
            )

            self.attention_layers.append(
                AxialSelfAttention(
                    emb_dim=self.in_channels,
                    num_heads=num_head,
                    attention_axis=axis,
                    dropout=0.1,
                    normalize_qk=self.normalize_qk,
                    dtype=self.dtype,
                    device=self.device,
                )
            )

            self.norm_layers_2.append(
                nn.GroupNorm(
                    min(max(self.in_channels // 4, 1), 32),
                    self.in_channels,
                    device=self.device,
                    dtype=self.dtype,
                )
            )

            self.dense_layers.append(
                nn.Linear(
                    in_features=in_channels,
                    out_features=in_channels,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            default_init(1.0)(self.dense_layers[level].weight)
            torch.nn.init.zeros_(self.dense_layers[level].bias)

        self.residual_layer = CombineResidualWithSkip(
            residual_channels=in_channels,
            skip_channels=in_channels,
            kernel_dim=self.kernel_dim,
            project_skip=False,
            dtype=self.dtype,
            device=self.device,
        )

    def forward(self, x: Tensor) -> Tensor:

        # Axial attention ops followed by a projection.
        h = x.clone()
        for level, (axis, add_emb, num_head) in enumerate(
            zip(self.attention_axes, self.add_position_embedding, self.num_heads)
        ):
            if add_emb:
                # Embedding
                h = reshape_jax_torch(
                    self.pos_emb_layers[level](reshape_jax_torch(h, self.kernel_dim)),
                    self.kernel_dim,
                )
            # Group Normalization
            h = self.norm_layers_1[level](h)
            # Attention Layer
            h = reshape_jax_torch(
                self.attention_layers[level](reshape_jax_torch(h, self.kernel_dim)),
                self.kernel_dim,
            )
            # Group Normalization
            h = self.norm_layers_2[level](h)
            # Dense Layer
            h = reshape_jax_torch(
                self.dense_layers[level](reshape_jax_torch(h, self.kernel_dim)),
                self.kernel_dim,
            )
        # Residual Connection out of the Loop!
        h = self.residual_layer(residual=h, skip=x)

        return h


class ChannelToSpace(nn.Module):
    """Reshapes data from the channel to spatial dims as a way to upsample.

    As an example, for an input of shape (*batch, x, y, z) and block_shape of
    (a, b), additional spatial dimensions are first formed from the channel
    dimension (always the last one), i.e. reshaped into
    (*batch, x, y, a, b, z//(a*b)). Then the new axes are interleaved with the
    original ones to arrive at shape (*batch, x, a, y, b, z//(a*b)). Finally, the
    new axes are merged with the original axes to yield final shape
    (*batch, x*a, y*b, z//(a*b)).

    Args:
      inputs: The input array to upsample.
      block_shape: The shape of the block that will be formed from the channel
        dimension. The number of elements (i.e. prod(block_shape) must divide the
        number of channels).
      kernel_dim: Defines the dimension of the input 1D, 2D or 3D
      spatial_resolution: Tuple with the spatial resolution components

    Returns:
      The upsampled array.
    """

    def __init__(
        self,
        block_shape: Sequence[int],
        in_channels: int,
        kernel_dim: int,
        spatial_resolution: Sequence[int],
    ):
        super(ChannelToSpace, self).__init__()

        self.block_shape = block_shape
        self.in_channels = in_channels
        self.kernel_dim = kernel_dim
        # Since also here a transformation happens from (bs, c, w, h, d) -> (bs, d, h, w, c)
        # Thus for the spatial_resolution: (w, h, d) -> (d, h, w)
        spatial_resolution = spatial_resolution[::-1]
        self.spatial_resolution = spatial_resolution
        self.input_dim = kernel_dim + 2  # batch size and channel dimensions are added

        if not self.input_dim > len(self.block_shape):
            raise ValueError(
                f"Ndim of `x` ({self.input_dim}) expected to be higher than the length of"
                f" `block_shape` {len(self.block_shape)}."
            )

        if self.in_channels % math.prod(self.block_shape) != 0:
            raise ValueError(
                f"The number of channels in the input ({self.in_channels}) must be"
                f" divisible by the block size ({math.prod(self.block_shape)})."
            )

        new_spatial_resolution = [
            self.spatial_resolution[i] * self.block_shape[i]
            for i in range(len(self.spatial_resolution))
        ]
        new_spatial_resolution = tuple(new_spatial_resolution)
        self.out_channels = self.in_channels // math.prod(self.block_shape)
        self.new_shape = (-1,) + new_spatial_resolution + (self.out_channels,)

        # Further precomputation
        batch_ndim = self.input_dim - len(self.block_shape) - 1
        # Interleave old and new spatial axes.
        spatial_axes = [i for i in range(1, 2 * len(self.block_shape) + 1)]
        reshaped = [
            spatial_axes[i : i + len(self.block_shape)]
            for i in range(0, len(spatial_axes), len(self.block_shape))
        ]
        permuted = list(map(list, zip(*reshaped)))
        # flattened and spatial_axes is reshaped to column major row
        self.new_axes = tuple([item for sublist in permuted for item in sublist])

        # compute permutation axes:
        self.permutation_axes = (
            tuple(range(batch_ndim))
            + self.new_axes
            + (len(self.new_axes) + batch_ndim,)
        )

    def forward(self, inputs: Tensor) -> Tensor:

        inputs = reshape_jax_torch(inputs, self.kernel_dim)
        x = torch.reshape(
            inputs,
            (-1,)
            + self.spatial_resolution
            + tuple(self.block_shape)
            + (self.out_channels,),
        )
        x = x.permute(self.permutation_axes)
        reshaped_tensor = torch.reshape(x, self.new_shape)

        return reshape_jax_torch(reshaped_tensor, self.kernel_dim)

class UStack(nn.Module):
    """Upsampling Stack.

    Takes in features at intermediate resolutions from the downsampling stack
    as well as final output, and applies upsampling with convolutional blocks
    and combines together with skip connections in typical UNet style.
    Optionally can use self attention at low spatial resolutions.

    Attributes:
        num_channels: Number of channels at each resolution level.
        num_res_blocks: Number of resnest blocks at each resolution level.
        upsample_ratio: The upsampling ration between levels.
        padding: Type of padding for the convolutional layers.
        dropout_rate: Rate for the dropout inside the transformed blocks.
        use_attention: Whether to use attention at the coarser (deepest) level.
        num_heads: Number of attentions heads inside the attention block.
        channels_per_head: Number of channels per head.
        dtype: Data type.
    """

    def __init__(
        self,
        spatial_resolution: Sequence[int],
        emb_channels: int,
        num_channels: tuple[int, ...],
        num_res_blocks: tuple[int, ...],
        upsample_ratio: tuple[int, ...],
        use_spatial_attention: Sequence[bool],
        num_input_proj_channels: int = 128,
        num_output_proj_channels: int = 128,
        padding_method: str = "circular",
        dropout_rate: float = 0.0,
        num_heads: int = 8,
        channels_per_head: int = -1,
        normalize_qk: bool = False,
        use_position_encoding: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Any | None = None,
    ):
        super(UStack, self).__init__()

        self.kernel_dim = len(spatial_resolution)
        self.emb_channels = emb_channels
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.upsample_ratio = upsample_ratio
        self.padding_method = padding_method
        self.dropout_rate = dropout_rate
        self.use_spatial_attention = use_spatial_attention
        self.num_input_proj_channels = num_input_proj_channels
        self.num_output_proj_channels = num_output_proj_channels
        self.num_heads = num_heads
        self.channels_per_head = channels_per_head
        self.normalize_qk = normalize_qk
        self.use_position_encoding = use_position_encoding
        self.dtype = dtype
        self.device = device

        # Calculate channels for the residual block
        in_channels = []

        # calculate list of upsample resolutions
        list_upsample_resolutions = [spatial_resolution]
        for level, channel in enumerate(self.num_channels):
            downsampled_resolution = tuple(
                [
                    int(res / self.upsample_ratio[level])
                    for res in list_upsample_resolutions[-1]
                ]
            )
            list_upsample_resolutions.append(downsampled_resolution)
        list_upsample_resolutions = list_upsample_resolutions[::-1]
        list_upsample_resolutions.pop()

        self.residual_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()  # ConvBlock
        self.attention_blocks = nn.ModuleList()  # AxialSelfAttentionBlock
        self.conv_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()  # ChannelToSpace

        for level, channel in enumerate(self.num_channels):
            self.conv_blocks.append(nn.ModuleList())
            self.attention_blocks.append(nn.ModuleList())
            self.residual_blocks.append(nn.ModuleList())

            for block_id in range(self.num_res_blocks[level]):
                if block_id == 0 and level > 0:
                    in_channels.append(self.num_channels[level - 1])
                else:
                    in_channels.append(channel)

                # Residual Block
                self.residual_blocks[level].append(
                    CombineResidualWithSkip(
                        residual_channels=in_channels[-1],
                        skip_channels=channel,
                        kernel_dim=self.kernel_dim,
                        project_skip=in_channels[-1] != channel,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
                # Convolution Block
                self.conv_blocks[level].append(
                    ConvBlock(
                        in_channels=in_channels[-1],
                        out_channels=channel,
                        emb_channels=self.emb_channels,
                        kernel_size=self.kernel_dim * (3,),
                        padding_mode=self.padding_method,
                        padding=1,
                        case=self.kernel_dim,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
                # Attention Block
                if self.use_spatial_attention[level]:
                    # attention requires input shape: (bs, x, y, z, c)
                    attn_axes = [1, 2, 3]  # attention along all spatial dimensions

                    self.attention_blocks[level].append(
                        AxialSelfAttentionBlock(
                            in_channels=channel,
                            spatial_resolution=list_upsample_resolutions[level],
                            attention_axes=attn_axes,
                            add_position_embedding=self.use_position_encoding,
                            num_heads=self.num_heads,
                            normalize_qk=self.normalize_qk,
                            dtype=self.dtype,
                            device=self.device,
                        )
                    )

            # Upsampling step
            up_ratio = self.upsample_ratio[level]
            self.conv_layers.append(
                ConvLayer(
                    in_channels=channel,
                    out_channels=up_ratio**self.kernel_dim * channel,
                    kernel_size=self.kernel_dim * (3,),
                    padding_mode=self.padding_method,
                    padding=1,
                    case=self.kernel_dim,
                    kernel_init=default_init(1.0),
                    dtype=self.dtype,
                    device=self.device,
                )
            )

            self.upsample_layers.append(
                ChannelToSpace(
                    block_shape=self.kernel_dim * (up_ratio,),
                    in_channels=up_ratio**self.kernel_dim * channel,
                    kernel_dim=self.kernel_dim,
                    spatial_resolution=list_upsample_resolutions[level],
                )
            )

        # DStack Input - UStack Output Residual Connection
        self.res_skip_layer = CombineResidualWithSkip(
            residual_channels=self.num_channels[-1],
            skip_channels=self.num_input_proj_channels,
            kernel_dim=self.kernel_dim,
            project_skip=(self.num_channels[-1] != self.num_input_proj_channels),
            dtype=self.dtype,
            device=self.device,
        )

        # Add Output Layer
        self.conv_layers.append(
            ConvLayer(
                in_channels=self.num_channels[-1],
                out_channels=self.num_output_proj_channels,
                kernel_size=self.kernel_dim * (3,),
                padding_mode=self.padding_method,
                padding=1,
                case=self.kernel_dim,
                kernel_init=default_init(1.0),
                dtype=self.dtype,
                device=self.device,
            )
        )

    def forward(self, x: Tensor, emb: Tensor, skips: list[Tensor]) -> Tensor:
        assert x.ndim == 5
        assert x.shape[0] == emb.shape[0]
        assert len(self.num_channels) == len(self.num_res_blocks)
        assert len(self.upsample_ratio) == len(self.num_res_blocks)

        h = x

        for level, channel in enumerate(self.num_channels):
            for block_id in range(self.num_res_blocks[level]):
                # Residual
                h = self.residual_blocks[level][block_id](residual=h, skip=skips.pop())
                # Convolution Blocks
                h = self.conv_blocks[level][block_id](h, emb)
                # Spatial Attention Blocks
                if self.use_spatial_attention[level]:
                    h = self.attention_blocks[level][block_id](h)

            # Upsampling Block
            h = self.conv_layers[level](h)
            # Shift channels to increase the resolution, similar to torch.nn.PixelShift
            h = self.upsample_layers[level](h)

        # Output - Input Residual Connection
        h = self.res_skip_layer(residual=h, skip=skips.pop())
        # Output Layer
        h = self.conv_layers[-1](h)

        return h

class DStack(nn.Module):
    """Downsampling stack.

    Repeated convolutional blocks with occasional strides for downsampling.
    Features at different resolutions are concatenated into output to use
    for skip connections by the UStack module.
    """

    def __init__(
        self,
        in_channels: int,
        spatial_resolution: Sequence[int],
        emb_channels: int,
        num_channels: Sequence[int],
        num_res_blocks: Sequence[int],
        downsample_ratio: Sequence[int],
        use_spatial_attention: Sequence[bool],
        num_input_proj_channels: int = 128,
        padding_method: str = "circular",  # LATLON
        dropout_rate: float = 0.0,
        num_heads: int = 8,
        channels_per_head: int = -1,
        use_position_encoding: bool = False,
        normalize_qk: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Any | None = None,
    ):
        super(DStack, self).__init__()

        self.in_channels = in_channels
        self.kernel_dim = len(spatial_resolution)  # number of dimensions
        self.emb_channels = emb_channels
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.downsample_ratio = downsample_ratio
        self.padding_method = padding_method
        self.dropout_rate = dropout_rate
        self.use_spatial_attention = use_spatial_attention
        self.num_input_proj_channels = num_input_proj_channels
        self.num_heads = num_heads
        self.channels_per_head = channels_per_head
        self.use_position_encoding = use_position_encoding
        self.normalize_qk = normalize_qk
        self.dtype = dtype
        self.device = device

        # ConvLayer
        self.conv_layer = ConvLayer(
            in_channels=self.in_channels,
            out_channels=self.num_input_proj_channels,
            kernel_size=self.kernel_dim * (3,),
            padding_mode=self.padding_method,
            padding=1,
            case=self.kernel_dim,
            kernel_init=default_init(1.0),
            dtype=self.dtype,
            device=self.device,
        )

        # Input channels for the downsampling layer
        dsample_in_channels = [self.num_input_proj_channels, *self.num_channels[:-1]]
        list_spatial_resolution = [spatial_resolution]

        self.dsample_layers = nn.ModuleList()  # DownsampleConv layer
        self.conv_blocks = nn.ModuleList()  # ConvBlock
        self.attention_blocks = nn.ModuleList()  # AxialSelfAttentionBlock

        for level, channel in enumerate(self.num_channels):

            # Compute resolution after downsampling:
            downsampled_resolution = tuple(
                [
                    int(res / self.downsample_ratio[level])
                    for res in list_spatial_resolution[-1]
                ]
            )
            list_spatial_resolution.append(downsampled_resolution)

            # Downsample Layers
            self.dsample_layers.append(
                DownsampleConv(
                    in_channels=dsample_in_channels[level],
                    out_channels=channel,
                    spatial_resolution=spatial_resolution,
                    ratios=(self.downsample_ratio[level],) * self.kernel_dim,
                    kernel_init=default_init(1.0),
                    case=self.kernel_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            self.conv_blocks.append(nn.ModuleList())
            self.attention_blocks.append(nn.ModuleList())

            for block_id in range(self.num_res_blocks[level]):
                # Convblocks
                self.conv_blocks[level].append(
                    ConvBlock(
                        in_channels=channel,
                        out_channels=channel,
                        emb_channels=self.emb_channels,
                        kernel_size=self.kernel_dim * (3,),
                        padding_mode=self.padding_method,
                        padding=1,
                        case=self.kernel_dim,
                        dropout=self.dropout_rate,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )

                if self.use_spatial_attention[level]:
                    # attention requires input shape: (bs, x, y, z, c)
                    attn_axes = [1, 2, 3]  # attention along all spatial dimensions

                    self.attention_blocks[level].append(
                        AxialSelfAttentionBlock(
                            in_channels=channel,
                            spatial_resolution=list_spatial_resolution[-1],
                            attention_axes=attn_axes,
                            add_position_embedding=self.use_position_encoding,
                            num_heads=self.num_heads,
                            normalize_qk=self.normalize_qk,
                            dtype=self.dtype,
                            device=self.device,
                        )
                    )

                if block_id != 0:
                    list_spatial_resolution.append(downsampled_resolution)

    def forward(self, x: Tensor, emb: Tensor) -> list[Tensor]:
        assert x.ndim == 5
        assert x.shape[0] == emb.shape[0]
        assert len(self.num_channels) == len(self.num_res_blocks)
        assert len(self.downsample_ratio) == len(self.num_res_blocks)

        skips = []

        h = self.conv_layer(x)
        skips.append(h)

        for level, channel in enumerate(self.num_channels):
            h = self.dsample_layers[level](h)

            for block_id in range(self.num_res_blocks[level]):
                h = self.conv_blocks[level][block_id](h, emb)

                if self.use_spatial_attention[level]:
                    h = self.attention_blocks[level][block_id](h)

                skips.append(h)
        return skips

def _maybe_broadcast_to_list(
    source: bool | Sequence[bool], reference: Sequence[Any]
) -> list[bool]:
    """Broadcasts to a list with the same length if applicable."""
    if isinstance(source, bool):
        return [source] * len(reference)
    else:
        if len(source) != len(reference):
            raise ValueError(f"{source} must have the same length as {reference}!")
        return list(source)


class UNet3DGenCFDBase(nn.Module):
    """UNet model for 3D time-space input.

    This model processes 3D spatiotemporal data using a UNet architecture. It
    progressively downsamples the input for efficient feature extraction at
    multiple scales. Features are extracted using 2D spatial convolutions along
    with spatial and/or temporal axial attention blocks. Upsampling and
    combination of features across scales produce an output with the same shape as
    the input.

    Attributes:
      out_channels: Number of output channels (should match the input).
      kernel_dim: Dimension of spatial resolution. Adds info if it's a 2 or 3D dataset
      resize_to_shape: Optional input resizing shape. Facilitates greater
        downsampling flexibility. Output is resized to the original input shape.
      num_channels: Number of feature channels in intermediate convolutions.
      downsample_ratio: Spatial downsampling ratio per resolution (must evenly
        divide spatial dimensions).
      num_blocks: Number of residual convolution blocks per resolution.
      noise_embed_dim: Embedding dimensions for noise levels.
      input_proj_channels: Number of input projection channels.
      output_proj_channels: Number of output projection channels.
      padding: 2D padding type for spatial convolutions.
      dropout_rate: Dropout rate between convolution layers.
      use_spatial_attention: Whether to enable axial attention in spatial
        directions at each resolution.
      use_temporal_attention: Whether to enable axial attention in the temporal
        direction at each resolution.
      use_position_encoding: Whether to add position encoding before axial
        attention.
      num_heads: Number of attention heads.
      cond_resize_method: Resize method for channel-wise conditioning.
      cond_embed_dim: Embedding dimension for channel-wise conditioning.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_resolution: Sequence[int],
        time_cond: bool,
        num_channels: Sequence[int] = (128, 256, 256),
        downsample_ratio: Sequence[int] = (2, 2, 2),
        num_blocks: int = 4,
        noise_embed_dim: int = 128,
        input_proj_channels: int = 128,
        output_proj_channels: int = 128,
        padding_method: str = "circular",
        dropout_rate: float = 0.0,
        use_spatial_attention: bool | Sequence[bool] = (False, False, False),
        use_position_encoding: bool = True,
        num_heads: int = 8,
        normalize_qk: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        super(UNet3DGenCFDBase, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = num_channels
        self.spatial_resolution = spatial_resolution
        self.time_cond = time_cond
        self.kernel_dim = len(spatial_resolution)
        self.downsample_ratio = downsample_ratio
        self.num_blocks = num_blocks
        self.noise_embed_dim = noise_embed_dim
        self.input_proj_channels = input_proj_channels
        self.output_proj_channels = output_proj_channels
        self.padding_method = padding_method
        self.dropout_rate = dropout_rate
        self.use_spatial_attention = use_spatial_attention
        self.use_position_encoding = use_position_encoding
        self.num_heads = num_heads
        self.normalize_qk = normalize_qk
        self.device = device
        self.dtype = dtype

        self.use_spatial_attention = _maybe_broadcast_to_list(
            source=self.use_spatial_attention, reference=self.num_channels
        )

        if self.time_cond:
            self.time_embedding = FourierEmbedding(
                dims=self.noise_embed_dim, dtype=self.dtype, device=self.device
            )

        self.sigma_embedding = FourierEmbedding(
            dims=self.noise_embed_dim, dtype=self.dtype, device=self.device
        )

        self.emb_channels = (
            self.noise_embed_dim * 2 if self.time_cond else self.noise_embed_dim
        )

        self.DStack = DStack(
            in_channels=self.in_channels,
            spatial_resolution=self.spatial_resolution,
            emb_channels=self.emb_channels,
            num_channels=self.num_channels,
            num_res_blocks=len(self.num_channels) * (self.num_blocks,),
            downsample_ratio=self.downsample_ratio,
            use_spatial_attention=self.use_spatial_attention,
            num_input_proj_channels=self.input_proj_channels,
            padding_method=self.padding_method,
            dropout_rate=self.dropout_rate,
            num_heads=self.num_heads,
            use_position_encoding=self.use_position_encoding,
            normalize_qk=self.normalize_qk,
            dtype=self.dtype,
            device=self.device,
        )

        self.UStack = UStack(
            spatial_resolution=self.spatial_resolution,
            emb_channels=self.emb_channels,
            num_channels=self.num_channels[::-1],
            num_res_blocks=len(self.num_channels) * (self.num_blocks,),
            upsample_ratio=self.downsample_ratio[::-1],
            use_spatial_attention=self.use_spatial_attention[::-1],
            num_input_proj_channels=self.input_proj_channels,
            num_output_proj_channels=self.output_proj_channels,
            padding_method=self.padding_method,
            dropout_rate=self.dropout_rate,
            num_heads=self.num_heads,
            normalize_qk=self.normalize_qk,
            use_position_encoding=self.use_position_encoding,
            dtype=self.dtype,
            device=self.device,
        )

        self.norm = nn.GroupNorm(
            min(max(self.output_proj_channels // 4, 1), 32),
            self.output_proj_channels,
            device=self.device,
            dtype=self.dtype,
        )

        self.conv_layer = ConvLayer(
            in_channels=self.output_proj_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_dim * (3,),
            padding_mode=self.padding_method,
            padding=1,
            case=self.kernel_dim,
            kernel_init=default_init(),
            dtype=self.dtype,
            device=self.device,
        )

    def forward(
        self,
        x: Tensor,
        sigma: Tensor,
        time: Tensor,
    ) -> Tensor:
        """Predicts denoised given noised input and noise level.

        Args:
          x: The model input (i.e. noised sample) with shape `(batch,
            **spatial_dims, channels)`.
          sigma: The noise level, which either shares the same batch dimension as
            `x` or is a scalar (will be broadcasted accordingly).

        Returns:
          An output array with the same dimension as `x`.
        """
        if sigma.ndim < 1:
            sigma = sigma.expand(x.size(0))

        if sigma.ndim != 1 or x.shape[0] != sigma.shape[0]:
            raise ValueError(
                "`sigma` must be 1D and have the same leading (batch) dimension as x"
                f" ({x.shape[0]})!"
            )

        if time.ndim < 1:
            time = time.expand(x.size(0))

        if time.ndim != 1 or x.shape[0] != time.shape[0]:
            raise ValueError(
                "`time` must be 1D and have the same leading (batch) dimension as x"
                f" ({x.shape[0]})!"
            )

        if not x.ndim == 5:
            raise ValueError(
                "5D inputs (batch, x,y,z, features)! x.shape:" f" {x.shape}"
            )

        if len(self.num_channels) != len(self.downsample_ratio):
            raise ValueError(
                f"`num_channels` {self.num_channels} and `downsample_ratio`"
                f" {self.downsample_ratio} must have the same lengths!"
            )

        # Embedding
        emb_sigma = self.sigma_embedding(sigma)
        if self.time_cond:
            emb_time = self.time_embedding(time)
            emb = torch.cat((emb_sigma, emb_time), dim=-1)
        else:
            emb = emb_sigma

        # Downsampling
        skips = self.DStack(x, emb)

        # Upsampling
        h = self.UStack(skips[-1], emb, skips)

        h = F.silu(self.norm(h))
        h = self.conv_layer(h)

        return h


class PreconditionedDenoiser3D(UNet3DGenCFDBase, nn.Module):
    """Preconditioned 3-dimensional UNet denoising model.

    Attributes:
      sigma_data: The standard deviation of the data population. Used to derive
        the appropriate preconditioning coefficients to help ensure that the
        network deal with inputs and outputs that have zero mean and unit
        variance.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_resolution: Sequence[int],
        time_cond: bool,
        num_channels: Sequence[int] = (128, 256, 256),
        downsample_ratio: Sequence[int] = (2, 2, 2),
        num_blocks: int = 4,
        noise_embed_dim: int = 128,
        input_proj_channels: int = 128,
        output_proj_channels: int = 128,
        padding_method: str = "circular",  # LATLON
        dropout_rate: float = 0.0,
        use_spatial_attention: bool | Sequence[bool] = (False, False, False),
        use_position_encoding: bool = True,
        num_heads: int = 8,
        normalize_qk: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
        sigma_data: float = 1.0,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_resolution=spatial_resolution,
            time_cond=time_cond,
            num_channels=num_channels,
            downsample_ratio=downsample_ratio,
            num_blocks=num_blocks,
            noise_embed_dim=noise_embed_dim,
            input_proj_channels=input_proj_channels,
            output_proj_channels=output_proj_channels,
            padding_method=padding_method,
            dropout_rate=dropout_rate,
            use_spatial_attention=use_spatial_attention,
            use_position_encoding=use_position_encoding,
            num_heads=num_heads,
            normalize_qk=normalize_qk,
            dtype=dtype,
            device=device
        )

        self.sigma_data = sigma_data

    def forward(
        self,
        x: Tensor,
        sigma: Tensor,
        y: Tensor = None,
        time: Tensor = None,
        cond: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Runs preconditioned denoising."""
        if sigma.ndim < 1:
            sigma = sigma.expand(x.size(0))

        if sigma.ndim != 1 or x.size(0) != sigma.shape[0]:
            raise ValueError(
                "sigma must be 1D and have the same leading (batch) dimension as x"
                f" ({x.shape[0]})!"
            )

        total_var = self.sigma_data**2 + sigma**2
        c_skip = self.sigma_data**2 / total_var
        c_out = sigma * self.sigma_data / torch.sqrt(total_var)
        c_in = 1 / torch.sqrt(total_var)
        c_noise = 0.25 * torch.log(sigma)

        expand_shape = [-1] + [1] * (
            self.kernel_dim + 1
        )  # resolution + channel dimension
        # Expand dimensions of the coefficients
        c_in = c_in.view(*expand_shape)
        c_out = c_out.view(*expand_shape)
        c_skip = c_skip.view(*expand_shape)

        inputs = c_in * x

        if y is not None:
            # stack conditioning y
            inputs = torch.cat((inputs, y), dim=1)

        f_x = super().forward(inputs, sigma=c_noise, time=time, cond=cond)

        return c_skip * x + c_out * f_x

class UNet3DGenCFD(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 spatial_resolution: Sequence[int] = (128, 128, 128),
                 time_cond: bool = False,
                 num_channels: Sequence[int] = (128, 256, 256),
                 downsample_ratio: Sequence[int] = (2, 2, 2),
                 num_blocks: int = 4,
                 noise_embed_dim: int = 128,
                 input_proj_channels: int = 128,
                 output_proj_channels: int = 128,
                 padding_method: str = "circular",
                 dropout_rate: float = 0.0,
                 use_spatial_attention: bool | Sequence[bool] = (False, False, False),
                 use_position_encoding: bool = True,
                 num_heads: int = 8,
                 normalize_qk: bool = False,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = None,
                 ):
        super().__init__()

        self.model = UNet3DGenCFDBase(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_resolution=spatial_resolution,
            time_cond=time_cond,
            num_channels=num_channels,
            downsample_ratio=downsample_ratio,
            num_blocks=num_blocks,
            noise_embed_dim=noise_embed_dim,
            input_proj_channels=input_proj_channels,
            output_proj_channels=output_proj_channels,
            padding_method=padding_method,
            dropout_rate=dropout_rate,
            use_spatial_attention=use_spatial_attention,
            use_position_encoding=use_position_encoding,
            num_heads=num_heads,
            normalize_qk=normalize_qk,
            dtype=dtype,
            device=device
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: Optional[torch.LongTensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
    ):
        """
        Args:
            hidden_states: input tensor of shape `(batch_size, num_channels, height, width, depth)`
            timestep: timestep tensor of shape `(batch_size, 1)`
            class_labels: class label tensor of shape `(batch_size, 1)`
        """

        sigma = torch.tensor(0.0).to(hidden_states.device)
        time = torch.tensor(0.0).to(hidden_states.device)

        out = self.model.forward(hidden_states,
                                 sigma=sigma,
                                 time=time,)

        return Output3D(reconstructed=out,
                        sample=out)