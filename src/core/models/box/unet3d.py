from typing import Dict, Optional, Any, Sequence, Union
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings

from src.core.models.box.output3d import Output3D
from src.core.models.box.pixelshuffle import PixelShuffle3d

import torch.nn as nn
import torch

class ConditionedEncoder3DBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 num_groups: int = 32,):
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
                 feature_embedding_dim: Union[int, Sequence[int]],
                 num_downsampling_layers: int,
                 embedding_dim: int,
                 repetitions: int = 1,
                 num_groups: int = 32):
        super().__init__()
        self.in_channels = in_channels

        if isinstance(feature_embedding_dim, Sequence):
            self.feature_embedding_dim = feature_embedding_dim
        else:
            self.feature_embedding_dim = [feature_embedding_dim * 2 ** i
                                          for i in range(num_downsampling_layers+1)]

        self.repetitions = repetitions
        self.num_downsampling_layers = num_downsampling_layers
        self.embedding_dim = embedding_dim
        self.feature_embed = nn.Conv3d(in_channels, self.feature_embedding_dim[0], 3, 1, 1)
        self.downsampling_layers = nn.ModuleList()
        for i in range(num_downsampling_layers):
            self.downsampling_layers.append(
                nn.Conv3d(self.feature_embedding_dim[i], self.feature_embedding_dim[i+1], 3, 2, 1)
            )
        self.blocks = nn.ModuleList()
        for i in range(num_downsampling_layers - 1):
            self.blocks.extend([
                ConditionedEncoder3DBlock(self.feature_embedding_dim[i+1], embedding_dim,
                                          num_groups=num_groups) for _ in range(repetitions)]
            )


    def forward(self, x, embedding):

        x = self.feature_embed(x)

        res_list = [x]

        x = self.downsampling_layers[0](x)

        for i in range(self.num_downsampling_layers - 1):
            for j in range(self.repetitions):
                x = self.blocks[i*self.repetitions+j](x, embedding)
            res_list.append(x)
            x = self.downsampling_layers[i+1](x)

        res_list.append(x)

        return res_list

ConditionedDecoder3DBlock = ConditionedEncoder3DBlock

class DecoderUpsamplingBlock(nn.Module):

        def __init__(self,
                    in_channels: int,
                    out_channels: int,
                    factor: Optional[int] = None):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels

            if factor is None:
                factor = int((out_channels / in_channels) * 8)
            else:
                factor = 4

            self.linear_conv = nn.Conv3d(in_channels, out_channels * 8, 1)
            self.shuffle = PixelShuffle3d(2)

        def forward(self, x):
            x = self.linear_conv(x)
            x = self.shuffle(x)
            return x

class ConditionedDecoder3D(nn.Module):

        def __init__(self,
                    out_channels: int,
                    feature_embedding_dim: Union[int, Sequence[int]],
                    num_upsampling_layers: int,
                    embedding_dim: int,
                    repetitions: int = 1,
                    features_first_layer: int = None,
                    upsample_factor: Optional[int] = None,
                    num_groups: int = 32):
            super().__init__()
            self.out_channels = out_channels

            if isinstance(feature_embedding_dim, Sequence):
                self.feature_embedding_dim = feature_embedding_dim
            else:
                self.feature_embedding_dim = [feature_embedding_dim * 2 ** (num_upsampling_layers - i)
                                              for i in range(num_upsampling_layers+1)]

            self.num_upsampling_layers = num_upsampling_layers
            self.embedding_dim = embedding_dim
            self.repetitions = repetitions

            self.decompress = nn.Conv3d(self.feature_embedding_dim[-1], out_channels, 3, 1, 1)

            self.blocks = nn.ModuleList()
            for i in range(num_upsampling_layers - 1):
                self.blocks.extend(
                    [ConditionedDecoder3DBlock(self.feature_embedding_dim[i + 1], embedding_dim,
                                              num_groups=num_groups) for _ in range(self.repetitions)]
                )


            if features_first_layer is None:
                features_first_layer = feature_embedding_dim

            self.upsampling_layers = nn.ModuleList()

            local_feature_dim = self.feature_embedding_dim[1]
            self.upsampling_layers.append(
                DecoderUpsamplingBlock(features_first_layer, local_feature_dim,
                                       factor=upsample_factor)
            )
            for i in range(num_upsampling_layers-1):
                local_feature_dim = self.feature_embedding_dim[i + 1]
                self.upsampling_layers.append(
                    DecoderUpsamplingBlock(local_feature_dim,
                                           self.feature_embedding_dim[i + 2],
                                           factor=upsample_factor)
                )

        def forward(self, x, embedding, encoder_outputs):

            x = self.upsampling_layers[0](x)
            x += encoder_outputs[::-1][1]

            for i in range(self.num_upsampling_layers - 1):
                for j in range(self.repetitions):
                    x = self.blocks[i*self.repetitions+j](x, embedding)
                x = self.upsampling_layers[i+1](x)
                x += encoder_outputs[::-1][i+2]

            x = self.decompress(x)

            return x

def UNet3D_S(image_size: int, in_channels: int, out_channels: Optional[int] = None):

    if out_channels is None:
        out_channels = in_channels

    return UNet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_embedding_dim=72,
        num_downsampling_layers=4,
        time_embedding_dim=64,
        num_groups=24,
        num_classes=1000,
        class_dropout_prob=0.1,
        repetitions=2,
    )

def UNet3D_M(image_size: int, in_channels: int, out_channels: Optional[int] = None):

    if out_channels is None:
        out_channels = in_channels

    return UNet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_embedding_dim=96,
        num_downsampling_layers=4,
        time_embedding_dim=96,
        num_groups=32,
        num_classes=1000,
        class_dropout_prob=0.1,
        repetitions=2
    )

def UNet3D_L(image_size: int, in_channels: int, out_channels: Optional[int] = None):

    if out_channels is None:
        out_channels = in_channels

    return UNet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_embedding_dim=128,
        num_downsampling_layers=4,
        time_embedding_dim=96,
        num_groups=32,
        num_classes=1000,
        class_dropout_prob=0.1,
        repetitions=3
    )

class UNet3D(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 feature_embedding_dim: int = 64,
                 num_downsampling_layers: int = 3,
                 time_embedding_dim: int = 64,
                 num_groups: int = 32,
                 num_classes: int = 1000,
                 repetitions: int = 2,
                 class_dropout_prob: float = 0.1):
        super().__init__()

        # self.enable_gradient_checkpointing()

        self.feature_embedding_dim = feature_embedding_dim
        self.time_embedding_dim = time_embedding_dim

        self.class_embedding = CombinedTimestepLabelEmbeddings(
            num_classes=num_classes, embedding_dim=time_embedding_dim, class_dropout_prob=class_dropout_prob)

        self.encoder = ConditionedEncoder3D(
            in_channels=in_channels,
            feature_embedding_dim=feature_embedding_dim,
            num_downsampling_layers=num_downsampling_layers,
            embedding_dim=time_embedding_dim,
            repetitions=repetitions,
            num_groups=num_groups,
        )

        self.latent_size = feature_embedding_dim * 2 ** num_downsampling_layers

        self.decoder = ConditionedDecoder3D(
            out_channels=out_channels,
            feature_embedding_dim=feature_embedding_dim,
            num_upsampling_layers=num_downsampling_layers,
            embedding_dim=time_embedding_dim,
            features_first_layer=feature_embedding_dim * 2 ** num_downsampling_layers,
            repetitions=repetitions,
            num_groups=num_groups,
        )

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

        if class_labels is None:
            class_labels = torch.Tensor([0]).to(x.device).long().repeat(x.shape[0])

        if len(class_labels.shape) == 0:
            class_labels = class_labels.unsqueeze(0)
            class_labels = class_labels.repeat(x.shape[0])
            class_labels = class_labels.to(x.device)

        emb = self.class_embedding(timestep, class_labels)
        output = self.encoder(x, emb)

        return Output3D(hidden_states=output, embedding=emb)

    def decode(self, x, embedding, residuals):

        if isinstance(embedding, Sequence):
            embedding = embedding[-1]

        reconstructed = self.decoder(x, embedding, residuals)
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
            pde_parameters: pde parameters tensor of shape `(batch_size, 1)` Currently not used
        """

        encoded = self.encode(hidden_states, timestep, class_labels, pde_parameters)

        state = encoded.hidden_states[-1]
        emb = encoded.embedding

        out = self.decoder(state, emb, encoded.hidden_states)

        return Output3D(reconstructed=out,
                        sample=out,
                        embedding=emb)