from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from diffusers import ModelMixin, ConfigMixin

import torch
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import PatchEmbed, get_timestep_embedding
from diffusers.utils import is_torch_version, BaseOutput
import torch.nn.functional as F

class MaskEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim=16,
        **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj1 = nn.Embedding(2, embed_dim)

    def forward(self, mask):
        mask = self.proj1(mask)
        return mask

class TokenEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        id_embed_dim=64,
        max_id_embed=1000,
        **kwargs
    ):
        super().__init__()

        self.id_embed = nn.Embedding(max_id_embed, id_embed_dim)
        self.proj1 = nn.Linear(2, embed_dim // 2)
        self.embed_dim = embed_dim
        self.id_embed_dim = id_embed_dim

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.transpose(1, 2)
        x = x.reshape(-1, 2)

        u = self.proj1(x)
        v = get_timestep_embedding(x[:, 1], self.embed_dim // 2)
        i = self.id_embed(x[:, 0].long())

        x = torch.cat([i, u, v], dim=-1)

        x = x.reshape(batch_size, -1, self.embed_dim + self.id_embed_dim)

        return x

class ConditionedPatchEmbedding(nn.Module):
    def __init__(
        self,
        height=160,
        width=160,
        patch_size=4,
        in_channels=1,
        embed_dim=768,
        id_embed_dim=32,
        time_embed_dim=32,
        max_id_embed=1000,
        **kwargs
    ):
        super().__init__()

        self.pos_embed = PatchEmbed(
            height=height,
            width=width,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            **kwargs
        )
        self.id_embed_dim = id_embed_dim
        self.id_embed = nn.Embedding(max_id_embed, id_embed_dim)

        self.time_embed_dim = time_embed_dim

    def forward(self, x, idx, t):
        x = self.pos_embed(x)
        idx = self.id_embed(idx)
        idx = idx.unsqueeze(1).expand(-1, x.shape[1], self.id_embed_dim)

        t = get_timestep_embedding(t, self.time_embed_dim)
        t = t.unsqueeze(1).expand(-1, x.shape[1], self.time_embed_dim)

        x = torch.cat((x, idx, t), dim=-1)
        return x

@dataclass
class MaskedDiTTransformerModelOutput(BaseOutput):
    """
    The output of [`MaskedDiTTransformerModel`].

    Args:
        images (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
        tokens (`torch.Tensor` of shape `(batch_size, num_tokens, num_channels)`):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
    """

    images: "List[torch.Tensor]"  # noqa: F821
    tokens: "torch.Tensor"  # noqa: F821

class TokenReadout(nn.Module):

    def __init__(self, embed_dim=768, out_dim=1):
        super().__init__()
        self.proj1 = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        return self.proj1(x)

class MaskedDiTTransformerModel(ModelMixin, ConfigMixin):
    r"""
    A MaskedDiTTransformer model (modified from DiT (https://arxiv.org/abs/2212.09748)).

    Parameters:
        num_attention_heads (int, optional, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (int, optional, defaults to 72): The number of channels in each head.
        in_channels (int, defaults to 4): The number of channels in the input.
        out_channels (int, optional):
            The number of channels in the output. Specify this parameter if the output channel number differs from the
            input.
        num_layers (int, optional, defaults to 28): The number of layers of Transformer blocks to use.
        dropout (float, optional, defaults to 0.0): The dropout probability to use within the Transformer blocks.
        norm_num_groups (int, optional, defaults to 32):
            Number of groups for group normalization within Transformer blocks.
        attention_bias (bool, optional, defaults to True):
            Configure if the Transformer blocks' attention should contain a bias parameter.
        sample_size (int, defaults to 32):
            The width of the latent images. This parameter is fixed during training.
        patch_size (int, defaults to 2):
            Size of the patches the model processes, relevant for architectures working on non-sequential data.
        activation_fn (str, optional, defaults to "gelu-approximate"):
            Activation function to use in feed-forward networks within Transformer blocks.
        num_embeds_ada_norm (int, optional, defaults to 1000):
            Number of embeddings for AdaLayerNorm, fixed during training and affects the maximum denoising steps during
            inference.
        upcast_attention (bool, optional, defaults to False):
            If true, upcasts the attention mechanism dimensions for potentially improved performance.
        norm_type (str, optional, defaults to "ada_norm_zero"):
            Specifies the type of normalization used, can be 'ada_norm_zero'.
        norm_elementwise_affine (bool, optional, defaults to False):
            If true, enables element-wise affine parameters in the normalization layers.
        norm_eps (float, optional, defaults to 1e-5):
            A small constant added to the denominator in normalization layers to prevent division by zero.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        in_channels: int = 1,
        out_channels: Optional[int] = None,
        embed_dim: int = 512,
        id_embed_dim: int = 32,
        time_embed_dim: int = 32,
        max_id_embed: int = 1000,
        mask_embed_dim: int = 16,
        num_layers: int = 28,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        attention_bias: bool = True,
        sample_size: int = 32,
        patch_size: int = 2,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_zero",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        # Validate inputs.
        if norm_type != "ada_norm_zero":
            raise NotImplementedError(
                f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
            )
        elif norm_type == "ada_norm_zero" and num_embeds_ada_norm is None:
            raise ValueError(
                f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
            )

        # Set some common variables used across the board.
        self.attention_head_dim = attention_head_dim


        self.inner_dim = self.mask_embed_dim + self.time_embed_dim + self.id_embed_dim + self.embed_dim

        # self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = False

        # 2. Initialize the position embedding and transformer blocks.
        self.height = self.config.sample_size
        self.width = self.config.sample_size

        self.id_embed_dim = id_embed_dim
        self.max_id_embed = max_id_embed
        self.embed_dim = embed_dim
        self.mask_embed_dim = mask_embed_dim

        self.time_embed_dim = time_embed_dim
        self.patch_size = patch_size

        self.patch_embed = ConditionedPatchEmbedding(
            height=self.height,
            width=self.width,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            id_embed_dim=self.id_embed_dim,
            time_embed_dim=self.time_embed_dim,
            max_id_embed=self.max_id_embed,
        )

        self.token_embed = TokenEmbedding(
            embed_dim= self.embed_dim,
            id_embed_dim= self.id_embed_dim + self.time_embed_dim, # we add the time_embed dim here to account for the time embed in the patch embed
            max_id_embed=self.max_id_embed,
        )

        self.mask_embed = MaskEmbedding(
            embed_dim=self.mask_embed_dim,
        )

        self.read_tokens = TokenReadout(
            embed_dim=self.patch_size * self.patch_size * self.out_channels,
            out_dim=1,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # 3. Output blocks.
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(
            self.inner_dim, self.config.patch_size * self.config.patch_size * self.out_channels
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        images: List[torch.Tensor],
        images_idx: List[torch.Tensor],
        images_time: List[torch.Tensor],
        mask: torch.Tensor,
        tokens: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        return_dict: bool = True,
    ):
        """
        The [`DiTTransformer2DModel`] forward method.

        Args:
            images ( `List[torch.Tensor]`):
                The input image tensor.
            images_idx ( `List[torch.Tensor]`):
                The input image index tensor.
            mask ( `List[torch.Tensor]`):
                The input mask tensor.
            tokens ( `torch.Tensor`):
                The input tokens tensor.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        if class_labels is None:
            class_labels = torch.zeros(tokens.shape[0], dtype=torch.long)
            class_labels = class_labels.to(tokens.device)

        # 1. Input
        heights, widths = [img.shape[-2] // self.patch_size for img in images], [img.shape[-1] // self.patch_size for img in images]

        hidden_states = []
        for img, img_id, img_time in zip(images, images_idx, images_time):
            hidden_states.append(self.patch_embed(img, img_id, img_time))

        hidden_states = torch.concatenate(hidden_states, dim=1)

        tokens = (self.token_embed(tokens))

        hidden_states = torch.cat((hidden_states, tokens), dim=1)

        mask = self.mask_embed(mask.long())

        hidden_states = torch.cat((mask, hidden_states), dim=-1)

        # 2. Blocks
        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    None,
                    None,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )

        # 3. Output
        conditioning = self.transformer_blocks[0].norm1.emb(timestep, class_labels, hidden_dtype=hidden_states.dtype)
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        hidden_states = self.proj_out_2(hidden_states)

        images = []
        # unpatchify
        for height, width in zip(heights, widths):
            img_tokens = hidden_states[:, :height * width]
            hidden_states = hidden_states[:, height * width:]
            img_tokens = img_tokens.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            img_tokens = torch.einsum("nhwpqc->nchpwq", img_tokens)
            img_tokens = img_tokens.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )
            images.append(img_tokens)

        tokens = hidden_states
        tokens = self.read_tokens(tokens)

        if not return_dict:
            return images, tokens

        return MaskedDiTTransformerModelOutput(images = images, tokens = tokens)