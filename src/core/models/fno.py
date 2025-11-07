from dataclasses import dataclass
from typing import Optional, Dict, Any

from diffusers import ModelMixin, ConfigMixin
import torch
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput

import torch.nn as nn
import torch.nn.functional as F

from neuralop.layers.spectral_convolution import SpectralConv

from neuralop.layers.padding import DomainPadding
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.mlp import MLP
from neuralop.models import FNO
from neuralop.models.base_model import BaseModel


@dataclass
class FourierNeuralOperatorOutput(BaseOutput):
    """
    The output of [`FourierNeuralOperator`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`
    """

    sample: "torch.Tensor"  # noqa: F821

def get_nonlinearity(nonlinearity: str):

    if nonlinearity == "gelu":
        return F.gelu
    else:
        raise ValueError(f"Non-linearity {nonlinearity} not supported")

def get_spectral_conv(spectral_conv: str):

        if spectral_conv == "spectral_conv":
            return SpectralConv
        else:
            raise ValueError(f"Spectral Convolution {spectral_conv} not supported")

class FourierNeuralOperator(ModelMixin, ConfigMixin):
    """N-Dimensional Fourier Neural Operator

        Parameters
        ----------
        n_modes : int tuple
            number of modes to keep in Fourier Layer, along each dimension
            The dimensionality of the TFNO is inferred from ``len(n_modes)``
        hidden_channels : int
            width of the FNO (i.e. number of channels)
        in_channels : int, optional
            Number of input channels, by default 3
        out_channels : int, optional
            Number of output channels, by default 1
        lifting_channels : int, optional
            number of hidden channels of the lifting block of the FNO, by default 256
        projection_channels : int, optional
            number of hidden channels of the projection block of the FNO, by default 256
        n_layers : int, optional
            Number of Fourier Layers, by default 4
        max_n_modes : None or int tuple, default is None
            * If not None, this allows to incrementally increase the number of
              modes in Fourier domain during training. Has to verify n <= N
              for (n, m) in zip(max_n_modes, n_modes).

            * If None, all the n_modes are used.

            This can be updated dynamically during training.
        fno_block_precision : str {'full', 'half', 'mixed'}
            if 'full', the FNO Block runs in full precision
            if 'half', the FFT, contraction, and inverse FFT run in half precision
            if 'mixed', the contraction and inverse FFT run in half precision
        stabilizer : str {'tanh'} or None, optional
            By default None, otherwise tanh is used before FFT in the FNO block
        use_mlp : bool, optional
            Whether to use an MLP layer after each FNO block, by default False
        mlp_dropout : float , optional
            droupout parameter of MLP layer, by default 0
        mlp_expansion : float, optional
            expansion parameter of MLP layer, by default 0.5
        non_linearity : nn.Module, optional
            Non-Linearity module to use, by default F.gelu
        norm : F.module, optional
            Normalization layer to use, by default None
        preactivation : bool, default is False
            if True, use resnet-style preactivation
        fno_skip : {'linear', 'identity', 'soft-gating'}, optional
            Type of skip connection to use in fno, by default 'linear'
        mlp_skip : {'linear', 'identity', 'soft-gating'}, optional
            Type of skip connection to use in mlp, by default 'soft-gating'
        separable : bool, default is False
            if True, use a depthwise separable spectral convolution
        factorization : str or None, {'tucker', 'cp', 'tt'}
            Tensor factorization of the parameters weight to use, by default None.
            * If None, a dense tensor parametrizes the Spectral convolutions
            * Otherwise, the specified tensor factorization is used.
        joint_factorization : bool, optional
            Whether all the Fourier Layers should be parametrized by a single tensor
            (vs one per layer), by default False
        rank : float or rank, optional
            Rank of the tensor factorization of the Fourier weights, by default 1.0
        fixed_rank_modes : bool, optional
            Modes to not factorize, by default False
        implementation : {'factorized', 'reconstructed'}, optional, default is 'factorized'
            If factorization is not None, forward mode to use::
            * `reconstructed` : the full weight tensor is reconstructed from the
              factorization and used for the forward pass
            * `factorized` : the input is directly contracted with the factors of
              the decomposition
        decomposition_kwargs : dict, optional, default is {}
            Optionaly additional parameters to pass to the tensor decomposition
        domain_padding : None or float, optional
            If not None, percentage of padding to use, by default None
        domain_padding_mode : {'symmetric', 'one-sided'}, optional
            How to perform domain padding, by default 'one-sided'
        fft_norm : str, optional
            by default 'forward'
        """
    @register_to_config
    def __init__(
            self,
            n_modes,
            hidden_channels,
            in_channels=3,
            out_channels=1,
            sample_size=32,
            lifting_channels=256,
            projection_channels=256,
            n_layers=4,
            output_scaling_factor=None,
            max_n_modes=None,
            fno_block_precision="full",
            use_mlp=False,
            mlp_dropout=0,
            mlp_expansion=0.5,
            non_linearity="gelu",
            stabilizer=None,
            norm=None,
            preactivation=False,
            fno_skip="linear",
            mlp_skip="soft-gating",
            separable=False,
            factorization=None,
            rank=1.0,
            joint_factorization=False,
            fixed_rank_modes=False,
            implementation="factorized",
            decomposition_kwargs=dict(),
            domain_padding=None,
            domain_padding_mode="one-sided",
            fft_norm="forward",
            spectral_conv="spectral_conv",
            **kwargs
    ):
        super().__init__()

        non_linearity = get_nonlinearity(non_linearity)
        spectral_conv = get_spectral_conv(spectral_conv)

        self.sample_size = sample_size
        self.fno = FNO(n_modes, hidden_channels, in_channels, out_channels, lifting_channels,
                       projection_channels, n_layers, output_scaling_factor, max_n_modes,
                       fno_block_precision, use_mlp, mlp_dropout, mlp_expansion, non_linearity,
                       stabilizer, norm, preactivation, fno_skip, mlp_skip, separable,
                       factorization, rank, joint_factorization, fixed_rank_modes, implementation,
                       decomposition_kwargs, domain_padding, domain_padding_mode, fft_norm,
                       spectral_conv)

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: Optional[torch.LongTensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            return_dict: bool = True,
    ):

        output = self.fno(hidden_states)

        if not return_dict:
            return (output,)

        return FourierNeuralOperatorOutput(sample=output)


### Implementation based on https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/models/fno.py ###

class FourierNeuralOperator2D(FourierNeuralOperator):
    """2D Fourier Neural Operator

    For the full list of parameters, see :class:`neuralop.models.FNO`.

    Parameters
    ----------
    n_modes_width : int
        number of modes to keep in Fourier Layer, along the width
    n_modes_height : int
        number of Fourier modes to keep along the height
    """

    def __init__(
        self,
        n_modes_height,
        n_modes_width,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        sample_size=32,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        output_scaling_factor=None,
        max_n_modes=None,
        fno_block_precision="full",
        non_linearity="gelu",
        stabilizer=None,
        use_mlp=False,
        mlp_dropout=0,
        mlp_expansion=0.5,
        norm=None,
        skip="soft-gating",
        separable=False,
        preactivation=False,
        factorization=None,
        rank=1.0,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="one-sided",
        fft_norm="forward",
        **kwargs
    ):
        super().__init__(
            n_modes=(n_modes_height, n_modes_width),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            sample_size=sample_size,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            output_scaling_factor=output_scaling_factor,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            use_mlp=use_mlp,
            mlp_dropout=mlp_dropout,
            mlp_expansion=mlp_expansion,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
            norm=norm,
            skip=skip,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization,
            rank=rank,
            joint_factorization=joint_factorization,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
            fft_norm=fft_norm,
        )
        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width

class FourierNeuralOperator3D(FourierNeuralOperator):
    """3D Fourier Neural Operator

    For the full list of parameters, see :class:`neuralop.models.FNO`.

    Parameters
    ----------
    n_modes_width : int
        number of modes to keep in Fourier Layer, along the width
    n_modes_height : int
        number of Fourier modes to keep along the height
    """

    def __init__(
        self,
        n_modes_height,
        n_modes_width,
        n_modes_depth,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        sample_size=32,
        lifting_channels=6,
        projection_channels=6,
        n_layers=2,
        output_scaling_factor=None,
        max_n_modes=None,
        fno_block_precision="full",
        non_linearity="gelu",
        stabilizer=None,
        use_mlp=False,
        mlp_dropout=0,
        mlp_expansion=0.5,
        norm=None,
        skip="soft-gating",
        separable=False,
        preactivation=False,
        factorization=None,
        rank=1.0,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="one-sided",
        fft_norm="forward",
        **kwargs
    ):
        super().__init__(
            n_modes=(n_modes_height, n_modes_width, n_modes_depth),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            sample_size=sample_size,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            output_scaling_factor=output_scaling_factor,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            use_mlp=use_mlp,
            mlp_dropout=mlp_dropout,
            mlp_expansion=mlp_expansion,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
            norm=norm,
            skip=skip,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization,
            rank=rank,
            joint_factorization=joint_factorization,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
            fft_norm=fft_norm,
        )
        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width
        self.n_modes_depth = n_modes_depth