from typing import List, Optional, Tuple, Union

import torch
from diffusers import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor


class MaskedImageModelingPipeline(DiffusionPipeline):
    r"""
    Pipeline for masked image modeling

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        model ([`nn.Module`]):
            A transformer model to predict masked patches
    """

    model_cpu_offload_seq = "model"

    def __init__(self, model):
        super().__init__()
        self.register_modules(model=model)

    @torch.no_grad()
    def __call__(
        self,
        images: torch.Tensor,
        bool_masked_pos: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for masked image modeling.

        Args:
            images (`torch.Tensor`):
                A tensor of shape `(batch_size, num_channels, height, width)` representing the input images.
            bool_masked_pos (`torch.Tensor`, *optional*):
                A boolean tensor of shape `(1, num_patches)` to mask patches in the image.
            output_type (`str`, *optional*, defaults to `""`):
                The output format of the generated image. Currently only `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
        ```
        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        outputs = self.model(images, bool_masked_pos=bool_masked_pos)

        image = outputs.reconstruction

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)