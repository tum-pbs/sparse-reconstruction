from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import STEP_OUTPUT
from sympy.stats.rv import probability
from tensorboard.summary.v1 import image
from timm.scheduler import TanhLRScheduler
from torch.nn.functional import unfold
from tqdm import tqdm

from core.masked_modeling_v1.tasks import TaskMasking
from src.utils import instantiate_from_config

import numpy as np

def accumulate_embeddings_and_values(*args):

    embedding_list = []
    value_list = []

    for dict_ in args:
        for key, value in dict_.items():

            embedding_list.append(value['embedding'])
            value_list.append(value['value'])

    return embedding_list, value_list

def get_weighting_function(weighting: str):
    """
    Get weighting function for the loss
    :param weighting: type
    :return:
    """

    def identity(t, w):
        return w

    def constant(t, w):
        return torch.ones_like(t)

    if weighting == "id":
        return identity
    elif weighting == "constant":
        return constant
    else:
        raise ValueError(f"Weighting function {weighting} not implemented")

def sample_time(device, num: int = 1, t0: float = 0.0,
                t1: float = 1.0, eps=1e-5) -> torch.Tensor:
    """
    Sample time points between t0 and t1 based on batch size for variance reduction
    :param device: device
    :param num: batch size
    :param t0: minimum time
    :param t1: maximum time
    :param eps: make sure to not sample exactly at t0 or t1
    :return:
    """

    t0 += eps
    t1 -= eps
    times = torch.linspace(t0, t1, num+1, device=device)[:num]
    uniform = torch.rand((num,), device=device)
    uniform = uniform * ((t1 - t0) / num)
    time_samples = times + uniform

    return time_samples

def to_patch_representation(images: torch.Tensor, patch_size: int = 16):
    batch_size, channels, height, width = images.shape
    kh, kw = patch_size, patch_size  # kernel size
    dh, dw = patch_size, patch_size  # stride
    patches = images.unfold(2, kh, dh).unfold(3, kw, dw)
    patches = patches.contiguous().view(batch_size, -1, channels * kh * kw)

    return patches


def from_patch_representation(patches: torch.Tensor, height: int, width: int, patch_size: int = 16):
    batch_size = patches.shape[0]
    unfold_shape = (batch_size, -1, height // patch_size, width // patch_size, patch_size, patch_size)
    patches_orig = patches.view(unfold_shape)
    output_h = unfold_shape[2] * unfold_shape[4]
    output_w = unfold_shape[3] * unfold_shape[5]
    patches_orig = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
    patches_orig = patches_orig.view(batch_size, -1, output_h, output_w)

    return patches_orig


def get_masked_model_inputs(images: list[torch.tensor] , tokens: torch.tensor,
                            mask_images: list[torch.tensor], mask_tokens: torch.tensor,
                            t: torch.tensor, sigma_min: float, patch_size: int) -> Tuple:

    tokens_id = tokens[:, 0]
    tokens_value = tokens[:, 1]

    heights = [img.shape[2] for img in images]
    widths = [img.shape[3] for img in images]

    epsilon_images = [torch.randn(size=img.shape, device=img.device, dtype=img.dtype)
                        for img in images]
    epsilon_tokens = torch.randn(size=tokens_value.shape, device=tokens_value.device, dtype=tokens_value.dtype)

    psi_t = (1 - (1 - sigma_min) * t)
    psi_t_images = [torch.einsum("a,abcd->abcd", psi_t, epsilon_img) for epsilon_img
                    in epsilon_images]

    psi_t_images = [psi_t_img + torch.einsum("a, abcd->abcd", t, img)
                    for psi_t_img, img in zip(psi_t_images, images)]

    psi_t_images = [to_patch_representation(psi_t_img, patch_size=patch_size)
                    for psi_t_img in psi_t_images]

    images = [to_patch_representation(img, patch_size=patch_size) for img in images]

    psi_t_images = [(1 - mask_img) * psi_t_img + mask_img * img for psi_t_img, mask_img, img in
                    zip(psi_t_images, mask_images, images)]

    psi_t_images = [from_patch_representation(psi_t_img, height=height, width=width, patch_size=patch_size)
                    for psi_t_img, height, width in zip(psi_t_images, heights, widths)]

    psi_t_tokens = torch.einsum("a,ab->ab", psi_t, epsilon_tokens)
    psi_t_tokens = psi_t_tokens + torch.einsum("a, ab->ab", t, tokens_value)

    psi_t_tokens = (1 - mask_tokens) * psi_t_tokens + mask_tokens * tokens_value

    psi_t_tokens = torch.stack([tokens_id, psi_t_tokens], dim=1)

    return psi_t_images, psi_t_tokens, epsilon_images, epsilon_tokens

class MaskedFlowMatching(nn.Module):
    """
    Objective for conditional optimal transport matching the flow.
    See Lipman et al. 2023 "Flow Matching for Generative Modeling" for details.
    """

    def __init__(self, sigma_min: float = 1e-4, weighting: str = 'constant', patch_size: int = 4):

        super().__init__()

        self.patch_size = patch_size
        self.sigma_min = sigma_min
        self.weighting_function = get_weighting_function(weighting)
        self.start_time = 0.0
        self.end_time = 1.0


    def loss(self, model: nn.Module, batch: dict, **kwargs) -> Tuple[torch.Tensor, Dict]:

        channels = [img.unsqueeze(1) for img in batch['channels']] # add channel dim 1
        channels_idx = batch['channel_ids']
        channels_time = batch['simulation_times']
        mask = batch['mask']

        tokens = batch['tokens']
        tokens_value = tokens[:, 1]

        num_channels = [img.shape[1] for img in channels]
        height = [img.shape[2] for img in channels]
        width = [img.shape[3] for img in channels]

        t = sample_time(tokens.device, tokens.shape[0], t0=self.start_time,
                        t1=self.end_time)

        num_image_tokens = [height_ * width_ // (self.patch_size ** 2) for height_, width_ in zip(height, width)]

        num_image_tokens_cumsum = np.cumsum(num_image_tokens)

        mask_images = [ mask[:, start:end] for start, end in zip([0] + num_image_tokens_cumsum[:-1].tolist(),
                                                                 num_image_tokens_cumsum) ]

        mask_images = [ m.unsqueeze(2).repeat(1, 1, channels_ * self.patch_size * self.patch_size) for m, channels_ in
                        zip(mask_images, num_channels) ]

        mask_tokens = mask[:, num_image_tokens_cumsum[-1]:]

        psi_t_images, psi_t_tokens, epsilon_images, epsilon_tokens = (
            get_masked_model_inputs(channels, tokens, mask_images, mask_tokens,
                                    t, self.sigma_min, self.patch_size))

        target_images = [(img - (1 - self.sigma_min) * epsilon_img) for img, epsilon_img
                         in zip(channels, epsilon_images)]
        target_images = [to_patch_representation(img, patch_size=self.patch_size) for img in target_images]

        target_tokens = (tokens_value - (1 - self.sigma_min) * epsilon_tokens)

        prediction = model.forward(psi_t_images, channels_idx, channels_time, mask, psi_t_tokens, t)
        prediction_images = prediction.images
        prediction_tokens = prediction.tokens[:,:,0]

        prediction_images = [to_patch_representation(img, patch_size=self.patch_size)
                             for img in prediction_images]

        prediction_images = [(1 - mask_img) * prediction_img + mask_img * target_img
                             for prediction_img, mask_img, target_img in
                             zip(prediction_images, mask_images, target_images)]

        prediction_tokens = (1 - mask_tokens) * prediction_tokens + mask_tokens * target_tokens

        loss_images = [torch.mean((target_img - prediction_img) ** 2, dim=[1,2]) for target_img, prediction_img in
                       zip(target_images, prediction_images)]
        loss_images = torch.stack(loss_images, dim=1) # [batch_size, num_images]
        loss_images = loss_images.mean(dim=1)

        loss_tokens = torch.mean((target_tokens - prediction_tokens) ** 2, dim=[1])

        loss_tokens = torch.nan_to_num(loss_tokens, nan=0.0)

        loss = loss_images + loss_tokens
        loss = loss.mean()

        return loss, {}

import lightning

class FlowMatching(lightning.LightningModule):

    def __init__(self, model: dict,
                 tasks: list[str],
                 ckpt_path=None,
                 ignore_keys=None,
                 monitor: str = "val/loss_epoch",
                 frequency: int = 500,
                 steps: int = 20,
                 weighting: str = 'uniform',
                 patch_size: int = 4):

        super().__init__()

        self.monitor = monitor
        self.frequency = frequency
        self.patch_size = patch_size
        self.model = instantiate_from_config(model)

        self.fm = MaskedFlowMatching(patch_size=patch_size)
        self.steps = steps

        self.task_masking = TaskMasking(tasks, weighting, patch_size=patch_size)
        self.save_hyperparameters()

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path: str, ignore_keys:list[str]=None):

        if ignore_keys is None:
            ignore_keys = list()

        if Path(path).is_dir():
            path = Path(path).joinpath("last.ckpt")
        else:
            path = Path(path)

        if path.is_file():
            sd = torch.load(path, map_location="cpu")["state_dict"]
            keys = list(sd.keys())
            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        print("Deleting key {} from state_dict.".format(k))
                        del sd[k]
            self.load_state_dict(sd, strict=False)
            print(f"Restored from {path}")

    def forward(self, images: list[torch.Tensor],
                images_idx: list[torch.Tensor],
                masks: torch.Tensor,
                simulation_times: torch.Tensor,
                tokens: torch.Tensor,
                timestep: Optional[torch.Tensor],
                class_labels: Optional[torch.LongTensor]) -> torch.Tensor:

        return self.model(images, images_idx, masks,
                          tokens, timestep, class_labels)

    def shared_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:

        data = self.task_masking.get_inputs(batch, batch_idx)

        loss, _ = self.fm.loss(self.model, data)

        return loss

    def training_step(self, batch, batch_idx):

        loss = self.shared_step(batch, batch_idx)

        self.log("loss", loss.item(), prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):

        loss = self.shared_step(batch, batch_idx)

        self.log("val/loss", loss.item(), prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return {"val/loss": loss}

    def configure_optimizers(self):

        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = TanhLRScheduler(opt, t_initial=self.frequency, cycle_limit=10000)

        return [opt], [] # [{"scheduler": scheduler, "interval": "epoch"}]

    def _predict(self, mask, images, images_idx, images_times, tokens, DT=0.01):

        images = [img.unsqueeze(1) for img in images]  # add channel dim 1
        batch_size = mask.shape[0]
        T = torch.zeros(batch_size).to(tokens.device)

        num_channels = [img.shape[1] for img in images]
        heights = [img.shape[2] for img in images]
        widths = [img.shape[3] for img in images]

        num_image_tokens = [height_ * width_ // (self.patch_size ** 2) for height_, width_ in zip(heights, widths)] # noqa

        num_image_tokens_cumsum = np.cumsum(num_image_tokens)

        mask_images = [mask[:, start:end] for start, end in zip([0] + num_image_tokens_cumsum[:-1].tolist(),
                                                                num_image_tokens_cumsum)]

        mask_images = [m.unsqueeze(2).repeat(1, 1, channels_ * self.patch_size * self.patch_size) for m, channels_ in
                       zip(mask_images, num_channels)]

        mask_tokens = mask[:, num_image_tokens_cumsum[-1]:]

        psi_t_images, psi_t_tokens, _, _ = get_masked_model_inputs(images, tokens, mask_images, mask_tokens, T,
                                                                   self.fm.sigma_min, self.patch_size)

        mask = torch.concatenate([mask_img[:, :, 0] for mask_img in mask_images] + [mask_tokens], dim=1)

        with torch.no_grad():
            for _ in range(int(1 / DT)):
                output = self.model.forward(psi_t_images, images_idx, images_times, mask, psi_t_tokens, T, class_labels=None)
                token_out = output.tokens
                images_out = output.images

                psi_t_images_patch = [to_patch_representation(psi_t_img, patch_size=self.patch_size) for psi_t_img in
                                      psi_t_images]
                images_out_patch = [to_patch_representation(img_out, patch_size=self.patch_size) for img_out in images_out]

                psi_t_images_patch = [psi_t_img_patch + DT * (1 - mask_img) * img_out_patch for
                                      psi_t_img_patch, mask_img, img_out_patch in
                                      zip(psi_t_images_patch, mask_images, images_out_patch)]
                psi_t_images = [
                    from_patch_representation(psi_t_img_patch, height=height, width=width, patch_size=self.patch_size) for
                    psi_t_img_patch, height, width in zip(psi_t_images_patch, heights, widths)]

                psi_t_tokens[:, -1] = psi_t_tokens[:, -1] + DT * (1 - mask_tokens) * token_out[:, :, 0]

                T += DT

        return psi_t_images, psi_t_tokens

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:

        return