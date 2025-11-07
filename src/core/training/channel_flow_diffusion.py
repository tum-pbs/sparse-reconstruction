from pathlib import Path
from typing import List

from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

from sampler.scheduler import OdeEulerScheduler
from src.core.generative_model_2d import GenerativeModel2D
from src.objectives import TrainingObjective

from src.utils import instantiate_from_config
import torch.nn as nn
import lightning
import torch

CHANNEL_0_MEAN = 0.5034929
CHANNEL_0_STD = 0.22558229

CHANNEL_1_MEAN = -6.7011285e-07
CHANNEL_1_STD = 0.022445366

CHANNEL_2_MEAN = 0.00015084329
CHANNEL_2_STD = 0.034942694

CHANNEL_3_MEAN = 1.7268538e-12
CHANNEL_3_STD = 0.002972702

class ChannelFlowDiffusion(lightning.LightningModule):
    def __init__(self,
                 model,
                 objective,
                 ckpt_path=None,
                 ignore_keys=None,
                 image_key=0,
                 monitor=None,
                 downsample_factor: int = 1,
                 optimizer='adamw',
                 training_mode='train',
                 use_deformation_tensor: bool = True,
                 ):
        """
        Single step diffusion model for generative modeling.
        Args:
            model: model configuration
            objective: objective configuration
            ckpt_path: path to checkpoint
            ignore_keys: keys to ignore in checkpoint
            image_key: key for image in batch
            monitor: monitor configuration
            downsample_factor: downsample factor for input
        """

        super().__init__()

        print("Channel Flow diffusion training mode: ", training_mode)
        self.training_mode = training_mode
        self.image_key = image_key
        self.optimizer = optimizer
        self.use_deformation_tensor = use_deformation_tensor

        # Score model
        self.model: nn.Module = instantiate_from_config(model)

        # Objective
        self.objective: TrainingObjective = (
            instantiate_from_config(objective))

        self.downsample_factor = downsample_factor

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.scheduler = OdeEulerScheduler(num_train_timesteps=100)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                **kwargs) -> torch.Tensor:

        return self.model(x, t, **kwargs)

    def get_pipeline_args(self):
        return {
            "unet": self.model,
        }

    # Useful function for getting the error https://github.com/Lightning-AI/pytorch-lightning/issues/17212
    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is None:
                print(name)

    def get_input(self, batch, batch_dim=True):

        data: torch.Tensor = batch["data"]
        meta_data_loading: dict = batch["loading_metadata"]
        meta_data_physical: dict = batch["physical_metadata"]
        constants: dict = batch["constants"]

        if batch_dim:
            x: torch.Tensor = data[:, 0]
            y: torch.Tensor = data[:, 1:]

            task_idx = meta_data_physical['PDE'][:, 0]
            re = constants[:, 0]

        else:
            x: torch.Tensor = data[0]
            y: torch.Tensor = data[1:]
            task_idx = meta_data_physical['PDE']
            re = constants[0]

            x = torch.unsqueeze(x, 0)
            y = torch.unsqueeze(y, 0)

            if not torch.is_tensor(task_idx):
                task_idx = torch.tensor(task_idx)

            if not torch.is_tensor(re):
                re = torch.tensor(re)

        re = re.long()

        return x, y, task_idx, re

    def normalize(self, x):

        # normalize each channel
        x[:, 0] = (x[:, 0] - CHANNEL_0_MEAN) / CHANNEL_0_STD
        x[:, 1] = (x[:, 1] - CHANNEL_1_MEAN) / CHANNEL_1_STD
        x[:, 2] = (x[:, 2] - CHANNEL_2_MEAN) / CHANNEL_2_STD
        x[:, 3] = (x[:, 3] - CHANNEL_3_MEAN) / CHANNEL_3_STD

        return x

    def denormalize(self, x):

        # denormalize each channel
        x[:, 0] = x[:, 0] * CHANNEL_0_STD + CHANNEL_0_MEAN
        x[:, 1] = x[:, 1] * CHANNEL_1_STD + CHANNEL_1_MEAN
        x[:, 2] = x[:, 2] * CHANNEL_2_STD + CHANNEL_2_MEAN
        x[:, 3] = x[:, 3] * CHANNEL_3_STD + CHANNEL_3_MEAN

        return x

    def test_step(self, batch, batch_idx):

        return 0, {}

    def init_from_ckpt(self, path: str, ignore_keys: List[str]=None):
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

    def training_step(self, batch, batch_idx):

        input_0, input_1, labels, re = self.get_input(batch)

        # input_0[:,:4]: velocity x, velocity y, velocity z, pressure -> generate
        # input_0[:,4:]: deformation xx, deformation yy, deformation zz -> conditioning
        deformation_tensor = input_0[:, 4:]
        if not self.use_deformation_tensor:
            deformation_tensor = torch.zeros_like(input_0[:, 4:])

        data = self.normalize(input_0[:, :4])

        score_loss, log_dict = self.objective.loss(self.model,
                                                   {"data": data, "y": deformation_tensor,
                                                    "class_labels": labels, "pde_parameters": re}, split="train")

        self.log("loss", score_loss.item(), prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)

        self.log_dict(log_dict, prog_bar=False, logger=True,
                      on_step=True, on_epoch=False, sync_dist=True)

        return score_loss

    def validation_step(self, batch, batch_idx):

        input_0, input_1, labels, re = self.get_input(batch)

        # input_0[:,:4]: velocity x, velocity y, velocity z, pressure -> generate
        # input_0[:,4:]: deformation xx, deformation yy, deformation zz -> conditioning
        deformation_tensor = input_0[:, 4:]
        if not self.use_deformation_tensor:
            deformation_tensor = torch.zeros_like(input_0[:, 4:])

        data = self.normalize(input_0[:, :4])

        score_loss, log_dict = self.objective.loss(self.model,
                                                   {"data": data, "y": deformation_tensor,
                                                    "class_labels": labels, "pde_parameters": re}, split="train")

        self.log("val/loss", score_loss.item(), prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)

        self.log_dict(log_dict, prog_bar=False, logger=True,
                      on_step=True, on_epoch=False)

        return self.log_dict

    def generate(self, deformation_tensor,
                     num_inference_steps=100,
                     generator=None, image_shape=None, **kwargs):

        batch_dim = deformation_tensor.shape[0]

        if not self.use_deformation_tensor:
            deformation_tensor = torch.zeros_like(deformation_tensor)

        if image_shape is None:
            image_shape = (batch_dim, 4, 96, 96, 192)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            x0 = randn_tensor(image_shape, generator=generator)
            x0 = x0.to(self.device)
        else:
            x0 = randn_tensor(image_shape, generator=generator, device=self.device)

        input = torch.cat([x0, deformation_tensor], dim=1)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps:
            # 1. predict noise model_output

            t_batched = torch.full((batch_dim,), t.item(), device=self.device)

            model_output = self.forward(input, t_batched, **kwargs).sample

            # 2. compute previous image: x_t -> x_t-1
            x0 = self.scheduler.step(model_output, t, x0, generator=generator).prev_sample

            input = torch.cat([x0, deformation_tensor], dim=1)

        return self.denormalize(x0)

    def configure_optimizers(self):

        if self.training_mode == 'train':
            params = self.parameters()
        elif self.training_mode == 'finetune_context':
            params = self.model.xT.seq2seq.parameters()
        else:
            raise ValueError(f"Training mode {self.training_mode} not supported;")

        if self.optimizer == 'adamw':

            opt = torch.optim.AdamW(params, lr=self.learning_rate,
                                    weight_decay=1e-15)

        elif self.optimizer == 'adabelief':

            from adabelief_pytorch import AdaBelief # noqa
            opt = AdaBelief(params, lr=self.learning_rate)

        else:

            raise ValueError(f"Optimizer {self.optimizer} not supported;")

        return [opt]

    def get_last_layer(self):
        return self.decoder.conv_out.weight

