
from pathlib import Path
from typing import List, Optional, Union

import numpy
from lightning import Callback, Trainer, LightningModule
from lightning.fabric.utilities import rank_zero_only

import torchvision

import os
import numpy as np

from PIL import Image

import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from src.utils import instantiate_from_config, get_pipeline


class ImageLogger(Callback):
    def __init__(self, frequency: int, max_images: int, pipelines: Union[DictConfig, OmegaConf], clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_first_step=True,
                 log_images_kwargs=None, prepare_plots: Optional[DictConfig]=None):
        super().__init__()

        self.pipelines = OmegaConf.to_container(pipelines)
        self.rescale = rescale
        self.frequency = frequency
        self.max_images = max_images

        if not prepare_plots is None:
            self.prepare_plots = instantiate_from_config(prepare_plots)
        else:
            self.prepare_plots = torch.nn.Identity()

        self.log_steps = [2 ** n for n in range(int(np.log2(self.frequency)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.frequency]

        self.clamp = clamp
        self.disabled = disabled

        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, save_dir: Path, name: str, images: List[numpy.ndarray],
                  current_epoch: int) -> None:

        root = os.path.join(save_dir, "images", name)

        images = torch.from_numpy(np.stack(images))

        grid = torchvision.utils.make_grid(images, nrow=4)

        grid = torch.permute(grid, (1, 2, 0))
        grid = grid.numpy()

        filename = "gs-e-{:06}_b.png".format(
            current_epoch,
            )
        path = os.path.join(root, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        Image.fromarray(grid).save(path)

    def log_img(self, trainer, pl_module):

        current_epoch = trainer.current_epoch

        pipelines = {key: get_pipeline(value, pl_module)
                     for key, value in self.pipelines.items()}

        is_train = pl_module.training

        if is_train:
            pl_module.eval()

        for pipeline in pipelines.keys():

            generator = torch.Generator(device=pl_module.device).manual_seed(trainer.global_rank)

            images = pipelines[pipeline](batch_size=self.max_images, generator=generator,
                                         output_type='numpy', return_dict=True).images

            images = torch.from_numpy(images)
            images = [self.prepare_plots(im).detach().cpu() for im in list(images)[:self.max_images]]

            logdir = trainer.logger.experiment.config["runtime"]["logdir"]
            self.log_local(logdir, pipeline, list(images), current_epoch)

            trainer.logger.experiment.log({
                pipeline: [wandb.Image(im, caption=k) for k, im in enumerate(images)]
            })

        if is_train:
            pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx + 1) % self.frequency) == 0 or (check_idx in self.log_steps) or (check_idx == 0 and self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def log_dataset(self, trainer, pl_module, dataloader, name):

        images_list = []

        # only visualize first dataset
        for batch in iter(dataloader):

            data, metadata = pl_module.get_input(batch)
            images_list.extend(list(data))

            if len(images_list) > self.max_images:
                break

        images = [self.prepare_plots(im).detach().cpu() for im in images_list[:self.max_images]]

        logdir = trainer.logger.experiment.config["runtime"]["logdir"]
        self.log_local(logdir, name, images[:self.max_images], 0)

        trainer.logger.experiment.log({
            name: [wandb.Image(im, caption=str(k)) for k, im in enumerate(images)]
        })

    def log_train_dataset(self, trainer: Trainer, pl_module: LightningModule):

        self.log_dataset(trainer, pl_module, trainer.train_dataloader, 'data_train')

    def log_test_dataset(self, trainer: Trainer, pl_module: LightningModule):

        if (dataloader := trainer.test_dataloaders) is None:
            trainer.test_loop.setup_data()
            dataloader = trainer.test_dataloaders

        self.log_dataset(trainer, pl_module, dataloader, 'data_test')


    @rank_zero_only
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.disabled and pl_module.current_epoch == 0:
            self.log_train_dataset(trainer, pl_module)
            self.log_test_dataset(trainer, pl_module)
            self.log_img(trainer, pl_module)


    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):

        if ((not self.disabled) and (pl_module.current_epoch > 0) and
                self.check_frequency(pl_module.current_epoch) and (self.max_images > 0)):
            self.log_img(trainer, pl_module)

class MIMImageLogger(ImageLogger):

    def __init__(self, frequency: int, max_images: int, pipelines: Union[DictConfig, OmegaConf], clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_first_step=True,
                 log_images_kwargs=None, prepare_plots: Optional[DictConfig]=None,
                 image_size: int = 256, patch_size: int = 16):
        super().__init__(frequency, max_images, pipelines, clamp, increase_log_steps,
                         rescale, disabled, log_first_step, log_images_kwargs, prepare_plots)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (self.image_size // self.patch_size) ** 2


    @rank_zero_only
    def log_local(self, save_dir: Path, name: str, images_masked: List[numpy.ndarray],
                  images_reconstructed: List[numpy.ndarray], current_epoch: int, split: str) -> None:

        root = os.path.join(save_dir, "images", name, split)

        images_masked = images_masked.cpu()
        images_reconstructed = images_reconstructed.cpu()

        for type, images in zip(["masked", "reconstructed"], [images_masked, images_reconstructed]):

            grid = torchvision.utils.make_grid(images, nrow=4)

            grid = torch.permute(grid, (1, 2, 0))
            grid = grid.numpy()

            filename = "{}-gs-e-{:06}_b.png".format(
                type, current_epoch,
            )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def create_mask(self, num_masks, generator=None):

        if generator is None:
            device = None
        else:
            device = generator.device

        bool_masked_pos_ = torch.randint(low=0, high=2, size=(num_masks, self.num_patches),
                                        generator=generator, device=device).bool()

        size = self.image_size // self.patch_size
        bool_masked_pos = bool_masked_pos_.reshape(-1, size, size)
        mask = (
            bool_masked_pos.repeat_interleave(self.patch_size, 1)
            .repeat_interleave(self.patch_size, 2)
            .unsqueeze(1)
            .contiguous()
        )

        return mask, bool_masked_pos_

    @staticmethod
    def get_images(num_images: int, pl_module, loader, generator=None):

        images_list = []

        # TODO make sure to always get the same data,
        #  even when shuffling is enabled
        enumerate(loader)
        for batch in iter(loader):

            images_list.extend(list(pl_module.get_input(batch)))

            if len(images_list) > num_images:
                break

        images_list = images_list[:num_images]

        images = torch.stack(images_list)

        return images


    def log_img(self, trainer, pl_module):

        current_epoch = trainer.current_epoch

        pipelines = {key: get_pipeline(value, pl_module)
                     for key, value in self.pipelines.items()}

        is_train = pl_module.training

        if is_train:
            pl_module.eval()

        for pipeline in pipelines.keys():

            reference_images_train = self.get_images(self.max_images, pl_module,
                                                     trainer.train_dataloader).to(pl_module.device)

            reference_images_val = self.get_images(self.max_images, pl_module,
                                                   trainer.val_dataloaders).to(pl_module.device)

            generator = torch.Generator(device=pl_module.device).manual_seed(trainer.global_rank)

            mask_train, bool_masked_pos_train = self.create_mask(self.max_images, generator)
            mask_val, bool_masked_pos_val = self.create_mask(self.max_images, generator)


            images_masked_train = torch.multiply(reference_images_train, mask_train.to(pl_module.device))
            images_masked_val = torch.multiply(reference_images_val, mask_val.to(pl_module.device))

            images_train = pipelines[pipeline](images=reference_images_train, bool_masked_pos=bool_masked_pos_train,
                                         output_type='numpy', return_dict=True).images

            images_val = pipelines[pipeline](images=reference_images_val, bool_masked_pos=bool_masked_pos_val,
                                            output_type='numpy', return_dict=True).images

            images_train = self.post_process_images(images_train)
            images_val = self.post_process_images(images_val)

            images_masked_train = self.post_process_images(images_masked_train)
            images_masked_val = self.post_process_images(images_masked_val)

            logdir = trainer.logger.experiment.config["runtime"]["logdir"]

            self.log_local(logdir, pipeline, images_masked_train, images_train, current_epoch, "train")
            self.log_local(logdir, pipeline, images_masked_val, images_val, current_epoch, "val")

            trainer.logger.experiment.log({
                f"{pipeline}_train": [wandb.Image(im, caption=str(k)) for k, im in enumerate(images_train)],
                f"{pipeline}_val": [wandb.Image(im, caption=str(k)) for k, im in enumerate(images_val)],
            })

        if is_train:
            pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx + 1) % self.frequency) == 0 or (check_idx in self.log_steps) or (
                check_idx == 0 and self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def post_process_images(self, image_list):

        return torch.stack([self.prepare_plots(im) for im in image_list])

    def log_dataset(self, trainer: Trainer, pl_module: LightningModule):

        generator = torch.Generator(device=pl_module.device).manual_seed(trainer.global_rank)

        images_train = self.get_images(self.max_images, pl_module, trainer.train_dataloader)
        images_val = self.get_images(self.max_images, pl_module, trainer.val_dataloaders)

        images_train = self.post_process_images(images_train).to(pl_module.device)
        images_val = self.post_process_images(images_val).to(pl_module.device)

        mask_train, _ = self.create_mask(self.max_images, generator=generator)
        mask_val, _ = self.create_mask(self.max_images, generator=generator)

        images_masked_train = torch.multiply(images_train, mask_train.to(pl_module.device))
        images_masked_val = torch.multiply(images_val, mask_val.to(pl_module.device))

        logdir = trainer.logger.experiment.config["runtime"]["logdir"]

        self.log_local(logdir, 'data', images_masked_train, images_train, 0, "train")
        self.log_local(logdir, 'data', images_masked_val, images_val, 0, "val")

        trainer.logger.experiment.log({
            'data_train': [wandb.Image(im, caption=str(k)) for k, im in enumerate(images_train)],
            'data_val': [wandb.Image(im, caption=str(k)) for k, im in enumerate(images_val)],
            'masked_train': [wandb.Image(im, caption=str(k)) for k, im in enumerate(images_masked_train)],
            'masked_val': [wandb.Image(im, caption=str(k)) for k, im in enumerate(images_masked_val)],
        })

class MultiTaskImageLogger(ImageLogger):
    def __init__(self, frequency: int, max_images: int, datasets: List[str], pipelines: Union[DictConfig, OmegaConf], clamp=True,
                 increase_log_steps=True,
                 rescale=True, disabled=False, log_first_step=True,
                 log_images_kwargs=None, prepare_plots: Optional[DictConfig]=None):
        super().__init__(frequency, max_images, pipelines, clamp, increase_log_steps,
                         rescale, disabled, log_first_step, log_images_kwargs, prepare_plots)

        self.datasets = datasets

    def log_img(self, trainer, pl_module):

        current_epoch = trainer.current_epoch

        pipelines = {key: get_pipeline(value, pl_module)
                     for key, value in self.pipelines.items()}

        is_train = pl_module.training

        if is_train:
            pl_module.eval()

        for dataset_idx, dataset_name in enumerate(self.datasets):

            task_idx = torch.tensor([dataset_idx] * self.max_images).to(pl_module.device)

            for pipeline in pipelines.keys():

                generator = torch.Generator(device=pl_module.device).manual_seed(trainer.global_rank)

                images = pipelines[pipeline](task_idx=task_idx, generator=generator,
                                             output_type='numpy', return_dict=True).images

                images = torch.from_numpy(images)
                images = [self.prepare_plots(im).detach().cpu() for im in list(images)[:self.max_images]]

                logdir = trainer.logger.experiment.config["runtime"]["logdir"]
                self.log_local(logdir, f'{pipeline}_{dataset_name}', list(images), current_epoch)

                trainer.logger.experiment.log({
                    f'{pipeline}_{dataset_name}': [wandb.Image(im, caption=k) for k, im in enumerate(images)]
                })

        if is_train:
            pl_module.train()
