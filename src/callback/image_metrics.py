from typing import Optional, Any, Dict

import torchmetrics
from lightning import Callback, Trainer, LightningModule
from lightning.fabric.utilities import rank_zero_only

import torch
from math import ceil
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.callback.ema import EMAOptimizer
from src.metric.metric import get_metrics
from src.utils import instantiate_from_config, get_pipeline

import logging
log = logging.getLogger(__name__)

from lightning.fabric.utilities.rank_zero import rank_prefixed_message, _get_rank, rank_zero_info

import os
import torch.nn as nn

class ImageMetricLogger(Callback):
    def __init__(self, frequency: int, num_samples: int, batch_size: int,
                 pipelines: DictConfig, log_first_step: bool = False,
                 test: bool = True,
                 metric_config: DictConfig = None,
                 prepare_plots: Optional[DictConfig]=None):

        super().__init__()

        self.pipelines = OmegaConf.to_container(pipelines)
        self.batch_size = batch_size
        self.disabled = False
        self.test = test
        self.num_samples = num_samples
        self.log_first_step = log_first_step
        self.frequency = frequency
        self.temp_directory = "/tmp"

        if not prepare_plots is None:
            self.prepare_plots = instantiate_from_config(prepare_plots)
        else:
            self.prepare_plots = torch.nn.Identity()

        self.metric_config = metric_config


    def tensor_to_image(self, tensor: torch.Tensor, scale:bool = False):

        if scale:
            tensor = tensor.clamp(-1, 1)
            tensor = (tensor + 1) / 2

        tensor = tensor * 255
        tensor = tensor.type(torch.uint8)

        return tensor

    def update_fake(self, metric, data):

        if isinstance(metric, torchmetrics.image.fid.FrechetInceptionDistance):
            metric.update(data.to(metric.device), real=False)
        elif isinstance(metric, torchmetrics.image.InceptionScore):
            metric.update(data.to(metric.device))

        return metric

    def update_real(self, metric, data):

        if isinstance(metric, torchmetrics.image.fid.FrechetInceptionDistance):
            metric.update(data.to(metric.device), real=True)

        return metric


    def update_metrics_fake(self, pipelines, trainer, pl_module):

        metrics_torch = {}

        for pipeline in pipelines.keys():
            metrics_torch[pipeline] = nn.ModuleList(get_metrics(
                self.metric_config)).to(pl_module.device)

        for pipeline in pipelines.keys():

            generator = torch.Generator(device=pl_module.device).manual_seed(trainer.global_rank)

            log.info(rank_prefixed_message(f"Pipeline is on device {pipelines[pipeline].device}", _get_rank()))

            count = 0
            for _ in tqdm(range(int(ceil(
                    self.num_samples / (trainer.world_size * self.batch_size))))):

                sample = pipelines[pipeline](batch_size=self.batch_size, generator=generator,
                                             output_type='numpy', return_dict=True).images

                data_img = torch.from_numpy(sample)
                data_img = torch.stack([self.prepare_plots(d).detach().cpu() for d in data_img], dim=0)

                images = data_img[:int(self.num_samples / trainer.world_size) - count]

                count += images.shape[0]

                # images = self.tensor_to_image(torch.from_numpy(images))
                # images = torch.permute(images, (0, 3, 1, 2))

                for metric in metrics_torch[pipeline]:
                    log.info(rank_prefixed_message(f'Updating {metric.__class__.__name__} for {pipeline} '
                                                   f'pipeline with fake data...', _get_rank()))
                    self.update_fake(metric, images)

        return metrics_torch


    def update_metrics(self, trainer : Trainer, pl_module : LightningModule, dataloader,
                       use_ema: bool = True):

        optimizers_ema = [optimizer for optimizer in trainer.optimizers
                          if isinstance(optimizer, EMAOptimizer)]

        if use_ema and len(optimizers_ema) > 0:

            ema: EMAOptimizer = optimizers_ema[0]

            with ema.swap_ema_weights():

                pipelines = {key: get_pipeline(value, pl_module)
                             for key, value in self.pipelines.items()}

                metrics_torch = self.update_metrics_fake(pipelines, trainer, pl_module)

        else:

                pipelines = {key: get_pipeline(value, pl_module)
                            for key, value in self.pipelines.items()}

                metrics_torch = self.update_metrics_fake(pipelines, trainer, pl_module)


        # UPDATE METRICS

        print("Updating metrics with real data...")

        count = 0
        for data in tqdm(dataloader):

            data = pl_module.get_input(data)
            data_img = torch.stack([self.prepare_plots(d).detach().cpu() for d in data], dim=0)

            data_img = data_img[:(int(self.num_samples / trainer.world_size) - count)]

            for pipeline in pipelines.keys():
                for metric in metrics_torch[pipeline]:
                    log.info(rank_prefixed_message(f"Updating {metric.__class__.__name__} for "
                                                   f"{pipeline} pipeline with real data...", _get_rank()))

                    self.update_real(metric, data_img)

            count += data_img.shape[0]

            if count >= self.num_samples / trainer.world_size:
                break

        # save metrics

        for pipeline in pipelines.keys():
            for metric in metrics_torch[pipeline]:

                metric_name = metric.__class__.__name__
                state = metric.__getstate__()

                log.info(rank_prefixed_message(f"Saving {metric_name} for {pipeline} pipeline...",
                                               _get_rank()))
                torch.save(state, f"{self.temp_directory}/{metric_name}_{pipeline}_"
                                  f"{trainer.current_epoch}_{trainer.global_rank}.pth")

        # join metrics on rank 0
        trainer.strategy.barrier(name='metric_update')

        if trainer.global_rank == 0:

            metric_dict = {"epoch": trainer.current_epoch}

            for pipeline in pipelines.keys():
                for metric in metrics_torch[pipeline]:

                    metric.reset()

                    metric_name = metric.__class__.__name__

                    state = torch.load(f"{self.temp_directory}/{metric_name}_{pipeline}_"
                                  f"{trainer.current_epoch}_0.pth")

                    import os
                    # delete the file after loading
                    os.remove(f"{self.temp_directory}/{metric_name}_{pipeline}_"
                                  f"{trainer.current_epoch}_0.pth")

                    log.info(rank_prefixed_message(f"Loading {metric_name} for {pipeline} pipeline...",
                                                   _get_rank()))
                    metric.__setstate__(state)

                    for i in range(1, trainer.world_size):

                        state = torch.load(f"{self.temp_directory}/{metric_name}_{pipeline}_"
                                  f"{trainer.current_epoch}_{i}.pth")

                        os.remove(f"{self.temp_directory}/{metric_name}_{pipeline}_"
                                  f"{trainer.current_epoch}_{i}.pth")

                        state_ = {key: value.to(metric.device) for key, value in state.items() if
                                  isinstance(value, torch.Tensor)}

                        state_.update({key: value for key, value in state.items() if
                                       not isinstance(value, torch.Tensor)})

                        metric._reduce_states(state_)

                    metric_value = metric.compute()

                    if isinstance(metric_value, tuple):
                        metric_dict[f"{pipeline}_{metric_name}_mean"] = metric_value[0]
                        metric_dict[f"{pipeline}_{metric_name}_std"] = metric_value[1]

                    else:
                        metric_dict[f"{pipeline}_{metric_name}"] = metric.compute()

            self.log_metrics(trainer, metric_dict)


    @rank_zero_only
    def log_metrics(self, trainer: Trainer, metric_dict: Dict[str, Any]):

        trainer.logger.experiment.log(metric_dict)

    def check_frequency(self, check_idx):
        # check_idx + 1 since current_idx starts with 0
        return ((check_idx + 1) % self.frequency) == 0 or (check_idx == 0 and self.log_first_step)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.test:
            self.update_metrics(trainer, pl_module, trainer.test_dataloaders)

    def on_train_epoch_end(self, trainer, pl_module):

        if ((not self.disabled) and
                self.check_frequency(pl_module.current_epoch)):
            self.update_metrics(trainer, pl_module, trainer.train_dataloader)

class ImageMetricLoggerSync(Callback):
    def __init__(self, frequency: int, num_samples: int, batch_size: int,
                 log_first_step: bool = True, prepare_plots: Optional[DictConfig]=None):
        super().__init__()
        self.batch_size = batch_size
        self.disabled = True # False
        self.num_samples = num_samples
        self.log_first_step = log_first_step
        self.frequency = frequency

        if not prepare_plots is None:
            self.prepare_plots = instantiate_from_config(prepare_plots)
        else:
            self.prepare_plots = torch.nn.Identity()

    def write_data(self, trainer : Trainer):

        trainer.strategy.barrier(name='image_writer_callback')

        # TODO generate big datasets and log them for each node/gpu

        pass

    def tensor_to_image(self, tensor: torch.Tensor):

        tensor = tensor.clamp(-1, 1)
        tensor = (tensor + 1) / 2
        tensor = tensor * 255
        tensor = tensor.type(torch.uint8)

        return tensor

    def _update_metrics(self, metric_list, data, real=False):

        data = data[1]
        data = torch.Tensor([self.prepare_plots(d).detach().cpu() for d in data])
        data = self.tensor_to_image(data)

        for metric in metric_list:

            if isinstance(metric, torchmetrics.image.fid.FrechetInceptionDistance):

                metric.update(data, real=real)

            elif isinstance(metric, torchmetrics.image.InceptionScore):

                metric.update(data)

            else:

                continue

    def update_metrics(self, trainer : Trainer, pl_module : LightningModule):

            # GENERATE SAMPLES

            for _ in tqdm(range(int(ceil(
                    self.num_samples // (trainer.world_size * self.batch_size))))):

                sample = pl_module.sample(self.batch_size)

                for sampler, data in zip(pl_module.samplers, sample):

                    self._update_metrics(pl_module.image_metrics[sampler.name],
                                        data, real=False)

            # UPDATE METRICS

            for data in trainer.train_dataloader:

                for sampler_name in pl_module.image_metrics:

                    self._update_metrics(pl_module.image_metrics[sampler_name],
                                        data, real=True)

    @rank_zero_only
    def log_metrics(self, trainer: Trainer, metric_dict: dict):

        trainer.logger.experiment.log(metric_dict)


    def compute_metrics(self, trainer, pl_module):

        metric_dict = {}

        for sampler_name in pl_module.image_metrics:

            metric_dict[sampler_name] = {}

            for metric in pl_module.image_metrics[sampler_name]:

                # metric compute should synchronize across devices
                metric_dict[sampler_name][metric.__class__.__name__] \
                    = metric.compute()

        self.log_metrics(trainer, metric_dict)

    def check_frequency(self, check_idx):
        return (check_idx % self.frequency) == 0 and (check_idx > 0 or self.log_first_step)

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.disabled and self.check_frequency(pl_module.global_step):
            self.update_metrics(trainer, pl_module)
            self.compute_metrics(trainer, pl_module)