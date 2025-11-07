import enum
from abc import ABC
from typing import Dict, Tuple, Optional

from torch import nn
import torch

from utils import instantiate_from_config


class AbstractTask(ABC):

    def __init__(self):
        pass

    def prepare_data(self, data: Dict, prob: float) -> dict:
        pass

def generate_channel_mask(channels: list[torch.Tensor], prob: float, patch_size: int) -> list[torch.Tensor]: # noqa

    masks = []

    for channel in channels:

        batch_size = channel.shape[0]

        num_channel_tokens = (channel.shape[1] // patch_size) * (channel.shape[2] // patch_size)
        num_channel_tokens = int(num_channel_tokens)

        mask = torch.ones(batch_size, num_channel_tokens,
                          device=channel.device)

        mask = torch.bernoulli(prob * mask)

        masks.append(mask)

    return masks

def generate_empty_channel_mask(channels: list[torch.Tensor], patch_size: int) -> list[torch.Tensor]: # noqa

    return generate_channel_mask(channels, prob=0.0, patch_size=patch_size)


def generate_full_channel_mask(channels: list[torch.Tensor], patch_size: int) -> list[torch.Tensor]: # noqa

    return generate_channel_mask(channels, prob=1.0, patch_size=patch_size)

def generate_token_mask(tokens: torch.Tensor, prob: float) -> torch.Tensor:

        batch_size = tokens.shape[0]
        num_tokens = tokens.shape[1]

        mask = torch.ones(batch_size, num_tokens, device=tokens.device)

        torch.bernoulli(prob * mask)

        return mask

def generate_empty_token_mask(tokens: torch.Tensor) -> torch.Tensor:

        return generate_token_mask(tokens, prob=0.0)

def generate_full_token_mask(tokens: torch.Tensor) -> torch.Tensor:

        return generate_token_mask(tokens, prob=1.0)


class GenerativeModeling(AbstractTask):

    def __init__(self, patch_size: int):
        super(GenerativeModeling, self).__init__()
        self.patch_size = patch_size

    def prepare_data(self, data: dict, prob: float) -> dict:

        mask_channels = []
        data['channels'] = []

        for temporal_data in data['temporal_data']:
            channels = temporal_data['channels']
            mask_channels.extend(generate_empty_channel_mask(channels, patch_size=self.patch_size))
            data['channels'].extend(channels)

        mask_channels = torch.concatenate(mask_channels, dim=1)

        mask_tokens = generate_full_token_mask(data['tokens'])

        data['mask_channels'] = mask_channels
        data['mask_tokens'] = mask_tokens

        data['mask'] = torch.concatenate([mask_channels, mask_tokens], dim=1)

        return data

class Superresolution(AbstractTask):

    def __init__(self):
        super(Superresolution, self).__init__()

    def prepare_data(self, data: Dict):
        pass

class MaskedModeling(AbstractTask):

    def __init__(self, patch_size: int):
        super(MaskedModeling, self).__init__()
        self.patch_size = patch_size

    def prepare_data(self, data: dict, prob: float) -> dict:

        mask_channels = []
        data['channels'] = []

        for temporal_data in data['temporal_data']:
            channels = temporal_data['channels']
            mask_channels.extend(generate_channel_mask(channels, prob=prob, patch_size=self.patch_size))
            data['channels'].extend(channels)

        mask_channels = torch.concatenate(mask_channels, dim=1)

        mask_tokens = generate_token_mask(data['tokens'], prob=prob)

        data['mask_channels'] = mask_channels
        data['mask_tokens'] = mask_tokens

        data['mask'] = torch.concatenate([mask_channels, mask_tokens], dim=1)

        return data

class PDEIdentification(AbstractTask):

    def __init__(self, patch_size: int):
        super(PDEIdentification, self).__init__()
        self.patch_size = patch_size

    def prepare_data(self, data: dict, prob: float) -> dict:

        mask_channels = []

        for temporal_data in data['temporal_data']:
            channels = temporal_data['channels']
            mask_channels.extend(generate_full_channel_mask(channels, patch_size=self.patch_size))

        data['channels'] = []
        for channels in data['temporal_data']:
            data['channels'].extend(channels['channels'])

        mask_channels = torch.concatenate(mask_channels, dim=1)

        mask_tokens = generate_empty_token_mask(data['tokens'])

        data['mask_channels'] = mask_channels
        data['mask_tokens'] = mask_tokens

        data['mask'] = torch.concatenate([mask_channels, mask_tokens], dim=1)

        return data

class ForwardPrediction(AbstractTask):

        def __init__(self, patch_size: int):
            super(ForwardPrediction, self).__init__()
            self.patch_size = patch_size

        def prepare_data(self, data: dict, prob: float) -> dict:

            mask_channels = []
            data['channels'] = []

            for temporal_data in data['temporal_data'][:-1]:
                channels = temporal_data['channels']
                mask_channels.extend(generate_full_channel_mask(channels, patch_size=self.patch_size))
                data['channels'].extend(channels)

            channels = data['temporal_data'][-1]['channels']
            mask_channels.extend(generate_empty_channel_mask(channels, patch_size=self.patch_size))
            data['channels'].extend(channels)

            mask_channels = torch.concatenate(mask_channels, dim=1)

            mask_tokens = generate_full_token_mask(data['tokens'])

            data['mask_channels'] = mask_channels
            data['mask_tokens'] = mask_tokens

            data['mask'] = torch.concatenate([mask_channels, mask_tokens], dim=1)

            return data

class BackwardPrediction(AbstractTask):

    def __init__(self, patch_size: int):
        super(BackwardPrediction, self).__init__()
        self.patch_size = patch_size

    def prepare_data(self, data: dict, prob: float) -> dict:
        mask_channels = []
        data['channels'] = []

        channels = data['temporal_data'][0]['channels']
        mask_channels.extend(generate_empty_channel_mask(channels, patch_size=self.patch_size))
        data['channels'].extend(channels)

        for temporal_data in data['temporal_data'][1:]:
            channels = temporal_data['channels']
            mask_channels.extend(generate_full_channel_mask(channels, patch_size=self.patch_size))
            data['channels'].extend(channels)

        mask_channels = torch.concatenate(mask_channels, dim=1)

        mask_tokens = generate_full_token_mask(data['tokens'])

        data['mask_channels'] = mask_channels
        data['mask_tokens'] = mask_tokens

        data['mask'] = torch.concatenate([mask_channels, mask_tokens], dim=1)

        return data

def get_task_from_str(task: str, patch_size: int) -> AbstractTask:
    init_dict = {'target': f'src.core.masked_modeling.tasks.{task}',
                 'params': {'patch_size': patch_size}}
    return instantiate_from_config(init_dict)

Task = enum.Enum('Task', [('GenerativeModeling', 1),
                          ('Superresolution', 2),
                          ('MaskedModeling', 3),
                          ('PDEIdentification', 4),
                          ('ForwardPrediction', 5),
                          ('BackwardPrediction', 6),
                          ('Interpolation', 7)])

Tokens = enum.Enum('Tokens', [('PDE', 1),
                              ('ReynoldsNumber', 2)])

class TaskMasking(nn.Module):

    def __init__(self, tasks: list[str], weighting, patch_size: int, downsampling_factor: int = 2):
        super(TaskMasking, self).__init__()
        self.patch_size = patch_size

        self.downsampling_factor = downsampling_factor
        self.initialize_tasks(tasks, weighting)


    def get_random_task(self, batch_idx: int):

        # sample from probabilities with seed batch_idx
        command_idx = torch.multinomial(self.weighting, 1,
                                        generator=torch.Generator().manual_seed(batch_idx))

        return command_idx[0].item()

    def initialize_tasks(self, tasks: list[str], weighting: str):

        self.tasks: list[AbstractTask] = [] # noqa
        for task in tasks:
            try:
                self.tasks.append(get_task_from_str(task, patch_size=self.patch_size))
            except Exception as e:
                print(f"Could not initialize task {task}: {e}")

        if len(self.tasks) == 0:
            raise ValueError("No tasks initialized")

        if isinstance(weighting, str):

            if weighting == 'uniform':
                self.weighting = torch.Tensor([1/len(self.tasks) for _ in self.tasks]) # noqa
            else:
                raise ValueError(f"Weighting {weighting} not implemented")

        else:

            self.weighting = weighting # noqa

    def get_masks(self, data: dict, task_idx: int, prob: float) -> dict:

        return self.tasks[task_idx].prepare_data(data, prob)

    def get_inputs_impl(self, batch: dict, task_idx: int, prob: float, window: int):

        data = self.get_data(batch)
        masks = self.get_masks(data, task_idx, prob)

        data.update(masks)

        return data

    def get_inputs(self, batch: dict, batch_idx: int, task: Optional[int] = None):

        if task is None:
            task = self.get_random_task(batch_idx)

        prob = torch.rand(1, device=batch['data'].device).item()

        return self.get_inputs_impl(batch, task, prob, batch_idx)

    def get_channels(self, batch: dict, idx: int) -> Tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]: # noqa
        """
        Get the data for a specific channel at a specific time step.

        Args:
            batch: batch data
            idx: index of the time step

        Returns:
            tuple: channels, simulation times, channel ids
        """

        channel_data: torch.Tensor = batch['data'][:, idx]

        if self.downsampling_factor > 1:
            channel_data = channel_data[:, :, ::self.downsampling_factor, ::self.downsampling_factor]

        channel_id = batch['physical_metadata']['Fields'].long()
        channel_time = idx * batch['time_step_stride']

        channel_data = channel_data.transpose(0, 1)
        channels = list(channel_data)

        channel_time = torch.unsqueeze(channel_time, -1)

        return channels, list(channel_time.transpose(0, 1)), list(channel_id.transpose(0, 1))

    def get_tokens(self, batch: dict) -> torch.tensor: # noqa

        token_ids = []
        token_values = [] # noqa

        # add type of PDE
        token_values.append(batch['physical_metadata']['PDE'])
        token_ids.append(Tokens.PDE.value + torch.zeros_like(token_values[-1])) # noqa

        # add Reynolds number
        token_values.append(batch['physical_metadata']['Reynolds Number'])
        token_ids.append(Tokens.ReynoldsNumber.value + torch.zeros_like(token_values[-1])) # noqa

        token_ids = torch.concatenate(token_ids, dim=1)
        token_values = torch.concatenate(token_values, dim=1)

        tokens = torch.stack([token_ids, token_values], dim=2)

        return tokens

    def get_data(self, batch: dict):

        timesteps = batch['data'].shape[1]

        data_temp = []

        for idx in range(timesteps):

            channels, simulation_times, channel_ids = self.get_channels(batch, idx)
            data_temp.append({'channels': channels, 'simulation_times': simulation_times, 'channel_ids': channel_ids})

        tokens = self.get_tokens(batch)

        result = {} # noqa

        result['temporal_data'] = data_temp

        result['simulation_times'] = []
        for data in data_temp:
            result['simulation_times'].extend(data['simulation_times'])

        result['channel_ids'] = []
        for data in data_temp:
            result['channel_ids'].extend(data['channel_ids'])

        result['tokens'] = tokens

        return result




