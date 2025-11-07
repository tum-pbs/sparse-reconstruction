from GIFt import enable_fine_tuning
from omegaconf import DictConfig, OmegaConf
from lightning import LightningModule
from lightning.fabric.utilities.rank_zero import rank_zero_info
from lightning.pytorch.loggers import WandbLogger
from src.utils import instantiate_from_config
from .info import (
    FineTuningInfo,
    fill_finetuning_info,
    plain_fine_tuning_info,
    wandb_text_fine_tuning_info,
)
from .weight_loader import WeightLoader


def fine_tuning_setup(
    model: LightningModule, config: DictConfig, wandb_logger: WandbLogger = None
):
    if "fine_tuning" in config.keys():

        # summarize the fine-tuning information before fine-tuning
        info = FineTuningInfo()
        info.num_para_before = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        # load the pre-trained weights
        try:
            path_pretrain_weight = config.fine_tuning.path_pretrained_weights
        except:
            raise ValueError("The path to the pretrained weights is not defined.")
        info.path_pretrained_weights = path_pretrain_weight
        weight_loader: WeightLoader = instantiate_from_config(
            OmegaConf.to_container(config.fine_tuning.weight_loader)
        )
        model = weight_loader.load(model, path_pretrain_weight)

        # apply fine-tuning strategy
        strategy_config = OmegaConf.to_container(
            config.fine_tuning.strategy
        )
        ft_strategy = instantiate_from_config(strategy_config)
        enable_fine_tuning(model, ft_strategy, replace_parameter_function=True)

        # summarize the fine-tuning information after fine-tuning
        info = fill_finetuning_info(model, ft_strategy, info)
        text_info = plain_fine_tuning_info(info)
        if wandb_logger is not None:
            for wandb_text_info in wandb_text_fine_tuning_info(info):
                wandb_logger.log_text(**wandb_text_info)
    else:
        text_info = "Fine-tuning not enabled"
    rank_zero_info(text_info)
