from src.core.generative_model_2d import GenerativeModel2D
from src.objectives import TrainingObjective

from src.utils import instantiate_from_config
import torch.nn as nn

import torch

class TransformerImageFlow(GenerativeModel2D):
    def __init__(self,
                 model,
                 objective,
                 embedding,
                 ckpt_path=None,
                 ignore_keys=None,
                 image_key=0,
                 monitor=None,
                 ):
        super().__init__()

        self.image_key = image_key

        # Score model
        self.model : nn.Module = instantiate_from_config(model)
        # self.embed_tasks = instantiate_from_config(embedding)

        # Objective
        self.objective : TrainingObjective = (
            instantiate_from_config(objective))

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                conditioning: torch.Tensor = None) -> torch.Tensor:

        return self.model(x, t)

    def get_pipeline_args(self):
        return {
            "unet": self.model,
        }

    def get_input(self, batch):

        data: torch.Tensor = batch["data"]
        meta_data: dict = batch["loading_metadata"]

        task_idx = meta_data['dataset_idx']

        x: torch.Tensor = data[self.image_key]
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 1)

        return x, task_idx

    def test_step(self, batch, batch_idx):

        return 0, {}

    def training_step(self, batch, batch_idx):

        inputs, task_idx = self.get_input(batch)

        # TODO use custom embeddings, for now: just feed task_idx to transformer model
        # labels = self.embed_tasks(task_idx)
        labels = task_idx

        score_loss, log_dict = self.objective.loss(self.model,
                                                   {"data": inputs, "class_labels": labels}, split="train")

        self.log("loss", score_loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)

        self.log_dict(log_dict, prog_bar=False, logger=True,
                      on_step=True, on_epoch=False, sync_dist=True)

        return score_loss

    def validation_step(self, batch, batch_idx):

        inputs, task_idx = self.get_input(batch)

        labels = task_idx

        score_loss, log_dict = self.objective.loss(self.model,
                                                   {"data": inputs, 'class_labels': labels}, split="val")

        self.log("val/loss", score_loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)

        self.log_dict(log_dict, prog_bar=False, logger=True,
                      on_step=True, on_epoch=False)

        return self.log_dict

    def configure_optimizers(self):

        #super().configure_optimizers()
        # Qiang: If use super().configure_optimizers(), will raise default warning

        #lr = self.learning_rate
        
        # Qiang: Do NOT use self.model.parameters() here, since fine-tuning will replace the parameters of the lightening model,
        # but not the parameters of the sub-models.
        
        #opt = torch.optim.AdamW(list(self.model.parameters()),
        #                       lr=lr)
        opt=torch.optim.AdamW(self.parameters(),lr=self.learning_rate)

        return [opt]

    def get_last_layer(self):
        return self.decoder.conv_out.weight