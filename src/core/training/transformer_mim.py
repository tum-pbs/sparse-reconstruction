

from src.core.generative_model_2d import GenerativeModel2D
from src.objectives import TrainingObjective

from src.utils import instantiate_from_config, get_obj_from_str
import torch.nn as nn

import torch

class TransformerMIM(GenerativeModel2D):
    def __init__(self,
                 model,
                 ckpt_path=None,
                 pretrained=True,
                 ignore_keys=None,
                 image_key=0,
                 monitor=None,
                 image_size=32
                 ):
        super().__init__()

        self.image_key = image_key

        # Score model
        self.model_class = get_obj_from_str(model["target"])

        if pretrained:
            self.model: nn.Module = self.model_class.from_pretrained(**model["params"])
        else:
            self.model: nn.Module = self.model_class(**model["params"])

        self.image_size = image_size
        self.num_patches = (self.image_size // self.model.config.patch_size) ** 2

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                conditioning: torch.Tensor = None) -> torch.Tensor:

        return self.model(x, t)

    def get_pipeline_args(self):
        return {
            "model": self.model,
        }

    def get_input(self, batch):
        x: torch.Tensor = batch[self.image_key]
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 1)

        return x

    def test_step(self, batch, batch_idx):

        return 0, {}

    def training_step(self, batch, batch_idx):

        inputs = self.get_input(batch)

        bool_masked_pos = torch.randint(low=0, high=2, size=(1, self.num_patches)).bool().to(self.device)

        outputs = self.model(inputs, bool_masked_pos=bool_masked_pos)

        loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction

        self.log("loss", loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)

        log_dict = {}
        self.log_dict(log_dict, prog_bar=False, logger=True,
                      on_step=True, on_epoch=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):

        inputs = self.get_input(batch)

        bool_masked_pos = torch.randint(low=0, high=2, size=(1, self.num_patches)).bool().to(self.device)

        outputs = self.model(inputs, bool_masked_pos=bool_masked_pos)

        loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction

        self.log("val/loss", loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)

        log_dict = {}
        self.log_dict(log_dict, prog_bar=False, logger=True,
                      on_step=True, on_epoch=False, sync_dist=True)

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