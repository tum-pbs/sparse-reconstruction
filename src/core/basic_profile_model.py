import torch
import torch.nn as nn
import lightning

from .task_embedding import TaskEmbedding


class BasicProfileModel(lightning.LightningModule):
    def __init__(self,
                channels=3,
                monitor=None,):
        super().__init__()

        #self.task_embedding = TaskEmbedding()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=4, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=3, out_channels=channels, kernel_size=3, stride=4, output_padding=1),
            nn.ReLU(),
        )

        if monitor is not None:
            self.monitor = monitor

    def forward(self, x):
        #t = self.task_embedding(x)

        return self.model(x["data"][:,0])

    def training_step(self, batch, batch_idx):
        b_out = self(batch)
        loss = nn.functional.mse_loss(b_out, batch["data"][:,1])

        self.log("loss", loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        b_out = self(batch)
        loss = nn.functional.mse_loss(b_out, batch["data"][:,1])

        self.log("val/loss", loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        opt=torch.optim.AdamW(self.parameters(),lr=self.learning_rate)
        return [opt]



class BasicProfileModel3D(lightning.LightningModule):
    def __init__(self,
                channels=3,
                many_layers=False,
                monitor=None,):
        super().__init__()

        if many_layers:
            chan = 48
            self.model = nn.Sequential(
                nn.Conv3d(in_channels=channels, out_channels=chan, kernel_size=3, stride=4, padding=1),
                nn.ReLU(),
                nn.Conv3d(in_channels=chan, out_channels=chan, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(in_channels=chan, out_channels=chan, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(in_channels=chan, out_channels=chan, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(in_channels=chan, out_channels=chan, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(in_channels=chan, out_channels=chan, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(in_channels=chan, out_channels=chan, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(in_channels=chan, out_channels=chan, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(in_channels=chan, out_channels=chan, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(in_channels=chan, out_channels=chan, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(in_channels=chan, out_channels=chan, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(in_channels=chan, out_channels=chan, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(in_channels=chan, out_channels=chan, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(in_channels=chan, out_channels=chan, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose3d(in_channels=chan, out_channels=channels, kernel_size=3, stride=4, output_padding=1),
                nn.ReLU(),
            )

        else:
            self.model = nn.Sequential(
                nn.Conv3d(in_channels=channels, out_channels=3, kernel_size=3, stride=4, padding=1),
                nn.ReLU(),
                nn.ConvTranspose3d(in_channels=3, out_channels=channels, kernel_size=3, stride=4, output_padding=1),
                nn.ReLU(),
            )

        if monitor is not None:
            self.monitor = monitor

        self.embedding = TaskEmbedding()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        emb = self.embedding(batch)

        b = batch["data"][0][:,0]
        b_out = self(b)
        loss = nn.functional.mse_loss(b_out, b)

        self.log("loss", loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        emb = self.embedding(batch)

        b = batch["data"][0][:,0]
        b_out = self(b)
        loss = nn.functional.mse_loss(b_out, b)

        self.log("val/loss", loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        opt=torch.optim.AdamW(self.parameters(),lr=self.learning_rate)
        return [opt]