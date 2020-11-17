import pytorch_lightning as pl
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import models


class MobileNetLightingModel(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(1280, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
        # https://github.com/shamangary/SSR-Net/blob/master/training_and_testing/TYY_model.py#L42

    def forward(self, x):
        return self.mobilenet(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x).squeeze()
        loss = mse_loss(y_hat, y.float())
        self.log('train_mse', loss.item(), logger=True, on_step=False, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x).squeeze()
        loss = mse_loss(y_hat, y.float())
        self.log('val_mse', loss.item(), prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]
