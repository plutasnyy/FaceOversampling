import pytorch_lightning as pl
from torch import nn
from torch.nn.functional import l1_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models


class MobileNetLightingModel(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.classifier[1] = nn.Linear(1280, 1)

    def forward(self, x):
        return self.mobilenet(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x).squeeze()
        loss = l1_loss(y_hat, y.float())
        self.log('train_mae', loss.item(), logger=True, on_step=False, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x).squeeze()
        loss = l1_loss(y_hat, y.float())
        self.log('val_mae', loss.item(), prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_mae'
        }
