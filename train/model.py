from operator import itemgetter

import pytorch_lightning as pl
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torchvision import models
import numpy as np


class MobileNetLightingModel(pl.LightningModule):
    """https://pytorch.org/hub/pytorch_vision_mobilenet_v2/"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.classifier[1] = nn.Linear(1280, 1)

    def forward(self, x):
        return self.mobilenet(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x).float()
        loss = mse_loss(y_hat, y.float())
        return {'loss': loss, 'train_mse': loss.item()}

    def training_epoch_end(self, outputs):
        avg_train_mse = np.mean(list(map(itemgetter('train_mse'), outputs)))
        self.logger.log_metrics({'train_mse': avg_train_mse}, step=self.current_epoch)

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x).float()
        loss = mse_loss(y_hat, y.float())
        return {'valid_mse': loss.item()}

    def validation_epoch_end(self, outputs):
        avg_valid_mse = np.mean(list(map(itemgetter('valid_mse'), outputs)))
        self.logger.log_metrics({'valid_mse': avg_valid_mse}, step=self.current_epoch)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return optimizer
