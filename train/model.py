import io

import pandas as pd
import plotly.express as px
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models

from l1loss import SmoothL1Loss


class MobileNetLightingModel(pl.LightningModule):

    def __init__(self, loss='mae', beta=1.0, logger=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.classifier[1] = nn.Linear(1280, 1)
        self.comet_logger = logger
        if loss == 'mae':
            self.loss = L1Loss()
        elif loss == 'huber':
            self.loss = SmoothL1Loss(beta=beta, reduction='mean')
        else:
            raise KeyError(f'Wrong loss type: {loss}')

    def forward(self, x):
        return self.mobilenet(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x).squeeze()
        loss = self.loss(y_hat, y.float())
        self.log('train_loss', loss.item(), logger=True, on_step=False, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x).squeeze()
        loss = self.loss(y_hat, y.float())
        self.log('val_loss', loss.item(), prog_bar=True, logger=True, on_step=False, on_epoch=True)

        y_true = y.cpu().numpy()
        y_pred = y_hat.cpu().numpy()

        age_absolute_error_tuple = [(y_t, abs(y_t - y_p)) for y_t, y_p in zip(y_true, y_pred)]

        df = pd.DataFrame(age_absolute_error_tuple, columns=['Age', 'Absolute Error']).set_index('Age')
        self.log('val_mae', torch.tensor(df.mean().values), prog_bar=True, logger=True)

        grouped_df = df.groupby('Age').agg('mean')
        self.log('val_wmae', torch.tensor(grouped_df.mean().values), prog_bar=True, logger=True)

        return age_absolute_error_tuple

    def validation_epoch_end(self, outs):
        age_absolute_error_tuple = [l for o in outs for l in o]
        print(len(age_absolute_error_tuple))
        df = pd.DataFrame(age_absolute_error_tuple, columns=['Age', 'Absolute Error']).set_index('Age')
        grouped_df = df.groupby('Age').agg('mean')

        if self.comet_logger:
            fig = px.bar(df, x=df.index, y='Absolute Error')
            fig.update_layout(
                title=f"Absolute Error per Age, Weighted MAE: {df['Absolute Error'].mean():.2f}",
                xaxis_title="Age",
                yaxis_title="Mean Absolute Error",
                yaxis=dict(range=[0, 45])
            )
            self.logger.experiment.log_image(name='WMAE', image_data=io.BytesIO(fig.to_image(format='png')),
                                             step=self.trainer.current_epoch)
            fig.data = []
        self.log('val_wmae_epoch', torch.tensor(grouped_df.mean().values), prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_wmae_epoch'
        }
