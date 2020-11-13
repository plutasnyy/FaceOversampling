import pytorch_lightning as pl

from dataset import FaceDataModule
from model import MobileNetLightingModel

data_module = FaceDataModule('data/imdb-wiki/wiki-crop-aligned')
model = MobileNetLightingModel()
trainer = pl.Trainer()
trainer.fit(model, datamodule=data_module)
