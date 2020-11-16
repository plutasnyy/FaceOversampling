from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from dataset import FaceDataModule
from model import MobileNetLightingModel

data_module = FaceDataModule('data/imdb-wiki/wiki_crop_aligned')
model = MobileNetLightingModel()
comet_logger = CometLogger(api_key="2ma9DWG8F7ul8RsBQTcXy3pCz", project_name="pbp", workspace="plutasnyy")
model_checkpoint = ModelCheckpoint(filepath='checkpoints/{epoch:02d}-{val_loss:.2f}',
                                   save_weights_only=True,
                                   save_top_k=3,
                                   monitor='val_loss',
                                   period=1)

trainer = Trainer(logger=comet_logger, max_epochs=2, gpus=1, checkpoints=model_checkpoint)
for k in model_checkpoint.best_k_models.keys():
    model_name = 'checkpoints/' + k.split('/')[-1]
    comet_logger.experiment.log_artifact(k, model_name)
comet_logger.experiment.set_property('best_model_score', model_checkpoint.best_model_score.tolist())