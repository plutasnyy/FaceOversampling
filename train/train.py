from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CometLogger

from dataset import FaceDataModule
from model import MobileNetLightingModel

LOGGING_PARAMS = {'comet_ml_logging': True}
LEARNING_PARAMS = {'epochs': 2}
ALL_PARAMS = {**LOGGING_PARAMS, **LEARNING_PARAMS}

data_module = FaceDataModule('data/imdb-wiki/wiki_crop_aligned')
model = MobileNetLightingModel()

comet_logger = CometLogger(api_key="2ma9DWG8F7ul8RsBQTcXy3pCz", project_name="pbp", workspace="plutasnyy")
comet_logger.log_hyperparams(ALL_PARAMS)
model_checkpoint = ModelCheckpoint(filepath='checkpoints/{epoch:02d}-{val_mse:.2f}', save_weights_only=True,
                                   save_top_k=3, monitor='val_mse', period=1)
lr_logger = LearningRateMonitor(logging_interval='epoch')

trainer = Trainer(logger=comet_logger, max_epochs=LEARNING_PARAMS['epochs'], callbacks=[model_checkpoint, lr_logger])
trainer.fit(model, datamodule=data_module)

for absolute_path in model_checkpoint.best_k_models.keys():
    comet_logger.experiment.log_model(Path(absolute_path).name, absolute_path)
comet_logger.log_metrics({'best_model_score': model_checkpoint.best_model_score.tolist()})
