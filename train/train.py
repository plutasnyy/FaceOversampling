from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CometLogger

from dataset import FaceDataModule
from model import MobileNetLightingModel

LOGGING_PARAMS = {'comet_ml_logging': True, 'description': ''}
LEARNING_PARAMS = {'epochs': 90, 'data_path': 'data/imdb-wiki/wiki_crop_aligned', 'batch_size': 32,
                   'weighted_samples': True}
ALL_PARAMS = {**LOGGING_PARAMS, **LEARNING_PARAMS}

logger, callbacks = False, list()
if LOGGING_PARAMS['comet_ml_logging']:
    logger = CometLogger(api_key="2ma9DWG8F7ul8RsBQTcXy3pCz", project_name="pbp", workspace="plutasnyy")
    logger.experiment.set_code(filename='train/train.py', overwrite=True)
    logger.log_hyperparams(ALL_PARAMS)
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

model_checkpoint = ModelCheckpoint(filepath='checkpoints/{epoch:02d}-{val_mae:.4f}', save_weights_only=True,
                                   save_top_k=3, monitor='val_mae', period=1)
early_stop_callback = EarlyStopping(monitor='val_mae', min_delta=0.01, patience=10, verbose=True, mode='min')
callbacks.extend([model_checkpoint, early_stop_callback])

data_module = FaceDataModule(LEARNING_PARAMS['data_path'], batch_size=LEARNING_PARAMS['batch_size'])
model = MobileNetLightingModel()

trainer = Trainer(logger=logger, min_epochs=20, max_epochs=LEARNING_PARAMS['epochs'], callbacks=callbacks, gpus=1)
trainer.fit(model, datamodule=data_module)

if LOGGING_PARAMS['comet_ml_logging']:
    for absolute_path in model_checkpoint.best_k_models.keys():
        logger.experiment.log_model(Path(absolute_path).name, absolute_path)
    logger.log_metrics({'best_model_score': model_checkpoint.best_model_score.tolist()})
