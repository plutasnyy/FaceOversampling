from configparser import ConfigParser
from pathlib import Path

import click
from easydict import EasyDict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CometLogger

from dataset import FaceDataModule
from model import MobileNetLightingModel
from utils import log_mae_per_age


@click.command()
@click.option('-n', '--name', required=True, type=str, help='Name will be visible in CometML')
@click.option('-ds', '--dataset', type=click.Choice(['imdb', 'utk']), required=True)
@click.option('-l', '--loss', type=click.Choice(['mae', 'huber']), default='mae',
              help='Loss used during train/val stage, MAE still will be calculated as a metric')
@click.option('-b', '--beta', type=float, default=1.0,
              help='Threshold used with Huber loss. It is applied only when --loss huber is given')
@click.option('--logger/--no-logger', default=True, help='Flag used to disable logging the data to CometML')
@click.option('-e', '--epochs', default=90, type=int, help='Maximum number of epochs')
@click.option('--seed', default=0, type=int)
@click.option('-bs', '--batch-size', default=32, type=int)
@click.option('--weighted-samples', is_flag=True, default=False, help='It forces equal sampling data in batches based on class')
@click.option('--oversample', is_flag=True, default=False, help='Concatenate oversampled dataset during training')
@click.option('-fdr', is_flag=True, default=False, help='Concatenate oversampled dataset during training')
def train(**params):
    params = EasyDict(params)
    seed_everything(params.seed)

    dataset_paths = {
        'imdb': 'data/imdb-wiki/wiki_crop_aligned',
        'utk': 'data/utk-face/utk_face_aligned'
    }

    config = ConfigParser()
    config.read('config.ini')

    logger, callbacks = False, list()
    if params.logger:
        comet_config = EasyDict(config['cometml'])
        logger = CometLogger(api_key=comet_config.apikey, project_name=comet_config.projectname,
                             workspace=comet_config.workspace)
        logger.experiment.set_code(filename='train/train.py', overwrite=True)
        logger.log_hyperparams(params)
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    model_checkpoint = ModelCheckpoint(filepath='checkpoints/{epoch:02d}-{val_mae:.4f}', save_weights_only=True,
                                       save_top_k=3, monitor='val_mae', period=1)
    early_stop_callback = EarlyStopping(monitor='val_mae', min_delta=0.01, patience=10, verbose=True, mode='min')
    callbacks.extend([model_checkpoint, early_stop_callback])

    data_module = FaceDataModule(data_dir=dataset_paths[params.dataset], batch_size=params.batch_size,
                                 weighted_samples=params.weighted_samples, oversample=params.oversample)

    model = MobileNetLightingModel(loss=params.loss, beta=params.beta)

    trainer = Trainer(logger=logger, max_epochs=params['epochs'], callbacks=callbacks, gpus=1,
                      deterministic=True, fast_dev_run=params.fdr)
    trainer.fit(model, datamodule=data_module)

    if params.logger:
        for absolute_path in model_checkpoint.best_k_models.keys():
            logger.experiment.log_model(Path(absolute_path).name, absolute_path)
        logger.log_metrics({'best_model_score': model_checkpoint.best_model_score.tolist()})

        best_model = MobileNetLightingModel.load_from_checkpoint(model_checkpoint.best_model_path)
        log_mae_per_age(best_model, data_module.val_dataloader(), logger.experiment)


if __name__ == '__main__':
    train()
