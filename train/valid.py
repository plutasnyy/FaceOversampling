from configparser import ConfigParser

import click
from comet_ml import APIExperiment
from easydict import EasyDict

from dataset import FaceDataModule
from model import MobileNetLightingModel
from utils import log_mae_per_age


@click.command()
@click.option('--experiment', required=True, type=str, help='For example ce132011516346c99185d139fb23c70c')
@click.option('--weights-path', required=True, type=str, help='For example epoch=25-val_mae=8.2030.ckpt')
def validate(experiment, weights_path):
    config = ConfigParser()
    config.read('config.ini')
    comet_config = EasyDict(config['cometml'])

    dataset_paths = {
        'imdb': 'data/imdb-wiki/wiki_crop_aligned',
        'utk': 'data/utk_face/utk_face_aligned'
    }

    experiment = APIExperiment(api_key=comet_config.apikey, previous_experiment=experiment)
    dataset = experiment.get_parameters_summary("data_path")['valueCurrent']

    experiment.download_model(name=weights_path, output_path='comet-ml/', expand=True)

    data_module = FaceDataModule(dataset_paths[dataset], batch_size=32)
    model = MobileNetLightingModel().load_from_checkpoint('comet-ml/' + weights_path)

    log_mae_per_age(model, data_module.val_dataloader(), experiment)


if __name__ == '__main__':
    validate()
