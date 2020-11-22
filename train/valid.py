from configparser import ConfigParser

from comet_ml import APIExperiment
from easydict import EasyDict

from dataset import FaceDataModule
from model import MobileNetLightingModel
from utils import log_mae_per_age

experiment_id = 'c30cb22c7a4b4866b2825b79fdf9862f'
model_name = 'epoch=42-val_mae=7.3106.ckpt'
dataset_path = 'data/imdb-wiki/wiki_crop_aligned'

config = ConfigParser(dict_type=EasyDict)
config.read('config.ini')

experiment = APIExperiment(api_key=config._sections.cometml.apikey, previous_experiment=experiment_id)
experiment.download_model(name=model_name, output_path='comet-ml/', expand=True)

data_module = FaceDataModule(dataset_path, batch_size=32)
model = MobileNetLightingModel().load_from_checkpoint('comet-ml/' + model_name)

log_mae_per_age(model, data_module.val_dataloader, experiment)
