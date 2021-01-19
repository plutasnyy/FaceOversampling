from configparser import ConfigParser

import click
from comet_ml import APIExperiment
from easydict import EasyDict

from model import MobileNetLightingModel
from dataset import FaceDataModule

experiment = 'ce132011516346c99185d139fb23c70c'
weights_path = 'epoch=25-val_mae=8.2030.ckpt'

config = ConfigParser()
config.read('config.ini')
comet_config = EasyDict(config['cometml'])

experiment = APIExperiment(api_key=comet_config.apikey, previous_experiment=experiment)
experiment.download_model(name=weights_path, output_path='comet-ml/', expand=True)

model = MobileNetLightingModel().load_from_checkpoint('comet-ml/' + weights_path)

dataset_paths = {
    'imdb': 'data/imdb-wiki/wiki_crop_aligned',
    'utk': 'data/utk-face/utk_face_aligned'
}
dataset = experiment.get_parameters_summary("data_path")['valueCurrent']
data_module = FaceDataModule(dataset, batch_size=32)

import pandas as pd
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt


def generate_plots(age1, gender1, age2, gender2):
    df = pd.read_csv('/content/FaceOversampling/data/utk-face/utk_face_aligned/test.csv')
    path1 = df[(df['age'] == age1) & (df['gender'] == gender1)].sample(1)['aligned_path'].values[0]
    path2 = df[(df['age'] == age2) & (df['gender'] == gender2)].sample(1)['aligned_path'].values[0]
    image1, image2 = Image.open(path1), Image.open(path2)
    tensor_image1, tensor_image2 = oversampler.transform(image1), oversampler.transform(image2)

    images = list()
    images.append(image1)
    for i in range(0, 11, 2):
        new_img = tensor2im(oversampler.interpolate(tensor_image2, tensor_image1, i / 10)).resize((256, 256))
        images.append(new_img)
    images.append(image2)

    preds = list()
    for img in images:
        batch = data_module.preprocess_valid(image=np.array(img))['image'].unsqueeze(0)
        with torch.no_grad():
            predicted_age = float(model.forward(batch)[0][0])
        preds.append(predicted_age)

    titles = list()
    titles.append(f'Age: {age1}, pred: {preds[0]:.2f}')
    for img, pred, alpha in zip(images[1:-1], preds[1:-1], list(range(0, 11, 2))):
        expected_age = age2 * alpha / 10 + age1 * (1 - alpha / 10)
        title = f'Alpha: {alpha / 10}, Age: {expected_age}, Pred: {pred:.2f}'
        titles.append(title)
    titles.append(f'Age: {age2}, pred: {preds[-1]:.2f}')

    fig, ax = plt.subplots(1, 8, clear=True, figsize=(40, 20))

    for i, (img, title) in enumerate(zip(images, titles)):
        ax[i].imshow(img)
        ax[i].set_title(title)
        ax[i].set_xticks([], [])
        ax[i].set_yticks([], [])
    plt.subplots_adjust(wspace=0.1, hspace=0)
    plt.savefig(f'{age1}_{gender1}_{age2}_{gender2}.jpg')
    plt.show()
