import numpy as np
import pandas as pd
import plotly.express as px
import torch
from comet_ml import APIExperiment
from sklearn.metrics import mean_absolute_error
from sklearn.utils import compute_sample_weight, compute_class_weight
from tqdm import tqdm

from dataset import FaceDataModule
from model import MobileNetLightingModel

experiment_id = '3220f193c2d0449ab6fc17317def1ed3'
model_name = 'epoch=26-val_mae=5.89.ckpt'
dataset_path = 'data/utk-face/utk_face_aligned'

experiment = APIExperiment(api_key="2ma9DWG8F7ul8RsBQTcXy3pCz", previous_experiment=experiment_id)
experiment.download_model(name=model_name, output_path='comet-ml/', expand=True)

data_module = FaceDataModule(dataset_path, batch_size=32)
model = MobileNetLightingModel().load_from_checkpoint('comet-ml/' + model_name)

y_list, y_pred_list = list(), list()
for x, y in tqdm(data_module.val_dataloader()):
    with torch.no_grad():
        y_pred = model(x)
    y_list.extend(y.tolist())
    y_pred_list.extend(y_pred.squeeze().tolist())

df = pd.DataFrame({
    'y': y_list,
    'ae': np.abs(np.array(y_list) - np.array(y_pred_list))
})
df = df.groupby(['y']).mean()

fig = px.bar(df, x=df.index, y='ae')
fig.update_layout(
    title=f"Absolute Error per Age, Weighted MAE: {df['ae'].mean():.2f}",
    xaxis_title="Age",
    yaxis_title="Mean Absolute Error",
)
fig.write_image('comet-ml/tmp_img.png')
experiment.log_image('comet-ml/tmp_img.png', image_name='Absolute Error per Age')

class_weight = compute_class_weight('balanced', np.unique(y_list), y_list)
fig = px.scatter(class_weight)
fig.update_layout(
    title="Class weights",
    xaxis_title="Age",
    yaxis_title="Weight",
)

fig.write_image('comet-ml/tmp_img.png')
experiment.log_image('comet-ml/tmp_img.png', image_name='Class weights')

sample_weight = compute_sample_weight('balanced', y_list)
mae = mean_absolute_error(y_list, y_pred_list, sample_weight=sample_weight)
experiment.log_metrics({'weighted mae': mae})
