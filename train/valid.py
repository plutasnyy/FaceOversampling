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

experiment = APIExperiment(api_key="2ma9DWG8F7ul8RsBQTcXy3pCz", previous_experiment='84d48a362c8744b0a9c7d6f3e5f19b8a')
experiment.download_model(name='epoch=31-val_mae=6.77.ckpt', output_path='comet-ml/', expand=True)

data_module = FaceDataModule('data/imdb-wiki/wiki_crop_aligned', batch_size=32)
model = MobileNetLightingModel().load_from_checkpoint('comet-ml/epoch=31-val_mae=6.77.ckpt')

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
    title=f"Absolute Error per Age, mae {df['ae'].mean():.2f}",
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
