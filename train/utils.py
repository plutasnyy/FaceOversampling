import io

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from plotly.io import to_html
from sklearn.metrics import mean_absolute_error
from sklearn.utils import compute_sample_weight
from tqdm import tqdm


def show_batch_of_images(data_module):
    import matplotlib.pyplot as plt

    def denormalise(image):
        image = image.numpy().transpose(1, 2, 0)  # PIL images have channel last
        mean = [0.485, 0.456, 0.406]
        stdd = [0.229, 0.224, 0.225]
        image = (image * stdd + mean).clip(0, 1)
        return image

    example_rows = 4
    example_cols = 8

    images, y = next(iter(data_module.train_dataloader()))
    plt.rcParams['figure.dpi'] = 120  # Increase size of pyplot plots

    fig, axes = plt.subplots(example_rows, example_cols, figsize=(9, 5))  # sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, image, label in zip(axes, images, y):
        ax.imshow(denormalise(image))
        ax.set_axis_off()
        ax.set_title(int(label), fontsize=7)

    fig.subplots_adjust(wspace=0.02, hspace=0)
    fig.suptitle('Augmented training set images', fontsize=20)
    plt.show()


def log_mae_per_age(model, val_dataloader, experiment):
    model.eval()
    model.cuda()

    y_list, y_pred_list = list(), list()
    with torch.no_grad():
        for x, y in tqdm(val_dataloader):
            y_pred = model(x.cuda())
            y_list.extend(y.tolist())
            y_pred_list.extend(y_pred.cpu().numpy().squeeze().tolist())

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
        yaxis=dict(range=[0, 45])
    )
    experiment.log_html(to_html(fig))
    experiment.log_image(name='Best WMAE', image_data=io.BytesIO(fig.to_image(format='png')))

    sample_weight = compute_sample_weight('balanced', y_list)
    mae = mean_absolute_error(y_list, y_pred_list, sample_weight=sample_weight)
    experiment.log_metrics({'final_wmae': mae})

    mae = mean_absolute_error(y_list, y_pred_list)
    experiment.log_metrics({'final_mae': mae})
