from pathlib import Path

import pandas as pd


def map_func(p: str):
    return str(base_path / 'images' / Path(p).name)


base_path = Path('data/utk-face/oversampled')

df = pd.read_csv(str(base_path / 'train_new.csv'), index_col=0)
print(df.head())

df['path'] = df['aligned_path'].apply(map_func)
df = df.drop('aligned_path', axis=1)
print(df.head())

df.to_csv(str(base_path / 'train.csv'))
