from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('max_columns', 500)
pd.set_option('max_colwidth', 10000)


@click.command()
@click.option('--path', type=str, help='Path to csv with dataset metadata')
def main(path):
    file_path: Path = Path(path)
    X = pd.read_csv(str(file_path))

    X = X[X.groupby('age').hash.transform(len) >= 2]
    y = X['age'].to_frame()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    X = X_train[X_train.groupby('age').hash.transform(len) >= 2]
    y = X['age'].to_frame()
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.25)

    X_train.set_index('hash')
    X_val.set_index('hash')
    X_test.set_index('hash')

    X_train.to_csv(file_path.parent / 'train.csv')
    X_val.to_csv(file_path.parent / 'val.csv')
    X_test.to_csv(file_path.parent / 'test.csv')


if __name__ == '__main__':
    main()
