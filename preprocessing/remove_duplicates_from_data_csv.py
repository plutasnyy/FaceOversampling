from pathlib import Path

import click
import pandas as pd

pd.set_option('max_columns', 500)
pd.set_option('max_colwidth', 10000)


@click.command()
@click.option('--path', type=str, help='Path to csv with dataset metadata')
def main(path):
    file_path: Path = Path(path)
    df = pd.read_csv(file_path)
    hashes_to_remove = (df['hash'].value_counts()).reset_index()
    hashes_in_list = hashes_to_remove[hashes_to_remove['hash'] >= 2]['index'].tolist()
    df = df[~df.hash.isin(hashes_in_list)]

    print(len(df))
    print(df.head())
    print(df['hash'].value_counts())

    df = df.set_index('hash')
    df.to_csv(str(file_path.parent / 'all_data.csv'))


if __name__ == '__main__':
    main()
