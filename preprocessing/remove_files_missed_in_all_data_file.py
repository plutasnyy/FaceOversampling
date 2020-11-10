from pathlib import Path

import click
import pandas as pd

pd.set_option('max_columns', 500)
pd.set_option('max_colwidth', 10000)


@click.command()
@click.option('--path', type=str, help='Path to csv with dataset metadata')
def main(path):
    file_path: Path = Path(path)
    hashes = pd.read_csv(str(file_path))['hash'].tolist()
    images_paths = (file_path.parent / 'images').rglob('*.jpg')
    filtered_images_paths = filter(lambda x: x.stem not in hashes, images_paths)
    for i, path in enumerate(filtered_images_paths):
        path.unlink(missing_ok=True)
        
    print(f'Done, removed {i + 1} images.')


if __name__ == '__main__':
    main()
