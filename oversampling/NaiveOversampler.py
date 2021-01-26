from collections import defaultdict
from pathlib import Path
from random import sample, uniform, seed

import imagehash
import pandas as pd
from PIL import Image
from tqdm import tqdm

seed(0)


class NaiveOversampler(object):
    def fit_transform(self, dataset_path, result_path):
        Path(result_path + '/images').mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(dataset_path)
        imgs = defaultdict(list)

        for idx, im in df.iterrows():
            imgs[im['age']].append(im['aligned_path'])

        no_samples = len(df) / len(imgs)
        new_imgs = list()

        for i in tqdm(imgs.keys(), total=len(imgs)):
            for j in range(int(no_samples - len(imgs[i]))):
                paths = sample(imgs[i], 2)
                alpha = uniform(0.15, 0.85)
                new_img = Image.blend(Image.open(paths[0]), Image.open(paths[1]), alpha)
                img_hash = str(imagehash.dhash(new_img, hash_size=16)) + ".jpg"
                new_img.save(result_path + '/images/' + img_hash)
                new_imgs.append({'aligned_path': '/images/' + img_hash, 'age': i, 'base_path': str(paths[0])})

        imgs_df = pd.DataFrame(new_imgs)
        imgs_df.to_csv(result_path + "/train.csv")
        print("Done")


if __name__ == "__main__":
    oversampler = NaiveOversampler()
    oversampler.fit_transform(dataset_path='data/imdb-wiki/wiki_crop_aligned/train.csv',
                              result_path='data/imdb-wiki/naive_oversampled')
    oversampler.fit_transform(dataset_path='data/utk-face/utk_face_aligned/train.csv',
                              result_path='data/utk-face/naive_oversampled')
