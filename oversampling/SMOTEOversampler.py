from collections import defaultdict, Counter
from pathlib import Path
from random import uniform, seed

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

seed(0)


class SMOTEOversampler:
    def __init__(self):
        self.X, self.y = None, None
        self.quantity = 0
        self.counter = None
        self.nn_cache = dict()

    def fit(self, dataset_path):
        df = pd.read_csv(dataset_path)
        imgs = defaultdict(list)

        x, y = [], []
        for idx, im in df.iterrows():
            x.append(np.array(Image.open(im['aligned_path'])).ravel())
            y.append(im['age'])

            if len(y) > 2000:
                break

        self.X = np.array(x)
        self.y = np.array(y)
        self.quantity: Counter = Counter(y)

    def transform(self, result_path):
        Path(result_path + '/images').mkdir(parents=True, exist_ok=True)
        new_imgs = list()
        max_quantity = int(self.quantity.most_common(1)[0][1] / len(self.quantity)) + 1
        for age, age_quantity in tqdm(self.quantity.items()):
            idx_to_look = np.array(np.where((age == self.y))).ravel()

            threshold = 1
            while len(idx_to_look) <= 15 and threshold <= 5:
                idx_to_look = np.array(np.where((age - threshold <= self.y) & (self.y <= age + threshold))).ravel()
                threshold += 1
            print(age, age_quantity, threshold - 1, len(idx_to_look))

            if len(idx_to_look) <= 15:
                continue

            knn = NearestNeighbors(n_neighbors=5).fit(self.X[idx_to_look, ::25])
            dist, ind = knn.kneighbors(self.X[idx_to_look, ::25])

            for _ in range(max(max_quantity - age_quantity, 0)):
                idx_to_look_in_selected_age = np.where(self.y[idx_to_look] == age)
                random_idx_in_selected_age = np.random.choice(np.array(idx_to_look_in_selected_age).ravel(), 1)

                chosen_nn = np.random.choice(list(range(1, len(ind[0]))), size=1)
                chosen_idx_of_nn = ind[random_idx_in_selected_age, chosen_nn]

                img_origin_idx = idx_to_look[random_idx_in_selected_age]
                img_idx = idx_to_look[chosen_idx_of_nn[0]]
                h = uniform(0.1, 0.9)

                new_X = (self.X[img_origin_idx] * h + self.X[img_idx] * (1 - h)).astype(np.uint8)
                new_y = np.round(self.y[img_origin_idx] * h + self.y[img_idx] * (1 - h)).astype(np.uint8)[0]

                new_img = Image.fromarray(new_X.reshape((256, 256, 3)))
                img_hash = str(imagehash.dhash(new_img, hash_size=8)) + ".jpg"

                new_img.save(result_path + '/images/' + img_hash)
                new_imgs.append({'aligned_path': '/images/' + img_hash, 'age': new_y})
                break
        imgs_df = pd.DataFrame(new_imgs)
        imgs_df.to_csv(result_path + "/train.csv")
        print("Done")

    def fit_transform(self, dataset_path, result_path):
        self.fit(dataset_path)
        self.transform(result_path)


if __name__ == "__main__":
    oversampler = SMOTEOversampler()
    oversampler.fit_transform(dataset_path='data/utk-face/utk_face_aligned/train.csv',
                              result_path='data/utk-face/smote_oversampled_k_5_s_3')
    # oversampler.fit_transform(dataset_path='data/utk-face/utk_face_aligned/train.csv',
    #                           result_path='data/utk-face/naive_oversampled')
