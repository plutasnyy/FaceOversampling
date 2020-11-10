from datetime import datetime
from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
from PIL.Image import fromarray
from imagehash import dhash
from scipy.io import loadmat
from skimage.transform import SimilarityTransform
from tqdm.contrib import tenumerate

from preprocessing.calculate_landmarks_from_directory import CALCULATED_COORDS
from preprocessing.face_detector import FaceDetector


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]
    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


def load_data(mat_path):
    d = loadmat(mat_path)
    return d["image"], d["gender"][0], d["age"][0], d["db"][0], d["img_size"][0, 0], d["min_score"][0, 0]


@click.command()
@click.option('--path', type=str, help='Path to IMDB dataset directory')
@click.option('--quantity', type=int, default=None,
              help='Number of images, to process only part of data when you would like to test script')
def align(path, quantity):
    output_directory: Path = Path(path + '_aligned')
    output_image_directory: Path = output_directory / 'images'
    output_image_directory.mkdir(parents=True, exist_ok=True)

    face_detector = FaceDetector(weights='models/retina_face_weights/Resnet50_Final.pth')
    meta = get_meta('data/imdb-wiki/wiki_crop/wiki.mat', 'wiki')
    iteration_number: int = len(meta[0]) if quantity is None else max(len(meta[0]), quantity)
    result_data = list()
    for i, (full_path, dob, gender, photo_taken, face_score, second_face_score, age) in \
            tenumerate(zip(*meta), total=iteration_number):
        if quantity is not None and i >= quantity:
            break

        if face_score < 1:
            continue

        if (not np.isnan(second_face_score)) and second_face_score > 0.0:
            continue

        if not 0 <= age <= 100:
            continue

        if np.isnan(gender):
            continue

        try:
            complete_image_path = Path(path) / str(full_path[0])
            img_raw = cv2.imread(str(complete_image_path), cv2.IMREAD_COLOR)

            dets, landmarks = face_detector.predict(img_raw)
            similarity_transform = SimilarityTransform()
            similarity_transform.estimate(landmarks.reshape((5, 2)), CALCULATED_COORDS.reshape((5, 2)))
            aligned_image = cv2.warpAffine(img_raw, similarity_transform.params[:2, :], (256, 256),
                                           borderMode=cv2.BORDER_REFLECT)
            hash = dhash(fromarray(img_raw))
            new_path = output_image_directory / f'{str(hash)}.jpg'

            result_data.append([complete_image_path, str(new_path), hash, int(gender), int(age)])
            cv2.imwrite(str(new_path), aligned_image)
        except Exception as e:
            print(f'Error {str(e)}')

    df = pd.DataFrame(result_data, columns=['base_path', 'aligned_path', 'hash', 'gender', 'age'])
    df = df.set_index('hash')

    df.to_csv(str(output_directory / 'all_data.csv'))


if __name__ == '__main__':
    align()
