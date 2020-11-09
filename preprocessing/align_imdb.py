from datetime import datetime
from pathlib import Path

import click
import numpy as np
from PIL import Image
from imagehash import dhash
from scipy.io import loadmat
from tqdm.contrib import tenumerate

# calculate_landmarks_from_directory.py for 256 x 256
from preprocessing.face_detector import FaceDetector

CALCULATED_COORDS = np.array([
    96.12349935, 119.87868143,
    159.29785576, 119.96951135,
    128.37833404, 157.49470474,
    99.63297665, 186.05553152,
    156.01534961, 186.22217416
])


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

    paths = list(Path(path).rglob('*.jpg'))
    if quantity is not None:
        paths = paths[:quantity]

    # face_detector = FaceDetector(weights='preprocessing/Pytorch_Retinaface/weights/Resnet50_Final.pth')
    meta = get_meta('data/imdb-wiki/wiki_crop/wiki.mat', 'wiki')
    for id, (full_path, dob, gender, photo_taken, face_score, second_face_score, age) in enumerate(zip(*meta)):
        hash = dhash(Image.open(str(image_path)))
        pass


if __name__ == '__main__':
    align()
