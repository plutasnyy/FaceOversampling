from pathlib import Path

import click
import cv2
import pandas as pd
from PIL.Image import fromarray
from imagehash import dhash
from skimage.transform import SimilarityTransform
from tqdm import tqdm

from preprocessing.calculate_landmarks_from_directory import CALCULATED_COORDS
from preprocessing.face_detector import FaceDetector


def resize(img, max_height=512, max_width=512):
    height, width = img.shape[:2]

    if max_height < height or max_width < width:
        scaling_factor = max_height / float(height)
        if max_width / float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    return img


@click.command()
@click.option('--path', type=str, help='Path to UTK dataset directory')
@click.option('--quantity', type=int, default=None,
              help='Number of images, to process only part of data when you would like to test script')
def align(path, quantity):
    output_directory: Path = Path(path) / 'utk_face_aligned'
    output_image_directory: Path = output_directory / 'images'
    output_image_directory.mkdir(parents=True, exist_ok=True)

    images_paths = (Path(path) / 'images').rglob('*.jpg')
    images_paths = list(images_paths)[:quantity]

    hashes = list(map(lambda x: x.stem, output_image_directory.rglob('*.jpg')))
    face_detector = FaceDetector(weights='models/retina_face_weights/Resnet50_Final.pth')
    for img_path in tqdm(images_paths, leave=False):  # first align images
        try:
            img_raw = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            img_raw = resize(img_raw, max_height=512, max_width=512)
            hash = dhash(fromarray(img_raw))
            if str(hash) in hashes:
                continue

            dets, landmarks = face_detector.predict(img_raw)
            similarity_transform = SimilarityTransform()
            similarity_transform.estimate(landmarks.reshape((5, 2)), CALCULATED_COORDS.reshape((5, 2)))
            aligned_image = cv2.warpAffine(img_raw, similarity_transform.params[:2, :], (256, 256),
                                           borderMode=cv2.BORDER_REFLECT)
            new_path = output_image_directory / f'{str(hash)}.jpg'

            cv2.imwrite(str(new_path), aligned_image)
        except Exception as e:
            print(f'Error {str(e)}')

    hashes = list(map(lambda x: x.stem, output_image_directory.rglob('*.jpg')))
    result_data = list()

    for img_path in tqdm(images_paths, leave=False):  # then compute dataset based on generated hashes

        img_raw = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img_raw = resize(img_raw, max_height=512, max_width=512)
        hash = dhash(fromarray(img_raw))

        if str(hash) not in hashes:
            continue

        try:
            age, gender, race, timestamp = img_path.stem.split('_')
            new_path = output_image_directory / f'{str(hash)}.jpg'
            result_data.append([str(img_path), str(new_path), hash, 1 - int(gender), int(age), int(race)])

        except Exception as e:
            print(f'Error {str(e)}')

    df = pd.DataFrame(result_data, columns=['base_path', 'aligned_path', 'hash', 'gender', 'age', 'race'])
    df = df.set_index('hash')

    df.to_csv(str(output_directory / 'all_data.csv'))


if __name__ == '__main__':
    align()
