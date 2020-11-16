from pathlib import Path

import cv2
from tqdm.contrib import tenumerate

from preprocessing.face_detector import FaceDetector
import numpy as np

# calculate_landmarks_from_directory.py for 256 x 256
CALCULATED_COORDS = np.array([
    96.12349935, 119.87868143,
    159.29785576, 119.96951135,
    128.37833404, 157.49470474,
    99.63297665, 186.05553152,
    156.01534961, 186.22217416
])


def main():
    quantity = None
    path = 'data/ffhq/'

    paths = list(Path(path).rglob('*.png'))
    if quantity is not None:
        paths = paths[:quantity]

    face_detector = FaceDetector(weights="models/retina_face_weights/Resnet50_Final.pth")
    calculated_coords = np.zeros((len(paths), 10))
    for i, image_path in tenumerate(paths):
        img_raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img_raw, (256, 256))
        dets, landmarks = face_detector.predict(img_resized)
        calculated_coords[i] = landmarks[0]

    print(np.mean(calculated_coords, axis=0))


if __name__ == '__main__':
    main()
