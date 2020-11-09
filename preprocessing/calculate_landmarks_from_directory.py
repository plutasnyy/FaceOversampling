from pathlib import Path

import cv2
from tqdm.contrib import tenumerate

from preprocessing.face_detector import FaceDetector
import numpy as np

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
