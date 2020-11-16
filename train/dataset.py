from pathlib import Path

import albumentations as A
import cv2
import pandas as pd
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


class FaceDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size=2):
        super().__init__()
        self.data_dir: Path = Path(data_dir)
        self.batch_size = batch_size

        self.preprocess_train = A.Compose([
            A.Resize(224, 224),
            A.Rotate(limit=20, p=0.25),
            A.OpticalDistortion(p=0.25),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.ChannelShuffle(p=0.5),
            ToTensorV2()
        ])
        self.preprocess_valid = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def train_dataloader(self):
        return DataLoader(FaceImagesDataset(self.data_dir / 'train.csv', transform=self.preprocess_train),
                          batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(FaceImagesDataset(self.data_dir / 'val.csv', transform=self.preprocess_valid),
                          batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(FaceImagesDataset(self.data_dir / 'test.csv', transform=self.preprocess_valid),
                          batch_size=self.batch_size)


class FaceImagesDataset(Dataset):
    def __init__(self, csv_file: Path, transform=None):
        self.data = pd.read_csv(str(csv_file))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = Path(self.data.iloc[idx]['aligned_path'])
        image = cv2.imread(str(img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        age = int(self.data.iloc[idx]['age'])

        if self.transform:
            image = self.transform(image=image)['image']

        return image, age
