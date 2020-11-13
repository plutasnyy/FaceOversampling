from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from skimage import io

import pandas as pd


class FaceDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size=64):
        super().__init__()
        self.data_dir: Path = Path(data_dir)
        self.batch_size = batch_size
        self.preprocess = transforms.Compose([  # TODO match with SSRnet, use __init__ for that too
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def train_dataloader(self):
        return DataLoader(FaceImagesDataset(self.data_dir / 'train.csv', transform=self.preprocess),
                          batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(FaceImagesDataset(self.data_dir / 'val.csv', transform=self.preprocess),
                          batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(FaceImagesDataset(self.data_dir / 'test.csv', transform=self.preprocess),
                          batch_size=self.batch_size)


class FaceImagesDataset(Dataset):
    def __init__(self, csv_file: Path, transform=None):
        self.data = pd.read_csv(str(csv_file))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = Path(self.data.iloc[idx, 'aligned_path'])
        age = Path(self.data.iloc[idx, 'age'])
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return image, age
