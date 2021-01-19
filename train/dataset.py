from pathlib import Path

import albumentations as A
import cv2
import pandas as pd
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from sklearn.utils import compute_sample_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import seaborn as sns
import matplotlib.pyplot as plt

class FaceDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size=2, weighted_samples=False, oversample=False):
        super().__init__()
        self.data_dir: Path = Path(data_dir)
        self.batch_size = batch_size
        self.weighted_samples = weighted_samples
        self.oversample = oversample

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
        csv_file_path = self.data_dir / 'train.csv'
        df = pd.read_csv(str(csv_file_path))
        sns.displot(df, x="age", discrete=True)
        plt.savefig('utk.jpg')
        if self.oversample:
            oversampled_path = self.data_dir.parent / 'oversampled' / 'train.csv'
            df_oversampled = pd.read_csv(str(oversampled_path))
            df_oversampled = df_oversampled.rename(columns={'path': 'aligned_path'})
            df = pd.concat([df, df_oversampled])

        dataset = FaceImagesDataset(df, transform=self.preprocess_train)

        sampler = None
        shuffle = True
        if self.weighted_samples:
            samples_weight = compute_sample_weight('balanced', dataset.target)
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            shuffle = False

        return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, num_workers=8, shuffle=shuffle)

    def val_dataloader(self):
        csv_file_path = self.data_dir / 'val.csv'
        df = pd.read_csv(str(csv_file_path))
        return DataLoader(FaceImagesDataset(df, transform=self.preprocess_valid),
                          batch_size=self.batch_size, num_workers=8, shuffle=False)

    def test_dataloader(self):
        csv_file_path = self.data_dir / 'test.csv'
        df = pd.read_csv(str(csv_file_path))
        return DataLoader(FaceImagesDataset(df, transform=self.preprocess_valid),
                          batch_size=self.batch_size, num_workers=8, shuffle=False)


class FaceImagesDataset(Dataset):
    def __init__(self, df, transform=None):
        self.data = df
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

    @property
    def target(self):
        return self.data['age'].to_numpy()
