
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
# For saving Adversarial Examples
class AEDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

        if len(self.data_frame.columns) < 2:
            raise ValueError("CSV file should include both images and labels columns")

        self.labels = self.data_frame.iloc[:, 0].values
        self.pixel_data = self.data_frame.iloc[:, 1:].values.astype(np.float32)
        self.images = self.pixel_data.reshape(-1, 3, 32, 32)
        # To make sure ae_transform
        self.images = self.images.transpose(0, 2, 3, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label