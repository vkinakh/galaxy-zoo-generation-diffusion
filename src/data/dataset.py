from glob import glob
from PIL import Image

import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms


def get_labels_train(file_galaxy_labels) -> torch.Tensor:
    df_galaxy_labels = read_csv(file_galaxy_labels)
    labels_train = df_galaxy_labels[df_galaxy_labels.columns[1:]].values
    labels_train = torch.from_numpy(labels_train).float()
    return labels_train


class GalaxyZooDataset(Dataset):
    def __init__(self,
                 folder_images,
                 file_labels,
                 image_size: int):
        self.path_images = folder_images
        file_list = glob(self.path_images + "*")
        file_list.sort()
        labels = get_labels_train(file_labels)  # labels are sorted already
        self.test_data = False
        self.data = []
        for file, label in zip(file_list, labels):
            self.data.append([file, label])

        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine((0, 360), (0.01,) * 2),  # rotation, -4 to 4 translation
            transforms.CenterCrop(207),
            transforms.Resize((image_size,) * 2),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)  # images normalized to [-1, 1]
        ])

        self.augment_test = transforms.Compose([
            transforms.CenterCrop(207),
            transforms.Resize((image_size,) * 2),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)  # images normalized to [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        file, label = self.data[index]
        img = Image.open(file)
        if self.test_data:
            img = self.augment_test(img)
        else:
            img = self.augment(img)
        return img, label


class MakeDataLoader:

    """Class that creates train, valid and test datasets/dataloaders"""

    def __init__(self,
                 folder_images,
                 file_labels,
                 image_size: int,
                 test_size: float = 0.1,
                 random_state=2,
                 augmented: bool = True):
        self.dataset = GalaxyZooDataset(folder_images, file_labels, image_size)
        if not augmented:
            self.dataset.test_data = True
        train_idx, test_idx = train_test_split(list(range(len(self.dataset))), test_size=test_size,
                                               random_state=random_state)
        valid_idx, test_idx = train_test_split(list(range(len(test_idx))), test_size=0.5, random_state=random_state + 1)

        self.dataset_train = Subset(self.dataset, np.array(train_idx))
        self.dataset_valid = Subset(self.dataset, valid_idx)
        self.dataset_test = Subset(self.dataset, test_idx)
        self.dataset_test.test_data = True

    def get_data_loader_full(self,
                             batch_size: int = 64,
                             shuffle: bool = True, **kwargs) -> DataLoader:
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True,
                          pin_memory=False, **kwargs)

    def get_data_loader_train(self,
                              batch_size: int = 64,
                              shuffle: bool = True, **kwargs) -> DataLoader:
        return DataLoader(self.dataset_train, batch_size=batch_size, shuffle=shuffle, drop_last=True,
                          pin_memory=False, **kwargs)

    def get_data_loader_test(self,
                             batch_size: int = 64,
                             shuffle: bool = False, **kwargs) -> DataLoader:
        return DataLoader(self.dataset_test, batch_size=batch_size, shuffle=shuffle, drop_last=True,
                          pin_memory=False, **kwargs)

    def get_data_loader_valid(self, batch_size: int = 64,
                              shuffle: bool = False,
                              **kwargs) -> DataLoader:
        return DataLoader(self.dataset_valid, batch_size=batch_size, shuffle=shuffle, drop_last=True,
                          pin_memory=False, **kwargs)
