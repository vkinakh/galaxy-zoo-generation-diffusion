import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data import MakeDataLoader
from src.metrics import get_fid_between_datasets, inception_score


class DatasetFromNumpy(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, return_labels: bool = True):
        self.data = data
        self.labels = labels
        self.return_labels = return_labels

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    def __getitem__(self, index):
        lbl = self.labels[index]
        img = self.data[index]
        img = self.transform(img)

        if self.return_labels:
            return img, lbl

        return img

    def __len__(self):
        return len(self.data)


class Evaluator:

    def __init__(self, path_original_images: str, path_original_labels: str,
                 imgs_gen: np.ndarray, labels_gen: np.ndarray,
                 device: str, batch_size: int = 64):
        self.imgs_gen = imgs_gen
        self.labels_gen = labels_gen

        self.device = device
        self.batch_size = batch_size

        self.ds_gen = DatasetFromNumpy(imgs_gen, labels_gen)
        self.make_dl = MakeDataLoader(path_original_images, path_original_labels, 64)

    @torch.no_grad()
    def evaluate(self):
        # fid_orig = self._compute_fid_score()
        # print(f'FID score for original images: {fid_orig}')
        is_mean, is_std = self._compute_inception_score()
        print(f'Inception score for original images: {is_mean} +- {is_std}')

    @torch.no_grad()
    def _compute_fid_score(self) -> float:
        """Computes original FID score with the Inception v3 model

        Returns:
            float: FID score
        """

        fid = get_fid_between_datasets(self.ds_gen, self.make_dl.dataset_test, self.device, self.batch_size,
                                       len(self.ds_gen))
        return fid

    @torch.no_grad()
    def _compute_inception_score(self):
        """Computes inception score

        Returns:
            float: inception score
        """

        ds = DatasetFromNumpy(self.imgs_gen, self.labels_gen, return_labels=False)
        score_mean, score_std = inception_score(ds, True, self.batch_size, True, 3)
        return score_mean, score_std


if __name__ == '__main__':
    path_gen_image = './results/classifier_free_70358_steps_250_all/test_generated_images.npy'
    gen_images = np.load(path_gen_image)
    path_labels = './results/classifier_free_70358_steps_250_all/test_labels.npy'
    labels = np.load(path_labels)

    path_data = '/home/kinakh/Datasets/galaxy-zoo/images_training_rev1/'
    path_labels = '/home/kinakh/Datasets/galaxy-zoo/training_solutions_rev1.csv'

    evaluator = Evaluator(path_data, path_labels, gen_images, labels, 'cuda')
    evaluator.evaluate()
