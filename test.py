import numpy as np

from sklearn.manifold import TSNE

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from chamferdist import ChamferDistance

from src.model import ResNetSimCLR
from src.data import MakeDataLoader
from src.metrics import get_fid_between_datasets, inception_score


# encoder parameters
SIMCLR_PATH = './models/galaxy_zoo_simclr.pth'
ENCODER_DIM = 128
BASE_MODEL = 'resnet50'


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

        n_channels = imgs_gen.shape[-1]
        img_size = imgs_gen.shape[1]

        self.device = device
        self.batch_size = batch_size

        self.make_dl = MakeDataLoader(path_original_images, path_original_labels, img_size)

        # load SimCLR encoder
        self.encoder = ResNetSimCLR(BASE_MODEL, n_channels, ENCODER_DIM).to(self.device)
        self.encoder.load_state_dict(torch.load(SIMCLR_PATH, map_location=self.device))
        self.encoder.eval()

    @torch.no_grad()
    def evaluate(self):
        # fid_orig = self._compute_fid_score()
        # print(f'FID score for original images: {fid_orig}')

        # is_mean, is_std = self._compute_inception_score()
        # print(f'Inception score for original images: {is_mean} +- {is_std}')

        chamfer_dist = self._compute_chamfer_distance()
        print(f'Chamfer distance for generated images: {chamfer_dist}')

    @torch.no_grad()
    def _compute_fid_score(self) -> float:
        """Computes original FID score with the Inception v3 model

        Returns:
            float: FID score
        """

        ds_gen = DatasetFromNumpy(self.imgs_gen, self.labels_gen, return_labels=False)
        fid = get_fid_between_datasets(ds_gen, self.make_dl.dataset_test, self.device, self.batch_size, len(ds_gen))
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

    @torch.no_grad()
    def _compute_chamfer_distance(self):
        """Computes chamfer distance

        Returns:
            float: chamfer distance
        """

        ds_gen = DatasetFromNumpy(self.imgs_gen, self.labels_gen, return_labels=False)
        dl_gen = DataLoader(ds_gen, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True)
        dl_real = self.make_dl.get_data_loader_test(self.batch_size, shuffle=False)
        n_batches = len(dl_gen)

        embeddings = []
        i = 0
        for (img, _) in dl_real:
            img = img.to(self.device)
            with torch.no_grad():
                h, _ = self.encoder(img)
            embeddings.append(h.cpu().numpy())

            i += 1
            if i == n_batches:
                break

        for img in dl_gen:
            img = img.to(self.device)
            with torch.no_grad():
                h, _ = self.encoder(img)
            embeddings.append(h.cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        tsne_emb = TSNE(n_components=3).fit_transform(embeddings)
        n = len(tsne_emb)
        tsne_real = tsne_emb[:n // 2]
        tsne_gen = tsne_emb[n // 2:]

        tsne_real = torch.from_numpy(tsne_real).unsqueeze(0)
        tsne_fake = torch.from_numpy(tsne_gen).unsqueeze(0)

        chamfer_dist = ChamferDistance()
        return chamfer_dist(tsne_real, tsne_fake).detach().item()


if __name__ == '__main__':
    path_gen_image = './results/classifier_free_70358_steps_250_all/test_generated_images.npy'
    gen_images = np.load(path_gen_image)
    path_labels = './results/classifier_free_70358_steps_250_all/test_labels.npy'
    labels = np.load(path_labels)

    path_data = '/home/kinakh/Datasets/galaxy-zoo/images_training_rev1/'
    path_labels = '/home/kinakh/Datasets/galaxy-zoo/training_solutions_rev1.csv'

    evaluator = Evaluator(path_data, path_labels, gen_images, labels, 'cuda')
    evaluator.evaluate()
