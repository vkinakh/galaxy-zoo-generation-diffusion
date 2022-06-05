import argparse

import matplotlib.pyplot as plt

import torch

from src.trainer import GeneratorModule
from src.utils import get_config


def main(args) -> None:
    config_path = args.config
    config = get_config(config_path)
    config['batch_size'] = 1
    path_checkpoint = config['fine_tune_from']

    module = GeneratorModule.load_from_checkpoint(checkpoint_path=path_checkpoint,
                                                  config=config, use_fp16=config['fp16'])
    module.eval()

    for im_in, lbl in module.test_dataloader():
        with torch.no_grad():
            im_out = module.forward(lbl, progress=True)

        im_in_np = im_in.squeeze().detach().cpu().numpy()
        im_in_np = im_in_np.transpose(1, 2, 0)
        im_in_np = im_in_np * 0.5 + 0.5

        im_out_np = im_out.squeeze().detach().cpu().numpy()
        im_out_np = im_out_np.transpose(1, 2, 0)
        im_out_np = im_out_np * 0.5 + 0.5

        plt.figure()
        plt.title('Input')
        plt.imshow(im_in_np)
        plt.show()

        plt.figure()
        plt.title('Output')
        plt.imshow(im_out_np)
        plt.show()

        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/generator.yaml')
    args = parser.parse_args()
    main(args)
