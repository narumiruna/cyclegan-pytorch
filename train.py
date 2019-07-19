import argparse
import itertools

import torch
from torch import optim

from cyclegan.datasets import ImageFolderLoader
from cyclegan.models import Discriminator, ResnetGenerator
from cyclegan.trainer import CycleGanTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--batch-size', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    dx = Discriminator()
    dy = Discriminator()
    gx = ResnetGenerator()
    gy = ResnetGenerator()

    models = [dx, dy, gx, gy]
    for m in models:
        m.to(device)

    loader_x = ImageFolderLoader('data/apple2orange/trainA',
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=8)
    loader_y = ImageFolderLoader('data/apple2orange/trainB',
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=8)

    optimizer_g = optim.Adam(itertools.chain(gx.parameters(), gy.parameters()), lr=2e-4, betas=(0.5, 0.999))
    optimizer_dx = optim.Adam(dx.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_dy = optim.Adam(dy.parameters(), lr=2e-4, betas=(0.5, 0.999))

    trainer = CycleGanTrainer(
        [gx, gy],
        [dx, dy],
        [optimizer_g, optimizer_dx, optimizer_dy],
        [loader_x, loader_y],
        device,
    )
    trainer.fit(200)


if __name__ == '__main__':
    main()
