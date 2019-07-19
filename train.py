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

    netD_A = Discriminator()
    netD_B = Discriminator()
    netG_A = ResnetGenerator()
    netG_B = ResnetGenerator()

    models = [netD_A, netD_B, netG_A, netG_B]
    for m in models:
        m.to(device)

    train_loader_A = ImageFolderLoader('data/apple2orange/trainA',
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=8)
    train_loader_B = ImageFolderLoader('data/apple2orange/trainB',
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=8)

    optimizer_G = optim.Adam(itertools.chain(netG_A.parameters(), netG_B.parameters()), lr=2e-4, betas=(0.5, 0.999))
    optimizer_DA = optim.Adam(netD_A.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_DB = optim.Adam(netD_B.parameters(), lr=2e-4, betas=(0.5, 0.999))

    trainer = CycleGanTrainer(
        [netG_A, netG_B],
        [netD_A, netD_B],
        [optimizer_G, optimizer_DA, optimizer_DB],
        [train_loader_A, train_loader_B],
        device,
    )
    trainer.fit(200)


if __name__ == '__main__':
    main()
