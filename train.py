import argparse
import itertools

import torch
from torch.optim import lr_scheduler

from cyclegan.datasets import DataFactory
from cyclegan.models import ModelFactory
from cyclegan.optim import OptimFactory
from cyclegan.trainer import CycleGanTrainer
from cyclegan.utils import AttrDict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/default.yaml')
    parser.add_argument('--no-cuda', action='store_true')
    return parser.parse_args()


def load_config():
    args = parse_args()
    config = AttrDict.from_yaml(args.config)

    config.update(vars(args))

    return config


def main():
    config = load_config()

    device = torch.device('cuda' if torch.cuda.is_available() and not config.no_cuda else 'cpu')

    models = {
        'G': ModelFactory.create(**config.generator),
        'F': ModelFactory.create(**config.generator),
        'DX': ModelFactory.create(**config.discriminator),
        'DY': ModelFactory.create(**config.discriminator),
    }

    for model in models.values():
        model.to(device)

    params_g = itertools.chain(models['G'].parameters(), models['F'].parameters())
    params_d = itertools.chain(models['DX'].parameters(), models['DY'].parameters())
    optimizers = {
        'G': OptimFactory.create(params_g, **config.optimizer_g),
        'D': OptimFactory.create(params_d, **config.optimizer_d)
    }

    dataloaders = DataFactory.create(**config.dataloader)

    scheduler_g = lr_scheduler.CosineAnnealingLR(optimizers['G'], T_max=100, eta_min=0)
    scheduler_d = lr_scheduler.CosineAnnealingLR(optimizers['D'], T_max=100, eta_min=0)
    schedulers = [scheduler_g, scheduler_d]

    trainer = CycleGanTrainer(
        models,
        optimizers,
        schedulers,
        dataloaders,
        device,
    )
    trainer.fit(200)


if __name__ == '__main__':
    main()
