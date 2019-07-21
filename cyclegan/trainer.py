import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import autograd, nn, optim
from torch.utils import data
from torchvision.utils import save_image
from tqdm import tqdm, trange

from .metrics import Average


def l1_loss(x, y):
    return (x - y).abs().mean()


def mse(x, y):
    return (x - y).pow(2).mean()


class CycleGanTrainer(object):

    def __init__(self, models: dict, optimizers: dict, schedulers, dataloaders: List[data.DataLoader],
                 device: torch.device):
        self.gx, self.gy = models['G'], models['F']
        self.dx, self.dy = models['DX'], models['DY']
        self.optimizer_g, self.optimizer_d = optimizers['G'], optimizers['D']
        self.scheduler_g, self.scheduler_d = schedulers
        self.loader_x, self.loader_y = dataloaders
        self.device = device

    def set_train(self):
        self.gx.train()
        self.gy.train()
        self.dx.train()
        self.dy.train()

    def set_eval(self):
        self.gx.eval()
        self.gy.eval()
        self.dx.eval()
        self.dy.eval()

    def fit(self, num_epochs=1):
        epochs = trange(1, num_epochs + 1, desc='Epochs', ncols=0)
        for epoch in epochs:
            if epoch >= 100:
                self.scheduler_d.step()
                self.scheduler_g.step()

            loss_d, loss_g = self.train()
            self.plot_sample(epoch)

            epochs.set_postfix_str(f'loss_d: {loss_d}, loss_g: {loss_g}')

    def train(self):
        self.set_train()

        avg_loss_d = Average()
        avg_loss_g = Average()

        dataloader = tqdm(zip(self.loader_x, self.loader_y), desc='Iterations', ncols=0)
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            # train generators
            fake_x = self.gx(y)
            fake_y = self.gy(x)

            rec_x = self.gx(fake_y)
            rec_y = self.gy(fake_x)

            loss_cyc = l1_loss(rec_x, x) + l1_loss(rec_y, y)
            loss_g = mse(self.dx(fake_x), 1) + mse(self.dy(fake_y), 1) + 10.0 * loss_cyc

            self.optimizer_g.zero_grad()
            loss_g.backward()
            self.optimizer_g.step()

            # train discriminator
            fake_x = fake_x.detach()
            fake_y = fake_y.detach()

            loss_dx = (mse(self.dx(x), 1) + mse(self.dx(fake_x), 0)) / 2.0
            loss_dy = (mse(self.dy(y), 1) + mse(self.dy(fake_y), 0)) / 2.0

            loss_d = loss_dx + loss_dy

            self.optimizer_d.zero_grad()
            loss_d.backward()
            self.optimizer_d.step()

            # update metrics
            avg_loss_d.update(loss_dx.item() + loss_dy.item(), number=x.size(0))
            avg_loss_g.update(loss_g.item(), number=x.size(0))

            dataloader.set_postfix_str(f'loss_d: {avg_loss_d}, loss_g: {avg_loss_g}')

        return avg_loss_d, avg_loss_g

    def plot_sample(self, epoch):
        self.set_eval()

        with torch.no_grad():
            os.makedirs('output_dir', exist_ok=True)

            x = next(iter(self.loader_x)).to(self.device)
            y = next(iter(self.loader_y)).to(self.device)

            fake_x = self.gx(y)
            fake_y = self.gy(x)

            rec_x = self.gx(fake_y)
            rec_y = self.gy(fake_x)

            image = torch.cat([x, fake_y, rec_x, y, fake_x, rec_y])
            save_image(image, f'output_dir/{epoch}.jpg', normalize=True, nrow=x.size(0))

    def gradient_penalty(self, discriminator, real, fake):
        batch_size = real.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=self.device)

        interpolates = epsilon * real + (1 - epsilon) * fake
        interpolates = interpolates.clone().detach().requires_grad_(True)
        gradients = autograd.grad(discriminator(interpolates),
                                  interpolates,
                                  grad_outputs=torch.ones(batch_size, device=self.device),
                                  create_graph=True)[0]

        return (gradients.view(batch_size, -1).norm(2, dim=1) - 1).pow(2)
