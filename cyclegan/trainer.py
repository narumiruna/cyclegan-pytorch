from typing import List

import torch
from torch import autograd, nn, optim
from torch.utils import data
from tqdm import tqdm

from .metrics import Average


def mse(x, y):
    return (x - y).pow(2).mean()


class CycleGanTrainer(object):

    def __init__(self, generators: List[nn.Module], discriminators: List[nn.Module], optimizers: List[optim.Optimizer],
                 dataloaders: List[data.DataLoader], device: torch.device):
        self.netG_A, self.netG_B = generators
        self.netD_A, self.netD_B = discriminators
        self.optimizers_G, self.optimizers_D = optimizers
        self.train_loader_A, self.train_loader_B = dataloaders
        self.device = device

    def set_train(self):
        self.netG_A.train()
        self.netG_B.train()
        self.netD_A.train()
        self.netD_B.train()

    def set_eval(self):
        self.netG_A.eval()
        self.netG_B.eval()
        self.netD_A.eval()
        self.netD_B.eval()

    def fit(self, num_epochs=1):
        for epoch in range(1, num_epochs + 1):
            self.train()

    def train(self):
        self.set_train()

        train_loss_d = Average()

        for i, (real_A, real_B) in enumerate(zip(self.train_loader_A, self.train_loader_B)):
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)

            fake_A = self.netG_A(real_B).detach()
            fake_B = self.netG_B(real_A).detach()

            loss_d = mse(self.netD_A(real_A), 1) + mse(self.netD_A(fake_A), -1)
            loss_d += mse(self.netD_B(real_B), 1) + mse(self.netD_B(fake_B), -1)

            self.optimizers_D.zero_grad()
            loss_d.backward()
            self.optimizers_D.step()

            train_loss_d.update(loss_d.item(), number=real_A.size(0) + real_B.size(0))
            print(f'Iter: {i+1}, discriminator loss: {train_loss_d}')

    def gradient_penalty(self, netD, real, fake):
        batch_size = real.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=self.device)

        interpolates = epsilon * real + (1 - epsilon) * fake
        interpolates = interpolates.clone().detach().requires_grad_(True)
        gradients = autograd.grad(netD(interpolates),
                                  interpolates,
                                  grad_outputs=torch.ones(batch_size, device=self.device),
                                  create_graph=True)[0]

        return (gradients.view(batch_size, -1).norm(2, dim=1) - 1).pow(2)

    def init_nets(self):
        self.netG_A.to(self.device)
        self.netG_B.to(self.device)
        self.netD_A.to(self.device)
        self.netD_B.to(self.device)
