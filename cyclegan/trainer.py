import os
from typing import List

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

    def __init__(self, generators: List[nn.Module], discriminators: List[nn.Module], optimizers: List[optim.Optimizer],
                 dataloaders: List[data.DataLoader], device: torch.device):
        self.netG_A, self.netG_B = generators
        self.netD_A, self.netD_B = discriminators
        self.optimizer_G, self.optimizer_DA, self.optimizer_DB = optimizers
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
        epochs = trange(1, num_epochs + 1, desc='Epochs', ncols=0)
        for epoch in epochs:
            loss_d, loss_g = self.train()
            self.plot_sample(epoch)

            epochs.set_postfix_str(f'loss_d: {loss_d}, loss_g: {loss_g}')

    def train(self):
        self.set_train()

        avg_loss_d = Average()
        avg_loss_g = Average()

        dataloader = tqdm(zip(self.train_loader_A, self.train_loader_B), desc='Iterations', ncols=0)
        for real_A, real_B in dataloader:
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)

            # train generators
            fake_A = self.netG_A(real_B)
            fake_B = self.netG_B(real_A)

            rec_A = self.netG_A(fake_B)
            rec_B = self.netG_B(fake_A)

            # idt_A = self.netG_A(real_A)
            # idt_B = self.netG_B(real_B)
            # loss_idt = l1_loss(idt_A, real_A) + l1_loss(idt_B, real_B)

            loss_cyc = l1_loss(rec_A, real_A) + l1_loss(rec_B, real_B)
            loss_g = mse(self.netD_A(fake_A), 1) + mse(self.netD_B(fake_B), 1) + 10.0 * loss_cyc

            self.optimizer_G.zero_grad()
            loss_g.backward()
            self.optimizer_G.step()

            # train discriminator
            fake_A = fake_A.detach()
            fake_B = fake_B.detach()

            loss_da = (mse(self.netD_A(real_A), 1) + mse(self.netD_A(fake_A), 0)) / 2.0
            loss_db = (mse(self.netD_B(real_B), 1) + mse(self.netD_B(fake_B), 0)) / 2.0

            self.optimizer_DA.zero_grad()
            loss_da.backward()
            self.optimizer_DA.step()

            self.optimizer_DB.zero_grad()
            loss_db.backward()
            self.optimizer_DB.step()

            # update metrics
            avg_loss_d.update(loss_da.item() + loss_db.item(), number=real_A.size(0))
            avg_loss_g.update(loss_g.item(), number=real_A.size(0))

            dataloader.set_postfix_str(f'loss_d: {train_loss_d}, loss_g: {train_loss_g}')

        return avg_loss_d, avg_loss_g

    def plot_sample(self, epoch):
        self.set_eval()

        with torch.no_grad():
            os.makedirs('output_dir', exist_ok=True)

            real_A = next(iter(self.train_loader_A)).to(self.device)
            real_B = next(iter(self.train_loader_B)).to(self.device)

            fake_A = self.netG_A(real_B)
            fake_B = self.netG_B(real_A)

            rec_A = self.netG_A(fake_B)
            rec_B = self.netG_B(fake_A)

            image = torch.cat([real_A, fake_B, rec_A, real_B, fake_A, rec_B])
            save_image(image, f'output_dir/{epoch}.jpg', normalize=True, nrow=real_A.size(0))

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
