import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from PIL import Image
import time
import math
from torch import vmap


class VAE(nn.Module):

    def __init__(self, latent_dim=10, device="cuda"):
        super(VAE, self).__init__()
        self.device = device
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # output (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # output (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # output (128, 8, 8)
            nn.ReLU(),
        )

        # latent mean and variance
        self.mean_layer = nn.Linear(256 * 80 * 80, latent_dim)
        self.logvar_layer = nn.Linear(256 * 80 * 80, latent_dim)

        # decoder
        self.predecode = nn.Linear(latent_dim, 256 * 80 * 80)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # output (64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # output (32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # output (1, 64, 64)
            nn.Sigmoid()
        )

    def encode(self, x):
        X = self.encoder(x)
        X = X.view(x.size(0), -1)
        mean, logvar = self.mean_layer(X), self.logvar_layer(X)
        return mean, logvar

    def reparameterization(self, mean, logvar):
        epsilon = torch.randn_like(logvar).to(self.device)
        z = mean + torch.exp(logvar)*epsilon
        return z

    def decode(self, z):
        Z = self.predecode(z)
        Z = Z.view(Z.size(0), 256, 80, 80)
        return self.decoder(Z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
