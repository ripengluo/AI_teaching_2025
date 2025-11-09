import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid
import seaborn as sns
import sys
import math
from utils import *



class VAE(nn.Module):

    def __init__(self, latent_dim=10, device="cuda"):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.device = device
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # output (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # output (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # output (128, 8, 8)
            nn.ReLU(),
        )

        # latent mean and variance
        self.mean_layer = nn.Linear(128 * 8 * 8, latent_dim)
        self.logvar_layer = nn.Linear(128 * 8 * 8, latent_dim)

        # decoder
        self.predecode = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # output (64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # output (32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # output (1, 64, 64)
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
        Z = Z.view(Z.size(0), 128, 8, 8)
        return self.decoder(Z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

