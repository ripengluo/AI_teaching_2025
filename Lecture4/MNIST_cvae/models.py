import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torch.nn.functional as F
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid
import seaborn as sns
import sys
import math
from utils import *

class CVAE(nn.Module):
    """Conditional Variational Auto‑Encoder (cVAE) for MNIST‑like 64×64 grayscale images
    -------------------------------------------------------------------------
    x ∈ ℝ^{1×64×64}, y ∈ {0,…,9} (class label)

    Encoding:   q_φ(z | x, y)   – CNN encoder + MLP → μ, logσ²
    Decoding:   p_θ(x | z, y)   – MLP + transposed‑CNN

    Usage
    -----
    >>> model = CVAE(latent_dim=2, num_classes=10, device='cuda')
    >>> x, y = next(iter(dataloader))  # x: (B,1,64,64), y: (B,)
    >>> x = x.to(device); y = y.to(device)
    >>> x_hat, μ, logσ2 = model(x, y)
    """

    def __init__(self, latent_dim: int = 10, num_classes: int = 10, device: str = "cuda"):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.device = device

        # ───── Encoder CNN ────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (32, 32, 32)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (64, 16, 16)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# (128, 8, 8)
            nn.ReLU(inplace=True),
        )
        enc_flat_dim = 128 * 8 * 8  # 8192

        # Mean & log‑variance layers (conditioned on y)
        self.fc_mu     = nn.Linear(enc_flat_dim + num_classes, latent_dim)
        self.fc_logvar = nn.Linear(enc_flat_dim + num_classes, latent_dim)

        # ───── Decoder MLP + transpose CNN ────────────────────────────────────
        self.fc_predecode = nn.Linear(latent_dim + num_classes, enc_flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (64,16,16)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # (32,32,32)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1,  kernel_size=3, stride=2, padding=1, output_padding=1),   # (1,64,64)
            nn.Sigmoid()
        )

    # ────────────────────────────── Helpers ──────────────────────────────────
    def _one_hot(self, y):
        """Turn labels into one‑hot vectors on the right device.
        Accepts Python int / list / tuple or any Tensor dtype."""
        # ensure we have a tensor on the correct device
        if not torch.is_tensor(y):
            y = torch.tensor(y, device=self.device)
        # cast to integer class indices
        y = y.to(torch.long)
        return F.one_hot(y, num_classes=self.num_classes).float()

    # ────────────────────────────── Encoder ──────────────────────────────────
    def encode(self, x: torch.Tensor, y: torch.Tensor):
        h = self.encoder(x)                # (B,128,8,8)
        h = h.view(x.size(0), -1)          # (B,8192)
        h = torch.cat([h, self._one_hot(y)], dim=1)  # concat conditioning
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ────────────────────────────── Decoder ──────────────────────────────────
    def decode(self, z: torch.Tensor, y: torch.Tensor):
        z = torch.cat([z, self._one_hot(y)], dim=1)       # (B, latent+cls)
        h = self.fc_predecode(z)
        h = h.view(h.size(0), 128, 8, 8)
        return self.decoder(h)

    # ────────────────────────────── Forward ──────────────────────────────────
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y)
        return x_hat, mu, logvar
