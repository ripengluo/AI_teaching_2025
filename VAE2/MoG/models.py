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

class MixtureGaussianVAE(nn.Module):
    def __init__(self, latent_dim=10, n_components=5, device="cuda"):
        super(MixtureGaussianVAE, self).__init__()
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.device = device

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # output (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # output (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # output (128, 8, 8)
            nn.ReLU(),
        )

        # Parameters for posterior q(z|x)
        self.mean_layer = nn.Linear(128 * 8 * 8, latent_dim)
        self.logvar_layer = nn.Linear(128 * 8 * 8, latent_dim)

        # Mixture of Gaussians prior parameters (learnable)
        self.mixture_means = nn.Parameter(torch.randn(n_components, latent_dim))
        self.mixture_logvars = nn.Parameter(torch.randn(n_components, latent_dim))

        # Decoder
        self.predecode = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x_enc = self.encoder(x)
        x_enc = x_enc.view(x.size(0), -1)
        mean, logvar = self.mean_layer(x_enc), self.logvar_layer(x_enc)
        return mean, logvar

    def reparameterization(self, mean, logvar):
        epsilon = torch.randn_like(logvar).to(self.device)
        return mean + torch.exp(0.5 * logvar) * epsilon

    def decode(self, z):
        z_dec = self.predecode(z)
        z_dec = z_dec.view(z_dec.size(0), 128, 8, 8)
        return self.decoder(z_dec)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar, self.mixture_means, self.mixture_logvars

"""

def _log_normal_diag(x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor):
    return -0.5 * (
        logvar
        + (x - mean) ** 2 / torch.exp(logvar)
        + math.log(2 * math.pi)
    ).sum(dim=-1)

class MixtureGaussianVAE(nn.Module):

    def __init__(
        self,
        latent_dim: int = 10,
        n_components: int = 10,
        device: str = "cuda",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.device = device

        # ---------- Encoder / Decoder share architecture with VAE ----------
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self._enc_out_dim = 128 * 8 * 8
        self.mean_layer = nn.Linear(self._enc_out_dim, latent_dim)
        self.logvar_layer = nn.Linear(self._enc_out_dim, latent_dim)

        self.predecode = nn.Linear(latent_dim, self._enc_out_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

        # ---------------- Mixture‑prior parameters ----------------
        self.component_logits = nn.Parameter(torch.zeros(n_components))  # unnorm log‑pi
        self.mixture_means = nn.Parameter(
            torch.randn(n_components, latent_dim) * 0.05
        )
        self.mixture_logvars = nn.Parameter(
            torch.zeros(n_components, latent_dim)
        )  # initialise unit variance

    # -------- Standard VAE ops --------
    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        h = h.view(x.size(0), -1)
        return self.mean_layer(h), self.logvar_layer(h)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor):
        eps = torch.randn_like(logvar)
        return mean + torch.exp(0.5 * logvar) * eps

    def decode(self, z: torch.Tensor):
        h = self.predecode(z)
        h = h.view(z.size(0), 128, 8, 8)
        return self.decoder(h)

    # ---------- KL(q(z|x) || p(z)) via 1‑sample Monte‑Carlo ----------
    def _kl_mog(self, mean: torch.Tensor, logvar: torch.Tensor):
        z = self.reparameterize(mean, logvar)  # (B, D)
        log_q_zx = _log_normal_diag(z, mean, logvar)  # (B,)

        # Mixture log‑prob: log p(z) = logsumexp_k [ log pi_k + log N(z|mu_k, var_k) ]
        log_pi = F.log_softmax(self.component_logits, dim=0)  # (K,)
        z_exp = z.unsqueeze(1)  # (B, 1, D)
        log_p_z_c = _log_normal_diag(
            z_exp, self.mixture_means.unsqueeze(0), self.mixture_logvars.unsqueeze(0)
        )  # (B, K)
        log_p_z = torch.logsumexp(log_pi + log_p_z_c, dim=1)  # (B,)

        kl = (log_q_zx - log_p_z).mean()
        return kl

    # ---------- Forward ----------
    def forward(self, x: torch.Tensor):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

    def elbo_loss(
        self, x: torch.Tensor, x_hat: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor
    ):
        recon = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
        kl = self._kl_mog(mean, logvar)
        return recon + kl, recon, kl

"""
