import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import seaborn as sns
import sys
import math


def kl_divergence(mu_q, log_var_q, mu_p=None, log_var_p=None):
    if mu_p is None:
        kl = -0.5 * torch.sum(1 + log_var_q - mu_q.pow(2) - log_var_q.exp())
    else:
        var_p = torch.exp(log_var_p)
        var_q = torch.exp(log_var_q)
        kl = 0.5 * (log_var_p - log_var_q + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1)
    return kl.mean()

def get_average(train_loader, device):
    x_ = []
    for batch_idx, (x, _) in enumerate(train_loader):
        x_.append(x)
    x_ = torch.cat(x_); x_ = torch.mean(x_, axis=0); x_ = x_.unsqueeze(0);
    x_ = x_.to(device)
    return x_

def loss_function(x, x_hat, mean, logvar):
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ logvar - mean.pow(2) - logvar.exp())
    return reproduction_loss + KLD

def loss_mivae(x, x_hat, mean_post, logvar_post, mean_prior, logvar_prior, order, epoch):
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    #f = 1/(1+np.exp(-1*(epoch-50)))
    KL = kl_divergence(mean_post[0], logvar_post[0])
    for i in range(1, order):
        KL += kl_divergence(mean_post[i], logvar_post[i], mean_prior[i-1], logvar_prior[i-1])
        KL += kl_divergence(mean_prior[i-1], logvar_prior[i-1])
    return reproduction_loss + KL/order


def train(model, optimizer, train_loader, log, epochs, batch_size, device, x_flat_dim):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()
            batch_idx += 1
        if epoch == 0 or loss_min > overall_loss:
            loss_min = overall_loss
            if epoch > int(0.5 * epochs):
                torch.save(model.state_dict(), "vae.pth")
                print("model saved!")

        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch_idx*batch_size))
        log[epoch, 0] = overall_loss/(batch_idx*batch_size)
    return overall_loss

def loss_mg(x, x_hat, mean, logvar, mixture_means, mixture_logvars):
    reconstruction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    
    mean = mean.unsqueeze(1)  # (batch, 1, latent_dim)
    logvar = logvar.unsqueeze(1)

    # Compute log probabilities for each mixture component
    log_probs = -0.5 * (
        mixture_logvars + torch.exp(logvar - mixture_logvars) + (mean - mixture_means).pow(2) / torch.exp(mixture_logvars)
    ).sum(dim=2)  # (batch, n_components)

    log_prior = torch.logsumexp(log_probs - np.log(mixture_means.size(0)), dim=1).sum()

    log_posterior = -0.5 * (1 + logvar).sum()

    kl_divergence = log_posterior - log_prior

    return reconstruction_loss + kl_divergence

def train_mgvae(model, optimizer, train_loader, epochs, device):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for x, _ in train_loader:
            x = x.to(device)

            optimizer.zero_grad()

            x_hat, mean, logvar, mixture_means, mixture_logvars = model(x)
            loss = loss_mg(x, x_hat, mean, logvar, mixture_means, mixture_logvars)

            loss.backward()
            optimizer.step()

            overall_loss += loss.item()
        if epoch == 0 or loss_min > overall_loss:
            loss_min = overall_loss
            if epoch > int(0.5 * epochs):
                torch.save(model.state_dict(), "MG.pth")
                print("model saved!")

        average_loss = overall_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}, Average Loss: {average_loss:.4f}")

"""
def train_mgvae(model, optimizer, train_loader, log, epochs, batch_size, device, x_flat_dim):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()

            x_hat, mean, logvar = model(x)
            loss, recon, kl = model.elbo_loss(x, x_hat, mean, logvar)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()
            batch_idx += 1
        if epoch == 0 or loss_min > overall_loss:
            loss_min = overall_loss
            if epoch > int(0.5 * epochs):
                torch.save(model.state_dict(), "MG.pth")
                print("model saved!")

        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch_idx*batch_size))
        log[epoch, 1] = overall_loss/(batch_idx*batch_size)

    return overall_loss

"""
