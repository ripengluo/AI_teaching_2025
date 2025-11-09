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
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x, y)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()
            batch_idx += 1
        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch_idx*batch_size))
        log[epoch, 0] = overall_loss/(batch_idx*batch_size)
    return overall_loss

