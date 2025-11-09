#!/bin/python

import torch
from torch import optim
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset, ConcatDataset
import matplotlib.pyplot as plt
from PIL import Image
import time
import math
from models import VAE
from utils import *
import sys

start_time = time.time() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on %s"%device)
latent_dim = 40 
x_dim = 640
batch_size = 10


lr = 1e-3
EPOCH =  5000

vae = VAE(latent_dim=latent_dim).to(device)


optimizer1 = Adam(vae.parameters(), lr=lr)
scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=int(EPOCH/4), gamma=0.2)


log = np.zeros([EPOCH, 2])


transform = transforms.Compose([
    transforms.Resize((x_dim, x_dim)), 
    transforms.ToTensor(),              # → [0, 1]
])

def load_png_as_celeba_tensor(png_path, device="cpu"):
    img = Image.open(png_path).convert("RGB")
    tensor = transform(img).to(device) 
    return tensor.unsqueeze(0)


image1 = load_png_as_celeba_tensor("Ripeng.png", device="cuda")
image2 = load_png_as_celeba_tensor("Daniel.png", device="cuda")


def show_two(img1, img2, titles=('image1', 'image2')):
    imgs = torch.cat([img1, img2], dim=0)   # (2,3,64,64)
    imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()         # → (B,64,64,3)

    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    for ax, im, title in zip(axes, imgs, titles):
        ax.imshow(im)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

show_two(image1, image2)


train_ds2 = TensorDataset(torch.cat([image1, image2], dim=0))
train_loader2 = DataLoader(train_ds2,
                           batch_size=2,
                           shuffle=True,
                           drop_last=False)


class TensorOnly(Dataset):
    def __init__(self, images):        
        self.images = images
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx]         

dataset = datasets.CelebA(root='./data', split='train', download=True, transform=transform)
dataset = TensorOnly(dataset)
subset = Subset(dataset, np.arange(198))
full_ds = ConcatDataset([subset, train_ds2])
train_loader = DataLoader(full_ds, batch_size=2, shuffle=False)




train_fine(vae, optimizer1, scheduler1, train_loader, log, 200, 2, device, x_dim**2)

def plot_face_interpolation2(vae, img1, img2, device, n_interpolations=5):
    """
    Generate and display latent space interpolations between two face images.

    Args:
        vae: Trained VAE model
        test_loader: Data loader (including the test set)
        device: Device (cuda or cpu)
        n_interpolations: Number of intermediate interpolations (default is 5)
    """
    vae.eval()
    print(img1.shape, img2.shape)

    # Encode into the latent space
    with torch.no_grad():
        mu1, logvar1 = vae.encode(img1)
        mu2, logvar2 = vae.encode(img2)
        z1 = vae.reparameterization(mu1, logvar1)
        z2 = vae.reparameterization(mu2, logvar2)

        # Create interpolation points
        alphas = np.linspace(0, 1, n_interpolations + 2)  # including endpoints
        interpolations = []
        for alpha in alphas:
            z = alpha * z1 + (1 - alpha) * z2
            recon = vae.decode(z).cpu().squeeze(0).permute(1, 2, 0).numpy()
            interpolations.append(recon)

    # Visualize the interpolations
    fig, axes = plt.subplots(1, n_interpolations + 2, figsize=(15, 3))
    titles = ["Image 1"] + [f"Mix {i+1}" for i in range(n_interpolations)] + ["Image 2"]

    for i, (img, title) in enumerate(zip(interpolations, titles)):
        axes[i].imshow(np.clip(img, 0, 1))
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


plot_face_interpolation2(vae, image1, image2, device, n_interpolations=3)


