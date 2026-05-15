#!/bin/python

import torch
from torch import optim
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
from models import VAE
from utils import *
import sys

start_time = time.time() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on %s"%device)
latent_dim = 40 
x_dim = 64
batch_size = 200

transform = transforms.Compose([
    transforms.Resize((64, 64)),    
    transforms.ToTensor(),          
])

dataset = datasets.CelebA(root='./data', split='train', download=True, transform=transform)

#list = [1,10,0,45,100,102,103,104,108,109]
subset = Subset(dataset, np.arange(5000))

train_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(Subset(dataset, np.arange(1410,1420)), batch_size=10, shuffle=False)
#test_loader = DataLoader(Subset(dataset, np.arange(410,420)), batch_size=10, shuffle=False)
#test_loader = DataLoader(Subset(dataset, np.arange(22410,22420)), batch_size=10, shuffle=False)

#test_loader = DataLoader(Subset(dataset, np.arange(7801,7811)), batch_size=10, shuffle=False)

lr = 1e-3
EPOCH =  5000

vae = VAE(latent_dim=latent_dim).to(device)


optimizer1 = Adam(vae.parameters(), lr=lr)
scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=int(EPOCH/4), gamma=0.2)


log = np.zeros([EPOCH, 2])

retrain = False #True 
if retrain:
    train(vae, optimizer1, scheduler1, train_loader, log, EPOCH, batch_size, device, x_dim**2)
else:
    vae.load_state_dict(torch.load("vae.pth", weights_only=True))


vae.eval()



with torch.no_grad():
    test_x, test_y = next(iter(test_loader))
    test_x = test_x.to(device)
    x_vae = vae(test_x)[0]
    x_vae = x_vae.cpu().permute(0, 2, 3, 1).detach().numpy()
    
    test_x = test_x.cpu().permute(0, 2, 3, 1).detach().numpy()


#zs = np.random.multivariate_normal(np.zeros(latent_dim), np.diag(np.ones(latent_dim)), 10)
#zs = torch.tensor(zs, dtype=torch.float).to(device)
#recon_images_vae = vae.decode(zs)
#recon_images_mivae = mivae.decode(zs)[0]Labels = ["VAE", "MIVAE3", "MIVAE5", "MIVAE10", "axes[6, i].set_title("MI-VAE (k=40)", fontsize=8)"]
Labels = ["VAE", "Ground_Truth"]
fig, axes = plt.subplots(len(Labels), 11, figsize=(16, 18))
plt.subplots_adjust(wspace=0.02, hspace=0.02)
for i in range(10):
    axes[0, i].imshow(x_vae[i])
    axes[0, i].axis('off')
    axes[-1, i].imshow(test_x[i])
    axes[-1, i].axis('off')
for j in range(len(Labels)):
    axes[j, -1].text(0.5, 0.5, Labels[j], ha='center', va='center', fontsize=10, bbox=None)
plt.tight_layout()
plt.show()


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")

def plot_face_interpolation(vae, test_loader, device, n_interpolations=5):
    """
    Generate and display latent space interpolations between two face images.

    Args:
        vae: Trained VAE model
        test_loader: Data loader (including the test set)
        device: Device (cuda or cpu)
        n_interpolations: Number of intermediate interpolations (default is 5)
    """
    vae.eval()

    # Randomly select two images from the test set
    test_images, _ = next(iter(test_loader))
    test_images, _ = next(iter(test_loader))
    print(test_images.shape)
    print(test_images[6].shape, test_images[5:6].shape)
    img1, img2 = test_images[5:6].to(device), test_images[8:9].to(device)

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

plot_face_interpolation(vae, train_loader, device, n_interpolations=3)

