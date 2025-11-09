import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid
import seaborn as sns
import sys
import math
from utils import *
from models import CVAE


x_dim = 64
# create a transofrm to apply to each datapoint
transform = transforms.Compose([transforms.Resize((x_dim, x_dim)),  transforms.ToTensor()])

# download the MNIST datasets
path = './datasets'
train_dataset = FashionMNIST(path, transform=transform, download=True)
test_dataset  = FashionMNIST(path, transform=transform, download=True)

# create train and test dataloaders
batch_size = 256 
latent_dim = 2 # 40
subset = Subset(train_dataset, range(1000))
train_loader = DataLoader(dataset=subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=Subset(test_dataset, np.arange(11480, 11490)), batch_size=batch_size, shuffle=False)

#print("The training set has size of", len(train_loader))
#print("The test set has the size of", len(test_loader))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)
print("###########################################Y. M. C. A.##################################################")

# get 9 sample training images for visualization
dataiter = iter(train_loader)
image = dataiter.__next__()

num_samples = 9
sample_images = [image[0][i,0] for i in range(num_samples)]
fig = plt.figure(figsize=(3, 3))
grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=0.1)

for ax, im in zip(grid, sample_images):
    ax.imshow(im, cmap='gray')
    ax.axis('off')
plt.show()


lr = 1e-3
cvae = CVAE(latent_dim, device=device).to(device)
optimizer1 = Adam(cvae.parameters(), lr=lr)


EPOCH = 5000
log = np.zeros([EPOCH, 2])


retrain = False #True 
#cvae.load_state_dict(torch.load("cvae.pth", weights_only=True))
if retrain:
    train(cvae, optimizer1, train_loader, log, epochs=EPOCH, batch_size = batch_size, device=device, x_flat_dim=x_dim**2)
    torch.save(cvae.state_dict(), "cvae.pth")
else:
    cvae.load_state_dict(torch.load("cvae.pth", weights_only=True))
    
test_x, test_y = iter(test_loader).__next__()
test_x = test_x.to(device)
test_y = test_y.to(device)

imag_dim = 10 
def reconstruct_digit(test_x, test_y):
    x_hat, _, _ = cvae(test_x, test_y)
    x_hat = x_hat.detach().cpu().reshape(-1, x_dim, x_dim)[:imag_dim] # reshape vector to 2d array

    test_x = test_x.detach().cpu().reshape(-1, x_dim, x_dim)[:imag_dim]
    fig = plt.figure(figsize=(2, imag_dim))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, imag_dim), axes_pad=0.1)
    labels = ["Original", "VAE"]
    for ax, im in zip(grid, torch.cat((test_x, x_hat), dim=0)):
        ax.imshow(im, cmap='gray')
        ax.axis('off')
    plt.show()

#for t in range(EPOCH):
#    print("epoch: %d  "%t, "  ".join(["%.4f"%w for w in log[t]]))
reconstruct_digit(test_x, test_y)

def plot_latent_space(model, label=2, scale=1.0, n=15, digit_size=64, figsize=15):
    # display a n*n 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n))

    # construct a grid
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    #digits = []
    label = torch.tensor([label], dtype=torch.float).to(device)
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
            x_decoded = model.decode(z_sample, label)
            digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size,] = digit
    plt.figure(figsize=(figsize, figsize))
    plt.title('VAE Latent Space Visualization')
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("sampled z [0]")
    plt.ylabel("sampled z [1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


plot_latent_space(cvae, label=4)



def plot_label_grid(model,
                    latent_dim: int = 2,
                    num_samples: int = 10,
                    device: torch.device | str = "cpu"):
    model.eval()
    with torch.no_grad():
        fig, axes = plt.subplots(num_samples, 10, figsize=(10, num_samples))

        for lbl in range(10):
            y = torch.full((num_samples,), lbl, dtype=torch.long, device=device)
            z = torch.empty(num_samples, latent_dim, device=device).uniform_(-1.0, 1.0)
            x_hat = model.decode(z, y).cpu()            # (B, 1, 64, 64)

            for row in range(num_samples):
                ax = axes[row, lbl]
                ax.imshow(x_hat[row, 0], cmap="gray")
                ax.axis("off")

            axes[0, lbl].set_title(str(lbl), fontsize=8)

        plt.tight_layout()
        plt.show()

plot_label_grid(cvae, latent_dim=2, device=device)

