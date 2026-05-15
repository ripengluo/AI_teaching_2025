#!/usr/bin/env python
# coding: utf-8

import os
import matplotlib.pyplot as plt
import torch
import torchdiffeq
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.models.unet import UNetModel

# ------------------------- hyper‑parameters ------------------------- #
savedir = "models/uncond_mnist"
os.makedirs(savedir, exist_ok=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 128
n_epochs = 20
x_dim = 64
sigma = 0.0
n_vis_steps = 5     # number of ODE timepoints shown in trajectory  (CHANGE 4)

# ------------------------- data ------------------------- #
trainset = datasets.MNIST(
    "data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.Resize((x_dim, x_dim)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    ),
)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)

# ------------------------- model & matcher ------------------------- #
model = UNetModel(                 # CHANGE 2
    dim=(1, x_dim, x_dim),
    num_channels=32,
    num_res_blocks=1,
    class_cond=False              # CHANGE 2
).to(device)

FM = ConditionalFlowMatcher(sigma=sigma)      # CHANGE 1
optimizer = torch.optim.Adam(model.parameters())
node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

# ------------------------- training ------------------------- #
retrain = False #True
if retrain:
    for epoch in range(n_epochs):
        for i, (x1, _) in enumerate(train_loader):   # labels ignored  (CHANGE 2)
            optimizer.zero_grad()
            x1 = x1.to(device)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = model(t, xt)                                 # CHANGE 2
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print(f"epoch: {epoch}, steps: {i}, loss: {loss.item():.4f}")
    torch.save(model.state_dict(), os.path.join(savedir, "uncond_cfm.pth"))
else:
    model.load_state_dict(torch.load(os.path.join(savedir, "uncond_cfm.pth"), map_location=device))

# ------------------------- validation & visualization ------------------------- #
with torch.no_grad():
    t_span = torch.linspace(0, 1, n_vis_steps, device=device)
    traj = torchdiffeq.odeint(
        lambda t, x: model.forward(t, x),
        torch.randn(100, 1, x_dim, x_dim, device=device),
        t_span,
        atol=1e-4,
        rtol=1e-4,
        method="dopri5",
    )  # [n_steps, B, C, H, W]

# ---- draw trajectory grid ---- #
fig, axs = plt.subplots(1, n_vis_steps, figsize=(2 * n_vis_steps, 2))
for i, ti in enumerate(t_span):
    grid = make_grid(
        traj[i][:50].clip(-1, 1),  # first 50 samples
        value_range=(-1, 1),
        padding=0,
        nrow=5,
    )
    axs[i].imshow(ToPILImage()(grid))
    axs[i].axis("off")
    axs[i].set_title(f"t={ti:.2f}")
plt.tight_layout()
plt.show()
print("Trajectory visualization completed.")  # CHANGE 4

