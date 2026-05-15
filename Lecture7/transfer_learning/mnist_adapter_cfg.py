# MNIST Conditional Diffusion with Adapter‑based Classifier‑Free Guidance
# ----------------------------------------------------------------------
# ‑ This file rewrites the user's original digit.py so that:
#   1. We can first train an unconditional **base** UNet (same as before).
#   2. We then freeze the base weights and **fine‑tune lightweight Adapter
#      modules + label embeddings** on digit labels (0‑9). Conditional‑
#      dropout (p_drop) teaches the model to output BOTH ε̂_∅ and ε̂_𝑦.
#   3. Inference runs **two forward passes** (base+adapter OFF / ON) and
#      mixes them with guidance_scale γ like classic CFG.
# ----------------------------------------------------------------------
# Author: ChatGPT (OpenAI‑o3) – 2025‑07‑20

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import random, math, os
import numpy as np
from typing import List, Optional
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm.utils import ModelEmaV3
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim

# --------------------------------------
#  Embeddings
# --------------------------------------

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
class SinusoidalTimeEmbedding(nn.Module):
    """Classic DDPM sinusoidal time embedding → (B, D)"""

    def __init__(self, time_steps: int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(time_steps, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (T, D)

    def forward(self, t: torch.Tensor):  # t shape (B,)
        return self.pe[t]  # (B, D)


class LabelEmbedding(nn.Module):
    """Learnable embedding for digit labels 0‑9 → (B, D)"""

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)

    def forward(self, y: torch.Tensor):  # y shape (B,)
        return self.embed(y)

# --------------------------------------
#  Adapter Module (LoRA‑style bottleneck)
# --------------------------------------

class Adapter(nn.Module):
    def __init__(self, channels: int, bottleneck: int = 16):
        super().__init__()
        self.down = nn.Conv2d(channels, bottleneck, 1)
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(bottleneck, channels, 1)
        # Zero‑init so base behaviour unchanged at start
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.relu(self.down(x)))

# --------------------------------------
#  Building Blocks
# --------------------------------------

class ResBlock(nn.Module):
    def __init__(self, C: int, num_groups: int, dropout_prob: float):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.GroupNorm(num_groups, C)
        self.norm2 = nn.GroupNorm(num_groups, C)
        self.conv1 = nn.Conv2d(C, C, 3, padding=1)
        self.conv2 = nn.Conv2d(C, C, 3, padding=1)
        self.dropout = nn.Dropout(dropout_prob, inplace=True)
        self.adapter = Adapter(C)  # ← NEW

    def forward(self, x, emb):  # emb shape (B, C,1,1)
        x = x + emb[:, : x.shape[1]]  # broadcast add time+label
        h = self.conv1(self.relu(self.norm1(x)))
        h = self.dropout(h)
        h = self.conv2(self.relu(self.norm2(h)))
        h = h + self.adapter(h)  # adapter residual
        return h + x


class Attention(nn.Module):
    def __init__(self, C: int, num_heads: int, dropout_prob: float):
        super().__init__()
        self.proj_qkv = nn.Linear(C, C * 3)
        self.proj_out = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout = dropout_prob

    def forward(self, x):  # x (B,C,H,W)
        H, W = x.shape[2:]
        x = rearrange(x, "b c h w -> b (h w) c")
        qkv = self.proj_qkv(x)
        qkv = rearrange(qkv, "b N (h d k) -> k b h N d", k=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        attn = rearrange(attn, "b h N d -> b N (h d)")
        out = self.proj_out(attn)
        out = rearrange(out, "b (h w) c -> b c h w", h=H, w=W)
        return out


class UnetLayer(nn.Module):
    def __init__(self, C: int, upscale: bool, attention: bool, num_groups: int, dropout_prob: float, num_heads: int):
        super().__init__()
        self.block1 = ResBlock(C, num_groups, dropout_prob)
        self.block2 = ResBlock(C, num_groups, dropout_prob)
        self.attn = Attention(C, num_heads, dropout_prob) if attention else None
        self.upscale = upscale
        if upscale:
            self.conv = nn.ConvTranspose2d(C, C // 2, 4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(C, C * 2, 3, stride=2, padding=1)

    def forward(self, x, emb):
        x = self.block1(x, emb)
        if self.attn is not None:
            x = self.attn(x)
        x = self.block2(x, emb)
        return self.conv(x), x

# --------------------------------------
#  UNet Backbone
# --------------------------------------

class UNET(nn.Module):
    def __init__(
        self,
        Channels: List[int] = [64, 128, 256, 512, 512, 384],
        Attentions: List[bool] = [False, True, False, False, False, True],
        Upscales: List[bool] = [False, False, False, True, True, True],
        num_groups: int = 32,
        dropout_prob: float = 0.1,
        num_heads: int = 8,
        input_channels: int = 1,
        output_channels: int = 1,
        time_steps: int = 1000,
        num_classes: int = 10,
    ):
        super().__init__()
        self.num_layers = len(Channels)
        self.shallow_conv = nn.Conv2d(input_channels, Channels[0], 3, padding=1)
        out_channels = (Channels[-1] // 2) + Channels[0]
        self.late_conv = nn.Conv2d(out_channels, out_channels // 2, 3, padding=1)
        self.output_conv = nn.Conv2d(out_channels // 2, output_channels, 1)
        self.relu = nn.ReLU(inplace=True)

        # Embedding modules
        embed_dim = max(Channels)
        self.time_emb = SinusoidalTimeEmbedding(time_steps, embed_dim)
        self.label_emb = LabelEmbedding(num_classes, embed_dim)
        self.embed_proj = nn.Linear(embed_dim, embed_dim)  # combine & project

        # UNet layers
        for i in range(self.num_layers):
            layer = UnetLayer(
                C=Channels[i],
                upscale=Upscales[i],
                attention=Attentions[i],
                num_groups=num_groups,
                dropout_prob=dropout_prob,
                num_heads=num_heads,
            )
            setattr(self, f"Layer{i+1}", layer)

    # --------------------------------------------------
    def _get_emb(self, t, y):
        """Return combined (time+label) embedding broadcast to (B,C,1,1)"""
        te = self.time_emb(t)  # (B,D)
        le = self.label_emb(y) if y is not None else 0.0  # (B,D) or 0
        emb = self.embed_proj(te + le)  # (B,D)
        return emb.unsqueeze(-1).unsqueeze(-1)  # (B,D,1,1)

    # --------------------------------------------------
    def forward(self, x, t, y=None):
        emb = self._get_emb(t, y)  # broadcast once
        x = self.shallow_conv(x)
        residuals = []
        # Down
        for i in range(self.num_layers // 2):
            layer = getattr(self, f"Layer{i+1}")
            x, r = layer(x, emb)
            residuals.append(r)
        # Up
        for i in range(self.num_layers // 2, self.num_layers):
            layer = getattr(self, f"Layer{i+1}")
            x = torch.cat((layer(x, emb)[0], residuals[self.num_layers - i - 1]), dim=1)
        return self.output_conv(self.relu(self.late_conv(x)))

# --------------------------------------
#  Scheduler (unchanged)
# --------------------------------------

class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int = 1000):
        super().__init__()
        beta = torch.linspace(1e-4, 0.02, num_time_steps).to(device)
        alpha = 1 - beta
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", torch.cumprod(alpha, dim=0))

    def forward(self, t):
        return self.beta[t], self.alpha[t]

# --------------------------------------
#  Utils
# --------------------------------------

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# --------------------------------------
#  Stage‑1: Train Base UNet (unconditional)
# --------------------------------------

def train_base(
    batch_size: int = 128,
    num_time_steps: int = 1000,
    num_epochs: int = 75,
    lr: float = 1e-4,
    ema_decay: float = 0.9999,
    ckpt_dir: str = "checkpoints",
):
    os.makedirs(ckpt_dir, exist_ok=True)
    set_seed()

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True, num_workers=4)

    scheduler = DDPM_Scheduler(num_time_steps)
    model = UNET().cuda()
    opt = optim.AdamW(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    mse = nn.MSELoss()

    for epoch in range(num_epochs):
        total = 0
        for x, _ in tqdm(loader, desc=f"Base Epoch {epoch+1}/{num_epochs}"):
            x = F.pad(x.cuda(), (2, 2, 2, 2))  # (B,1,32,32)
            b = x.size(0)
            t = torch.randint(0, num_time_steps, (b,), device=x.device)
            noise = torch.randn_like(x)
            alpha = scheduler.alpha[t].view(b, 1, 1, 1)
            xt = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise

            pred = model(xt, t)
            loss = mse(pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()
            ema.update(model)
            total += loss.item()
        print(f"Epoch {epoch+1} | loss = {total/len(loader):.4f}")

    torch.save(
        {
            "base": ema.module.state_dict(),
            "cfg": model.state_dict(),  # raw (non‑EMA) for resume
        },
        os.path.join(ckpt_dir, "mnist_base.pth"),
    )

# --------------------------------------
#  Stage‑2: Adapter Fine‑tune with Conditional Dropout
# --------------------------------------

def freeze_base(model: UNET):
    for n, p in model.named_parameters():
        if not ("adapter" in n or "label_emb" in n or "embed_proj" in n):
            p.requires_grad_(False)


def train_adapter(
    base_ckpt: str = "checkpoints/mnist_base.pth",
    batch_size: int = 128,
    num_time_steps: int = 1000,
    num_epochs: int = 75,
    lr: float = 5e-4,
    p_drop: float = 0.1,
    guidance_scale: float = 5.0,  # not used in training, but stored for logging
    ema_decay: float = 0.999,
    ckpt_dir: str = "checkpoints",):

    os.makedirs(ckpt_dir, exist_ok=True)
    set_seed()

    # Load base
    model = UNET().cuda()
    model.load_state_dict(torch.load(base_ckpt)["base"], strict=False)

    # Freeze base, enable adapters
    freeze_base(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params (adapter+label) = {sum(p.numel() for p in trainable)/1e6:.2f} M")

    opt = optim.AdamW(trainable, lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    mse = nn.MSELoss()

    dataset = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True, num_workers=4)

    scheduler = DDPM_Scheduler(num_time_steps)

    for epoch in range(num_epochs):
        total = 0
        for x, y in tqdm(loader, desc=f"Adapter Epoch {epoch+1}/{num_epochs}"):
            x = F.pad(x.cuda(), (2, 2, 2, 2))
            y = y.cuda()
            b = x.size(0)
            t = torch.randint(0, num_time_steps, (b,), device=x.device)
            noise = torch.randn_like(x)
            alpha = scheduler.alpha[t].view(b, 1, 1, 1)
            xt = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise

            # --- conditional dropout ---
            dropout_mask = (torch.rand(b, device=x.device) < p_drop)
            y_in = y.clone()
            y_in[dropout_mask] = 0  # digit "0" acts as null class for simplicity
            # forward
            pred = model(xt, t, y_in)
            loss = mse(pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()
            ema.update(model)
            total += loss.item()
        print(f"Epoch {epoch+1} | loss = {total/len(loader):.4f}")

    torch.save(
        {
            "base": torch.load(base_ckpt)["base"],
            "adapter": ema.module.state_dict(),
            "guidance_scale": guidance_scale,
        },
        os.path.join(ckpt_dir, "mnist_adapter_cfg.pth"),
    )

# --------------------------------------
#  Inference with CFG
# --------------------------------------

def sample_cfg(
    cfg_ckpt: str = "checkpoints/mnist_adapter_cfg.pth",
    num_time_steps: int = 1000,
    guidance_scale: Optional[float] = None,
    n_samples: int = 20,
):
    ckpt = torch.load(cfg_ckpt)
    base_state = ckpt["base"]
    adapter_state = ckpt["adapter"]
    guidance_scale = guidance_scale or ckpt.get("guidance_scale", 5.0)

    model_base = UNET().cuda()
    model_base.load_state_dict(base_state, strict=False)
    model_base.eval()

    model_cond = UNET().cuda()
    model_cond.load_state_dict(adapter_state, strict=False)
    model_cond.eval()

    scheduler = DDPM_Scheduler(num_time_steps)

    digits = torch.arange(10, device="cuda")
    digits = digits.repeat((n_samples + 9) // 10)[:n_samples]
 
    with torch.no_grad():
        imgs = []
        for y in digits:
            z = torch.randn(1, 1, 32, 32, device="cuda")
            for t in reversed(range(num_time_steps)):
                t_batch = torch.tensor([t], device="cuda")
                beta, alpha_cum = scheduler(beta_index := t_batch)
                alpha = torch.sqrt(alpha_cum)
                sigma = torch.sqrt(1 - alpha_cum)

                eps_uncond = model_base(z, t_batch, None)
                eps_cond = model_cond(z, t_batch, y.view(1))
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond) # mixing the predicted noise from both base and conditional models

                z = (1 / torch.sqrt(1 - beta)) * (z - beta / sigma * eps)
                if t > 0:
                    z += torch.randn_like(z) * torch.sqrt(beta)
            imgs.append(z.cpu())

    grid_cols = 10
    grid_rows = n_samples//grid_cols
    assert n_samples == grid_rows * grid_cols
    grid = torch.cat(imgs, dim=0)  # (N,1,32,32)
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols, grid_rows))
    for i, ax in enumerate(axes.flat):
        x = imgs[i].squeeze(0)
        x = rearrange(x, 'c h w -> h w c')
        x = x.numpy()
        ax.imshow(x, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.show()


# --------------------------------------
#  Quick CLI Helpers
# --------------------------------------

if __name__ == "__main__":

    #train_base()
    #train_adapter() 
    sample_cfg(n_samples=30)

