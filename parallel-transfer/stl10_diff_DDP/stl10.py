# https://towardsdatascience.com/diffusion-model-from-scratch-in-pytorch-ddpm-9d9760528946/
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange #pip install einops
from typing import List
import random
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
from timm.utils import ModelEmaV3 #pip install timm 
from tqdm import tqdm #pip install tqdm
import matplotlib.pyplot as plt #pip install matplotlib
import torch.optim as optim
from typing import List
import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
world_size = torch.cuda.device_count()


# Define the UNet
class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps:int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        emb = torch.zeros(time_steps, embed_dim, requires_grad=False)
        emb[:, 0::2] = torch.sin(position * div)
        emb[:, 1::2] = torch.cos(position * div)
        self.register_buffer('embeddings', emb)

    def forward(self, x, t):
        embeds = self.embeddings[t]
        return embeds[:, :, None, None]

# Residual Blocks
class ResBlock(nn.Module):
    def __init__(self, C: int, num_groups: int, dropout_prob: float):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(self, x, embeddings):
        x = x + embeddings[:, :x.shape[1], :, :]
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return r + x

class Attention(nn.Module):
    def __init__(self, C: int, num_heads:int , dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(C, C*3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        h, w = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        q,k,v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(q,k,v, is_causal=False, dropout_p=self.dropout_prob)
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, 'b h w C -> b C h w')

class UnetLayer(nn.Module):
    def __init__(self, 
            upscale: bool, 
            attention: bool, 
            num_groups: int, 
            dropout_prob: float,
            num_heads: int,
            C: int):
        super().__init__()
        self.ResBlock1 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock3 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock4 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        if upscale:
            self.conv = nn.ConvTranspose2d(C, C//2, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(C, C*2, kernel_size=3, stride=2, padding=1)
        if attention:
            self.attention_layer = Attention(C, num_heads=num_heads, dropout_prob=dropout_prob)

    def forward(self, x, embeddings):
        x = self.ResBlock1(x, embeddings)
        x = self.ResBlock2(x, embeddings)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        x = self.ResBlock3(x, embeddings)
        x = self.ResBlock4(x, embeddings)

        return self.conv(x), x


class UNET(nn.Module):
    def __init__(self,
            Channels: List = [64, 128, 256, 512, 512, 384],
            Attentions: List = [False, True, False, False, False, True],
            Upscales: List = [False, False, False, True, True, True],
            num_groups: int = 32,
            dropout_prob: float = 0.1,
            num_heads: int = 8,
            input_channels: int = 1,
            output_channels: int = 1,
            time_steps: int = 1000):
        super().__init__()
        self.num_layers = len(Channels)
        self.shallow_conv = nn.Conv2d(input_channels, Channels[0], kernel_size=3, padding=1)
        out_channels = (Channels[-1]//2)+Channels[0]
        self.late_conv = nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels//2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.embeddings = SinusoidalEmbeddings(time_steps=time_steps, embed_dim=max(Channels))
        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=Upscales[i],
                attention=Attentions[i],
                num_groups=num_groups,
                dropout_prob=dropout_prob,
                C=Channels[i],
                num_heads=num_heads
            )
            setattr(self, f'Layer{i+1}', layer)

    def forward(self, x, t):
        x = self.shallow_conv(x)
        residuals = []
        for i in range(self.num_layers//2):
            layer = getattr(self, f'Layer{i+1}')
            embeddings = self.embeddings(x, t)
            x, r = layer(x, embeddings)
            residuals.append(r)
        for i in range(self.num_layers//2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x = torch.concat((layer(x, embeddings)[0], residuals[self.num_layers-i-1]), dim=1)
        return self.output_conv(self.relu(self.late_conv(x)))


class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int=1000, rank="cuda"):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False).to(rank)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)

    def forward(self, t):
        return self.beta[t], self.alpha[t]


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def display_reverse(images: List):
    fig, axes = plt.subplots(1, 10, figsize=(10,1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = rearrange(x, 'c h w -> h w c')
        x = x.numpy()
        ax.imshow(np.clip(x, 0, 1))
        ax.axis('off')
    plt.show()

def inference(checkpoint_path: str=None,
              num_time_steps: int=2000,
              ema_decay: float=0.9999, ):
    checkpoint = torch.load(checkpoint_path)
    model = UNET(input_channels=3, output_channels=3, time_steps=num_time_steps).cuda()
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)

    from collections import OrderedDict
    ema_state = checkpoint["ema"]
    fixed_state = OrderedDict()
    for k, v in ema_state.items():
        new_k = k.replace("module.module.", "module.")
        fixed_state[new_k] = v
    ema.load_state_dict(fixed_state, strict=True)

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    times = [0,15,50,100,200,300,400,550,1200,1999]
    images = []

    with torch.no_grad():
        model = ema.module.eval()
        for i in range(10):
            z = torch.randn(1, 3, 32, 32)
            for t in reversed(range(1, num_time_steps)):
                t = [t]
                scheduler.alpha = scheduler.alpha.cpu(); scheduler.beta = scheduler.beta.cpu()
                temp = (scheduler.beta[t]/( (torch.sqrt(1-scheduler.alpha[t]))*(torch.sqrt(1-scheduler.beta[t]))))
                z = (1/(torch.sqrt(1-scheduler.beta[t])))*z - (temp*model(z.cuda(),t).cpu())
                if t[0] in times:
                    images.append(z)
                e = torch.randn(1, 3, 32, 32)
                z = z + (e*torch.sqrt(scheduler.beta[t]))
            scheduler.alpha = scheduler.alpha.cpu(); scheduler.beta = scheduler.beta.cpu()
            temp = scheduler.beta[0]/( (torch.sqrt(1-scheduler.alpha[0]))*(torch.sqrt(1-scheduler.beta[0])) )
            x = (1/(torch.sqrt(1-scheduler.beta[0])))*z - (temp*model(z.cuda(),[0]).cpu())

            images.append(x)
            x = rearrange(x.squeeze(0), 'c h w -> h w c').detach()
            x = x.numpy()
            plt.imshow(np.clip(x, 0, 1))
            plt.show()
            display_reverse(images)
            images = []

# ---------------- 新增：DDP 必要 import ----------------
import os, torch.distributed as dist, torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
# ------------------------------------------------------

def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"      # 同 torchrun 里的 master_port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

# ------------- train()：仅保留「一张卡」逻辑 ----------------
def train_ddp(rank, world_size,
              batch_size=50, num_time_steps=2000, num_epochs=200,
              ema_decay=0.9999, lr=2e-5, seed=42,
              ckpt_path=None):

    setup_ddp(rank, world_size)
    set_seed(seed + rank)          # 保证每个进程不同 seed

    # ---------------- 数据 -----------------
    dataset = datasets.STL10(
        root="./data", split="train+unlabeled", download=True,
        transform=transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    )
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader  = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                         num_workers=4, pin_memory=True, drop_last=True)

    # ---------------- 模型 -----------------
    model = UNET(input_channels=3, output_channels=3, time_steps=num_time_steps).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    scheduler = DDPM_Scheduler(num_time_steps, rank)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ema       = ModelEmaV3(model, decay=ema_decay, device=torch.device(rank))

    if ckpt_path is not None and rank == 0:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["weights"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        dist.barrier()   # 让其他 rank 等 rank‑0 加载完

    criterion = nn.MSELoss(reduction="mean")

    # ---------------- 训练循环 ----------------
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)   # ⚠️ DDP 需要
        model.train()
        total_loss = 0.0

        for bidx, (x,_) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            x = x.to(rank, non_blocking=True)

            # 每个进程本身就是 per‑GPU mini‑batch！
            bsz = x.size(0)
            t   = torch.randint(0, num_time_steps, (bsz,), device=rank)
            e   = torch.randn_like(x, device=rank, requires_grad=False)
            a   = scheduler.alpha[t].view(bsz,1,1,1).to(rank)

            noisy_x = torch.sqrt(a)*x + torch.sqrt(1-a)*e

            optimizer.zero_grad(set_to_none=True)
            out = model(noisy_x, t)
            loss = criterion(out, e)
            loss.backward()
            optimizer.step()
            ema.update(model)
            total_loss += loss.item()

        # 只让 rank‑0 打印 / 存模型
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss {(total_loss/len(loader)):.5f}")

    if rank == 0:
        torch.save(
            {"weights": model.module.state_dict(),   # DDP 下要 .module
             "optimizer": optimizer.state_dict(),
             "ema": ema.state_dict()},
            "checkpoints/ddpm_ddp.pth"
        )

    cleanup_ddp()


# ---------------- main：spawn 进程 ----------------
def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
    inference('checkpoints/ddpm_ddp.pth')

if __name__ == "__main__":
    main()


