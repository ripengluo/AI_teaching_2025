#!/usr/bin/env python
# coding: utf-8
# cif10_ddp.py —— CIFAR-10 SF2M 并行训练脚本（按 stl10.py 风格）

import os, math, copy, random
from copy import deepcopy
import torch
import torchdiffeq, torchsde
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from torchcfm.conditional_flow_matching import SchrodingerBridgeConditionalFlowMatcher
from torchcfm.models.unet.unet import UNetModelWrapper

# ---------- 通用工具 ----------
def set_seed(seed: int = 42):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); random.seed(seed)

def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

# ---------- 训练 ----------
def train_ddp(rank, world_size,
              x_dim=32, batch_size=128, total_steps=400_001,
              sigma=0.1, ema_decay=0.9999, lr=2e-4, seed=42,
              save_dir="models/cifar10"):

    setup_ddp(rank, world_size)
    set_seed(seed + rank)
    device = torch.device(f"cuda:{rank}")

    # 数据集 & DistributedSampler
    trainset = datasets.CIFAR10(
        root="data", train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ]),
    )
    sampler = torch.utils.data.DistributedSampler(
        trainset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True)

    n_epochs = math.ceil(total_steps / len(loader))

    # 模型（主流 + score）—— 按原脚本参数
    model = UNetModelWrapper(
        dim=(3, x_dim, x_dim), num_res_blocks=2,
        num_channels=128, channel_mult=[1, 2, 2, 2],
        num_heads=4, num_head_channels=64,
        attention_resolutions="16", dropout=0.1,
    ).to(device)

    score_model = copy.deepcopy(model).to(device)
    ema_model   = copy.deepcopy(model).to(device)      # 只做 EMA，不用 DDP

    # DDP 封装
    model       = DDP(model,       device_ids=[rank], output_device=rank, find_unused_parameters=False)
    score_model = DDP(score_model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(score_model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(step, 5_000) / 5_000)

    FM = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)

    global_step = 0
    pbar = tqdm(total=total_steps, unit="step")

    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)
        for x1, _ in loader:

            global_step += 1
            pbar.update(1)
            pbar.set_description(f"{global_step}/{total_steps} steps")
            x1 = x1.to(device, non_blocking=True)
            x0 = torch.randn_like(x1)

            t, xt, ut, eps = FM.sample_location_and_conditional_flow(x0, x1, return_noise=True)
            lambda_t = FM.compute_lambda(t).to(device)

            vt = model(t, xt)
            st = score_model(t, xt)

            flow_loss  = torch.mean((vt - ut) ** 2)
            score_loss = torch.mean((lambda_t[:, None, None, None] * st + eps) ** 2)
            loss = flow_loss + score_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(score_model.parameters()), 1.0)
            optimizer.step(); scheduler.step()

            # EMA（只需本地即可）
            with torch.no_grad():
                for p, ema_p in zip(model.module.parameters(), ema_model.parameters()):
                    ema_p.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            if global_step >= total_steps: break
        if global_step >= total_steps: break

    # 仅 rank-0 保存权重
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            "model":       model.module.state_dict(),
            "score_model": score_model.module.state_dict(),
            "ema_model":   ema_model.state_dict(),
        }, f"{save_dir}/sf2m_cifar10_ddp.pth")

    cleanup_ddp()

    # rank-0 退出 DDP 后做一次可视化采样
    #if rank == 0:
    #    run_sampling(save_dir, x_dim, device, sigma)

# ---------- 采样（与原脚本一致，只移到函数里并复用 EMA 权重） ----------
def run_sampling(save_dir, x_dim, sigma=0.1):
    device = torch.device("cuda")
    ckpt = torch.load(f"{save_dir}/sf2m_cifar10_ddp.pth", map_location=device)
    model = UNetModelWrapper(
        dim=(3, x_dim, x_dim), num_res_blocks=2,
        num_channels=128, channel_mult=[1, 2, 2, 2],
        num_heads=4, num_head_channels=64,
        attention_resolutions="16", dropout=0.1,
    ).to(device).eval()
    score = deepcopy(model)
    score.load_state_dict(ckpt["score_model"])
    model.load_state_dict(ckpt["model"])

    # ODE 采样
    with torch.no_grad():
        traj = torchdiffeq.odeint(
            lambda t, x: model.forward(t, x),
            torch.randn(100, 3, x_dim, x_dim, device=device),
            torch.linspace(0, 1, 2, device=device),
            atol=1e-4, rtol=1e-4, method="dopri5",
        )
    grid = make_grid(traj[-1, :100].clip(-1, 1), value_range=(-1, 1), padding=0, nrow=10)
    ToPILImage()(grid).show(title="SF2M-ODE")

    # SDE 采样
    class SDE(torch.nn.Module):
        noise_type, sde_type = "diagonal", "ito"
        def __init__(self, drift, score, sigma): super().__init__(); self.drift, self.score, self.sigma = drift, score, sigma
        def f(self, t, y):
            y = y.view(-1, 3, x_dim, x_dim)
            return self.drift(t, y).flatten(1) + self.score(t, y).flatten(1)
        
        def g(self, t, y): return torch.ones_like(y) * self.sigma

    sde = SDE(model, score, sigma)
    with torch.no_grad():
        sde_traj = torchsde.sdeint(
            sde,
            torch.randn(50, 3*x_dim*x_dim, device=device),
            ts=torch.linspace(0, 1, 2, device=device), dt=0.01,
        )
    grid = make_grid(sde_traj[-1, :100].view(-1, 3, x_dim, x_dim).clip(-1, 1),
                     value_range=(-1, 1), padding=0, nrow=10)
    ToPILImage()(grid).show(title="SF2M-SDE")

def main():
    world_size = torch.cuda.device_count()
    assert world_size >= 1, "至少需要 1 张 GPU！"
    #mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
    run_sampling("models/cifar10", 32, 0.1)

if __name__ == "__main__":
    main()

