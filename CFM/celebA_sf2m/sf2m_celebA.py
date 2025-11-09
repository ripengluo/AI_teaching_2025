#!/usr/bin/env python
# coding: utf-8
"""
SF2M on CelebA (64x64) with step-wise visualization like celeb.py.

- Train: Schrödinger-Bridge Conditional Flow Matching (CFM) on CelebA
- Sample: ODE and SDE trajectories from white noise to image
- Visualize: save a "strip" (1 x K) per sample, showing states at selected times

References to your uploads:
- Training structure, losses, ODE/SDE sampling style -> cif10_ddp.py  (CIFAR-10 SF2M)  [we remove DDP for simplicity]
- Step-wise visualization idea (times list, reverse display) -> celeb.py (DDPM CelebA)
"""

import os, math, copy, random
from copy import deepcopy
from typing import List, Sequence, Optional

import torch
import torch.nn as nn
import torchdiffeq, torchsde
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from tqdm import tqdm

# from your CIFAR-10 SF2M setup
from torchcfm.conditional_flow_matching import SchrodingerBridgeConditionalFlowMatcher
from torchcfm.models.unet.unet import UNetModelWrapper

# ------------- utils -------------
def set_seed(seed: int = 42):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denorm01(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1] for visualization."""
    return (x.clamp(-1, 1) + 1.0) * 0.5

def save_strip(frames: torch.Tensor, out_path: str, pad: int = 2):
    """
    frames: [T, C, H, W] for a single sample across T times.
    Save as a 1xT strip image from left (noise) to right (final).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frames01 = denorm01(frames)
    grid = make_grid(frames01, nrow=frames01.shape[0], padding=pad)  # one row
    ToPILImage()(grid).save(out_path)

def save_grid(frames: torch.Tensor, nrow: int, out_path: str, pad: int = 2):
    """
    frames: [B, C, H, W] at a single time; make a grid to show multiple samples.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frames01 = denorm01(frames)
    grid = make_grid(frames01, nrow=nrow, padding=pad)
    ToPILImage()(grid).save(out_path)

# ------------- dataset & model -------------
def get_celeba_loader(x_dim: int = 64, batch_size: int = 128, num_workers: int = 4) -> DataLoader:
    tfm = transforms.Compose([
        transforms.Resize((x_dim, x_dim)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),  # [-1,1] same as CIFAR-10 script
    ])
    ds = datasets.CelebA(root="data", split="all", download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=num_workers, pin_memory=True)
    return loader

def build_unet(x_dim: int = 64) -> UNetModelWrapper:
    # Match the CIFAR-10 script's UNet config; only the spatial size changes to 64x64.
    # attention_resolutions kept at "16" which also hits a 16x16 stage for 64x64 inputs.
    model = UNetModelWrapper(
        dim=(3, x_dim, x_dim),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    )
    return model

# ------------- training (single process) -------------
def train_celebA_sf2m(
    x_dim: int = 64,
    batch_size: int = 128,
    total_steps: int = 400_000,
    sigma: float = 0.1,
    ema_decay: float = 0.9999,
    lr: float = 2e-4,
    seed: int = 42,
    save_dir: str = "models/celebA_sf2m",
    log_every: int = 1000,
):
    """
    Train SF2M on CelebA. Based on your cif10_ddp.py loop but simplified to single-GPU/CPU.
    """
    set_seed(seed)
    device = get_device()
    loader = get_celeba_loader(x_dim=x_dim, batch_size=batch_size)
    n_epochs = math.ceil(total_steps / len(loader))
    print(f"Training with {n_epochs} epochs")

    drift = build_unet(x_dim).to(device)
    score = deepcopy(drift).to(device)
    ema_drift = deepcopy(drift).to(device)  # only EMA for the drift (as in your script)

    opt = torch.optim.Adam(list(drift.parameters()) + list(score.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: min(step, 5_000) / 5_000)

    FM = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)

    os.makedirs(save_dir, exist_ok=True)
    pbar = tqdm(total=total_steps, unit="step")
    global_step = 0

    for epoch in range(n_epochs):
        for x1, _ in loader:
            global_step += 1
            x1 = x1.to(device, non_blocking=True)
            x0 = torch.randn_like(x1)  # Gaussian reference

            # sample (t, x_t, u_t) and noise eps, same as in your CIFAR-10 code
            t, xt, ut, eps = FM.sample_location_and_conditional_flow(x0, x1, return_noise=True)
            lambda_t = FM.compute_lambda(t).to(device)

            vt = drift(t, xt)         # drift prediction
            st = score(t, xt)         # score prediction

            flow_loss  = torch.mean((vt - ut) ** 2)
            score_loss = torch.mean((lambda_t[:, None, None, None] * st + eps) ** 2)
            loss = flow_loss + score_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(drift.parameters()) + list(score.parameters()), 1.0)
            opt.step(); sched.step()

            # EMA on drift, as in your code
            with torch.no_grad():
                for p, ema_p in zip(drift.parameters(), ema_drift.parameters()):
                    ema_p.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            if global_step % log_every == 0:
                tqdm.write(f"[step {global_step}] loss={loss.item():.4f} (flow={flow_loss.item():.4f}, score={score_loss.item():.4f})")

            pbar.update(1)
            pbar.set_description(f"{global_step}/{total_steps} steps")
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    # save weights (drift, score, ema_drift)
    ckpt_path = os.path.join(save_dir, f"sf2m_celeba_{x_dim}.pth")
    torch.save({
        "drift": drift.state_dict(),
        "score": score.state_dict(),
        "ema_drift": ema_drift.state_dict(),
        "x_dim": x_dim,
        "sigma": sigma,
    }, ckpt_path)
    print(f"Saved checkpoint to: {ckpt_path}")

# ------------- sampling & visualization -------------
@torch.no_grad()
def sample_ode_trajectory(
    ckpt_path: str,
    n_samples: int = 8,
    ts: Optional[Sequence[float]] = None,
    out_dir: str = "samples/ode",
):
    """
    ODE sampling like your CIFAR-10 example, but we request the solution at custom time stamps.
    Saves:
      - grid at each time (all samples)
      - per-sample strip from noise to final
    """
    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    x_dim = ckpt.get("x_dim", 64)
    sigma = ckpt.get("sigma", 0.1)

    drift = build_unet(x_dim).to(device).eval()
    score = build_unet(x_dim).to(device).eval()  # not used in ODE, but we keep symmetry
    # prefer EMA drift if available
    drift.load_state_dict(ckpt.get("ema_drift", ckpt["drift"]))
    score.load_state_dict(ckpt["score"])

    if ts is None:
        # 10 frames from t=0 (noise) to t=1 (image)
        ts = torch.linspace(0.0, 1.0, 10, device=device)
    else:
        ts = torch.tensor(ts, device=device).float()

    y0 = torch.randn(n_samples, 3, x_dim, x_dim, device=device)

    # ODE: dy/dt = drift(t, y)
    def ode_fn(t, y):
        return drift(t, y)

    traj = torchdiffeq.odeint(ode_fn, y0, ts, atol=1e-4, rtol=1e-4, method="dopri5")
    # traj: [T, B, C, H, W]

    # save per-time grids and per-sample strips
    os.makedirs(out_dir, exist_ok=True)
    # time-wise grids
    for i, tval in enumerate(ts):
        frames = traj[i]  # [B, C, H, W]
        save_grid(frames, nrow=int(math.sqrt(n_samples)), out_path=os.path.join(out_dir, f"time_{i:02d}_t{float(tval):.2f}.png"))

    # sample-wise strips
    for b in range(n_samples):
        strip = traj[:, b]  # [T, C, H, W]
        save_strip(strip, out_path=os.path.join(out_dir, f"sample_{b:02d}_strip.png"))

@torch.no_grad()
def sample_sde_trajectory(
    ckpt_path: str,
    n_samples: int = 8,
    ts: Optional[Sequence[float]] = None,
    dt: float = 0.01,
    out_dir: str = "samples/sde",
):
    """
    SDE sampling as in your CIFAR-10 example: f = drift + score, g = sigma * I
    We evaluate at selected time stamps and visualize like celeb.py's stepwise display.
    """
    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    x_dim = ckpt.get("x_dim", 64)
    sigma = ckpt.get("sigma", 0.1)

    drift = build_unet(x_dim).to(device).eval()
    score = build_unet(x_dim).to(device).eval()
    drift.load_state_dict(ckpt.get("ema_drift", ckpt["drift"]))
    score.load_state_dict(ckpt["score"])

    if ts is None:
        ts = torch.linspace(0.0, 1.0, 10, device=device)
    else:
        ts = torch.tensor(ts, device=device).float()

    class SDE(nn.Module):
        noise_type, sde_type = "diagonal", "ito"
        def __init__(self, drift, score, sigma, c=3, h=64, w=64):
            super().__init__()
            self.drift, self.score = drift, score
            self.sigma = sigma
            self.c, self.h, self.w = c, h, w
        def f(self, t, y):
            y = y.view(-1, self.c, self.h, self.w)
            return (self.drift(t, y) + self.score(t, y)).flatten(1)
        def g(self, t, y):
            return torch.ones_like(y) * self.sigma

    sde = SDE(drift, score, sigma, c=3, h=x_dim, w=x_dim).to(device)

    y0 = torch.randn(n_samples, 3 * x_dim * x_dim, device=device)
    traj = torchsde.sdeint(sde, y0, ts=ts, dt=dt)  # [T, B, D]
    traj = traj.view(len(ts), n_samples, 3, x_dim, x_dim)

    os.makedirs(out_dir, exist_ok=True)
    # time-wise grids
    for i, tval in enumerate(ts):
        frames = traj[i]  # [B, C, H, W]
        save_grid(frames, nrow=int(math.sqrt(n_samples)), out_path=os.path.join(out_dir, f"time_{i:02d}_t{float(tval):.2f}.png"))

    # sample-wise strips
    for b in range(n_samples):
        strip = traj[:, b]  # [T, C, H, W]
        save_strip(strip, out_path=os.path.join(out_dir, f"sample_{b:02d}_strip.png"))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train SF2M on CelebA")
    parser.add_argument("--sample_ode", action="store_true", help="Sample ODE and visualize")
    parser.add_argument("--sample_sde", action="store_true", help="Sample SDE and visualize")
    parser.add_argument("--ckpt", type=str, default="models/celebA_sf2m/sf2m_celeba_64.pth", help="Checkpoint path")
    parser.add_argument("--x_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--total_steps", type=int, default=100_000)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--times", type=float, nargs="*", default=None,
                        help="Custom times in [0,1], e.g., --times 0 0.05 0.1 0.2 ... 1")
    parser.add_argument("--n_samples", type=int, default=8)
    args = parser.parse_args()

    if args.train:
        train_celebA_sf2m(
            x_dim=args.x_dim,
            batch_size=args.batch_size,
            total_steps=args.total_steps,
            sigma=args.sigma,
            ema_decay=args.ema_decay,
            lr=args.lr,
            seed=args.seed,
            save_dir=os.path.dirname(args.ckpt) if args.ckpt else "models/celebA_sf2m",
        )

    if args.sample_ode:
        sample_ode_trajectory(
            ckpt_path=args.ckpt,
            n_samples=args.n_samples,
            ts=args.times,
            out_dir="samples/ode",
        )

    if args.sample_sde:
        sample_sde_trajectory(
            ckpt_path=args.ckpt,
            n_samples=args.n_samples,
            ts=args.times,
            out_dir="samples/sde",
        )

if __name__ == "__main__":
    main()

