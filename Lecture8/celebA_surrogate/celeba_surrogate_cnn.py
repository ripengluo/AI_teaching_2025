#!/usr/bin/env python3
# coding: utf-8

"""
CelebA surrogate CNN classifier for attributes, designed to be used later as a reward model.

- Multi-label (BCEWithLogitsLoss) for arbitrary attribute list (default: Curly_Hair + Smiling)
- Train / Validation / Test split
- Saves best checkpoint by val loss
- Provides reward helpers:
    - reward_logsigmoid(logits, target_bits): sum log-sigmoid for target bits (stable)
    - reward_prob(logits, target_bits): sum probabilities matching target bits

Run examples:
  # train
  python celeba_surrogate_cnn.py --train --attrs Curly_Hair Smiling --x_dim 64 --epochs 10

  # evaluate only
  python celeba_surrogate_cnn.py --eval --ckpt checkpoints/surrogate_celeba_64_attrs_Curly_Hair-Smiling.pth

  # use as reward (in your own code):
  #   model, meta = load_surrogate(ckpt)
  #   logits = model(x)  # x in [-1,1]
  #   r = reward_logsigmoid(logits, target_bits=[1,1])  # want both Curly_Hair=1 & Smiling=1
"""

import os
import math
import time
import argparse
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------- utils ----------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_attr_indices(attr_names: Sequence[str], attrs: Sequence[str]) -> List[int]:
    name_to_idx = {n: i for i, n in enumerate(attr_names)}
    missing = [a for a in attrs if a not in name_to_idx]
    if missing:
        raise ValueError(
            f"Unknown attrs: {missing}. "
            f"Example available names: {list(attr_names)[:15]} ..."
        )
    return [name_to_idx[a] for a in attrs]


# ---------------- model ----------------
class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))


class SimpleCNN(nn.Module):
    """
    A small CNN for 64x64 (also works for other x_dim if divisible by 2 a few times).
    Outputs logits [B, K] for K attributes.
    """
    def __init__(self, num_attrs: int, base: int = 128, dropout: float = 0.1):
        super().__init__()
        self.stem = ConvBlock(3, base, stride=1, dropout=0.0)
        self.b1 = nn.Sequential(
            ConvBlock(base, base, stride=2, dropout=dropout),
            ConvBlock(base, base, stride=1, dropout=dropout),
        )
        self.b2 = nn.Sequential(
            ConvBlock(base, base * 2, stride=2, dropout=dropout),
            ConvBlock(base * 2, base * 2, stride=1, dropout=dropout),
        )
        self.b3 = nn.Sequential(
            ConvBlock(base * 2, base * 4, stride=2, dropout=dropout),
            ConvBlock(base * 4, base * 4, stride=1, dropout=dropout),
        )
        self.b4 = nn.Sequential(
            ConvBlock(base * 4, base * 8, stride=2, dropout=dropout),
            ConvBlock(base * 8, base * 8, stride=1, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base * 8, num_attrs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        return self.head(x)


# ---------------- data ----------------
def make_celeba_transforms(x_dim: int, train: bool) -> transforms.Compose:
    # Match your SF2M pipeline: Normalize((0.5,)*3,(0.5,)*3) to [-1,1]. :contentReference[oaicite:1]{index=1}
    tfm = [
        transforms.Resize((x_dim, x_dim)),
    ]
    if train:
        tfm.append(transforms.RandomHorizontalFlip())
    tfm += [
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ]
    return transforms.Compose(tfm)


def make_loaders(
    data_root: str,
    x_dim: int,
    batch_size: int,
    num_workers: int,
    download: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    ds_train = datasets.CelebA(
        root=data_root,
        split="train",
        target_type="attr",
        download=download,
        transform=make_celeba_transforms(x_dim, train=True),
    )
    ds_val = datasets.CelebA(
        root=data_root,
        split="valid",
        target_type="attr",
        download=download,
        transform=make_celeba_transforms(x_dim, train=False),
    )
    ds_test = datasets.CelebA(
        root=data_root,
        split="test",
        target_type="attr",
        download=download,
        transform=make_celeba_transforms(x_dim, train=False),
    )

    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader, list(ds_train.attr_names)


# ---------------- loss/metrics/reward ----------------
@torch.no_grad()
def compute_pos_weight(train_loader: DataLoader, attr_indices: Sequence[int], device: torch.device) -> torch.Tensor:
    """
    pos_weight for BCEWithLogitsLoss: weight positive examples as (Nneg/Npos) per attribute.
    """
    pos = torch.zeros(len(attr_indices), device=device)
    neg = torch.zeros(len(attr_indices), device=device)
    for _x, attrs in train_loader:
        a = attrs[:, attr_indices].to(device)
        y = (a > 0).float()  # CelebA attr is -1/1
        pos += y.sum(dim=0)
        neg += (1.0 - y).sum(dim=0)
    # avoid division by zero
    pos = pos.clamp_min(1.0)
    return (neg / pos).clamp(min=1.0, max=100.0)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    attr_indices: Sequence[int],
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0

    # per-attr accuracy (threshold 0.5 on sigmoid)
    correct = torch.zeros(len(attr_indices), device=device)
    count = 0

    for x, attrs in loader:
        x = x.to(device, non_blocking=True)
        y = (attrs[:, attr_indices] > 0).float().to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(amp and device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total += bs

        pred = (torch.sigmoid(logits) > 0.5).float()
        correct += (pred == y).float().sum(dim=0)
        count += bs

    acc_per = (correct / max(1, count)).detach().cpu().tolist()
    return {
        "loss": total_loss / max(1, total),
        "acc_mean": float(sum(acc_per) / max(1, len(acc_per))),
        **{f"acc_{i}": float(acc_per[i]) for i in range(len(acc_per))},
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    attr_indices: Sequence[int],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))
    total_loss = 0.0
    total = 0

    for x, attrs in loader:
        x = x.to(device, non_blocking=True)
        y = (attrs[:, attr_indices] > 0).float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(amp and device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total += bs

    return total_loss / max(1, total)


def reward_logsigmoid(logits: torch.Tensor, target_bits: Sequence[int]) -> torch.Tensor:
    """
    Stable reward: sum_i log(sigmoid( (2*bit-1)*logit_i )).
    logits: [B, K]
    target_bits: length K, each in {0,1}
    returns: [B]
    """
    device = logits.device
    t = torch.tensor(target_bits, device=device, dtype=logits.dtype).view(1, -1)
    s = (2.0 * t - 1.0)  # +1 if want 1, -1 if want 0
    return F.logsigmoid(s * logits).sum(dim=1)


def reward_prob(logits: torch.Tensor, target_bits: Sequence[int]) -> torch.Tensor:
    """
    Simpler reward: sum of probabilities matching target bits.
    returns: [B], range [0, K]
    """
    p = torch.sigmoid(logits)
    t = torch.tensor(target_bits, device=logits.device, dtype=logits.dtype).view(1, -1)
    return (t * p + (1.0 - t) * (1.0 - p)).sum(dim=1)


# ---------------- checkpoint I/O ----------------
@dataclass
class SurrogateMeta:
    attrs: List[str]
    attr_indices: List[int]
    x_dim: int
    normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)


def save_ckpt(path: str, model: nn.Module, meta: SurrogateMeta, extra: Optional[dict] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "meta": {
            "attrs": meta.attrs,
            "attr_indices": meta.attr_indices,
            "x_dim": meta.x_dim,
            "normalize_mean": meta.normalize_mean,
            "normalize_std": meta.normalize_std,
        }
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_surrogate(path: str, device: Optional[torch.device] = None) -> Tuple[nn.Module, SurrogateMeta]:
    if device is None:
        device = get_device()
    ckpt = torch.load(path, map_location=device)
    m = ckpt["meta"]
    meta = SurrogateMeta(
        attrs=list(m["attrs"]),
        attr_indices=list(m["attr_indices"]),
        x_dim=int(m["x_dim"]),
        normalize_mean=tuple(m.get("normalize_mean", (0.5, 0.5, 0.5))),
        normalize_std=tuple(m.get("normalize_std", (0.5, 0.5, 0.5))),
    )
    model = SimpleCNN(num_attrs=len(meta.attrs))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, meta


# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train + validate, save best ckpt")
    parser.add_argument("--eval", action="store_true", help="Evaluate ckpt on val/test")
    parser.add_argument("--ckpt", type=str, default="", help="Checkpoint path for --eval or resume")

    parser.add_argument("--data_root", type=str, default="data", help="CelebA root (same as SF2M script)")
    parser.add_argument("--download", action="store_true", help="Download CelebA if not present")

    parser.add_argument("--attrs", type=str, nargs="*", default=["Curly_Hair", "Smiling"])
    parser.add_argument("--x_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_pos_weight", action="store_true", help="Use pos_weight to counter class imbalance")
    parser.add_argument("--out_dir", type=str, default="checkpoints")

    args = parser.parse_args()
    if not (args.train or args.eval):
        parser.error("Please specify --train or --eval")

    set_seed(args.seed)
    device = get_device()

    train_loader, val_loader, test_loader, attr_names = make_loaders(
        data_root=args.data_root,
        x_dim=args.x_dim,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
    )

    # Allow a tiny alias for convenience
    alias = {"glasses": "Eyeglasses", "curly": "Curly_Hair", "smile": "Smiling"}
    attrs = [alias.get(a, a) for a in args.attrs]

    attr_indices = get_attr_indices(attr_names, attrs)

    if args.eval:
        if not args.ckpt:
            raise ValueError("--eval requires --ckpt")
        model, meta = load_surrogate(args.ckpt, device=device)
        # meta.attr_indices exists, but we evaluate with current attr_indices for safety
        pos_weight = None
        if args.use_pos_weight:
            pos_weight = compute_pos_weight(train_loader, attr_indices, device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        val_stats = eval_epoch(model, val_loader, attr_indices, criterion, device, amp=args.amp)
        test_stats = eval_epoch(model, test_loader, attr_indices, criterion, device, amp=args.amp)

        print(f"[EVAL] ckpt={args.ckpt}")
        print(f"  attrs={attrs}")
        print(f"  VAL : loss={val_stats['loss']:.4f} | acc_mean={val_stats['acc_mean']:.4f}")
        print(f"  TEST: loss={test_stats['loss']:.4f} | acc_mean={test_stats['acc_mean']:.4f}")
        for i, a in enumerate(attrs):
            print(f"    acc({a}) VAL={val_stats[f'acc_{i}']:.4f} | TEST={test_stats[f'acc_{i}']:.4f}")
        return

    # training
    model = SimpleCNN(num_attrs=len(attrs), dropout=args.dropout).to(device)

    # resume optional
    start_epoch = 0
    best_val = float("inf")
    if args.ckpt and os.path.isfile(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        best_val = float(ckpt.get("extra", {}).get("best_val_loss", best_val))
        start_epoch = int(ckpt.get("extra", {}).get("epoch", 0))
        print(f"[RESUME] loaded {args.ckpt} | start_epoch={start_epoch} | best_val={best_val:.4f}")

    pos_weight = None
    if args.use_pos_weight:
        pos_weight = compute_pos_weight(train_loader, attr_indices, device)
        print(f"[pos_weight] {pos_weight.detach().cpu().tolist()}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # cosine schedule (epochs-based)
    def lr_lambda(ep):
        return 0.5 * (1.0 + math.cos(math.pi * ep / max(1, args.epochs)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    meta = SurrogateMeta(attrs=list(attrs), attr_indices=list(attr_indices), x_dim=args.x_dim)

    ckpt_name = f"surrogate_celeba_{args.x_dim}_attrs_{'-'.join(attrs)}.pth"
    best_path = os.path.join(args.out_dir, ckpt_name)

    print(f"[TRAIN] device={device} | x_dim={args.x_dim} | attrs={attrs}")
    print(f"[TRAIN] out={best_path}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_epoch(
            model, train_loader, attr_indices, optimizer, criterion, device, amp=args.amp, grad_clip=args.grad_clip
        )
        val_stats = eval_epoch(model, val_loader, attr_indices, criterion, device, amp=args.amp)
        scheduler.step()

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"[epoch {epoch+1:03d}/{args.epochs:03d}] "
            f"lr={lr:.2e} | train_loss={train_loss:.4f} | val_loss={val_stats['loss']:.4f} "
            f"| val_acc_mean={val_stats['acc_mean']:.4f} | {dt:.1f}s"
        )
        for i, a in enumerate(attrs):
            print(f"  - val_acc({a})={val_stats[f'acc_{i}']:.4f}")

        # save best
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_ckpt(
                best_path,
                model,
                meta,
                extra={"epoch": epoch + 1, "best_val_loss": best_val},
            )
            print(f"  [BEST] saved to {best_path} | best_val_loss={best_val:.4f}")

    # final test on best
    best_model, _ = load_surrogate(best_path, device=device)
    test_stats = eval_epoch(best_model, test_loader, attr_indices, criterion, device, amp=args.amp)
    print(f"[DONE] best_ckpt={best_path}")
    print(f"  TEST: loss={test_stats['loss']:.4f} | acc_mean={test_stats['acc_mean']:.4f}")
    for i, a in enumerate(attrs):
        print(f"    acc({a}) TEST={test_stats[f'acc_{i}']:.4f}")


if __name__ == "__main__":
    main()

