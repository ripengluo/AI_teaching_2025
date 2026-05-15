#!/usr/bin/env python
# coding: utf-8

"""
SF2M on CelebA (64x64) + conditional generation (first: conditioned on `Sex` attr).

Key points:
- CelebA target_type="attr"
- labels = (attrs[:, idx_Sex] > 0).long()  # 0=female, 1=male
- UNet: class_cond=True, num_classes=2
- Training calls model(t, xt, labels)
- FlowMatcher guided_sample_* interface is version-dependent (y vs y1), so we handle both.

Later extension (multi-attr):
cond_attrs = ["Sex","Smiling","Young","Bangs"]
labels = bit-pack to class id in [0, 2^K-1], set num_classes=2**K
"""

import os, math, random
from copy import deepcopy
from typing import List, Sequence, Optional, Tuple

import torch
import torch.nn as nn
import torchdiffeq
import torchsde
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from torchcfm.conditional_flow_matching import SchrodingerBridgeConditionalFlowMatcher
from torchcfm.models.unet.unet import UNetModelWrapper


ATTR_NAME_ALIASES = {
    "Sex": "Male",
}


# ---------------- utils ----------------
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    grid = make_grid(frames01, nrow=frames01.shape[0], padding=pad)
    ToPILImage()(grid).save(out_path)


def save_grid(frames: torch.Tensor, nrow: int, out_path: str, pad: int = 2):
    """frames: [B, C, H, W] at a single time; make a grid to show multiple samples."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frames01 = denorm01(frames)
    grid = make_grid(frames01, nrow=nrow, padding=pad)
    ToPILImage()(grid).save(out_path)


# ---------------- conditioning helpers ----------------
def resolve_attr_name(attr_name: str) -> str:
    return ATTR_NAME_ALIASES.get(attr_name, attr_name)


def normalize_cond_attrs(cond_attrs: Sequence[str]) -> List[str]:
    return [resolve_attr_name(attr) for attr in cond_attrs]


def public_attr_name(attr_name: str) -> str:
    return "Sex" if attr_name == "Male" else attr_name


def public_cond_attrs(cond_attrs: Sequence[str]) -> List[str]:
    return [public_attr_name(attr) for attr in cond_attrs]


def resolve_ckpt_path(ckpt_path: str) -> str:
    if os.path.exists(ckpt_path):
        return ckpt_path
    legacy_ckpt_path = ckpt_path.replace("_cond_Sex", "_cond_Male")
    if legacy_ckpt_path != ckpt_path and os.path.exists(legacy_ckpt_path):
        return legacy_ckpt_path
    return ckpt_path


def get_attr_indices(attr_names: Sequence[str], cond_attrs: Sequence[str]) -> List[int]:
    cond_attrs = normalize_cond_attrs(cond_attrs)
    name_to_idx = {n: i for i, n in enumerate(attr_names)}
    missing = [a for a in cond_attrs if a not in name_to_idx]
    if missing:
        raise ValueError(
            f"cond_attrs has unknown names: {missing}. "
            f"Available example: {list(attr_names)[:10]}..."
        )
    return [name_to_idx[a] for a in cond_attrs]


def attrs_to_class_id(attrs: torch.Tensor, cond_attr_indices: Sequence[int]) -> torch.LongTensor:
    """
    attrs: [B, 40] (CelebA attrs, usually -1/1).
    Take K selected attrs -> bits (0/1) -> bit-pack into class id [0, 2^K-1].
    """
    if len(cond_attr_indices) == 0:
        raise ValueError("cond_attr_indices cannot be empty")
    bits = (attrs[:, cond_attr_indices] > 0).long()  # [B, K], 0/1
    weights = (2 ** torch.arange(len(cond_attr_indices), device=bits.device, dtype=torch.long)).view(1, -1)
    return torch.sum(bits * weights, dim=1)  # [B]


def make_class_id_from_bits(bits: Sequence[int]) -> int:
    cid = 0
    for i, b in enumerate(bits):
        cid |= (int(b) & 1) << i
    return cid


# ---------------- dataset & model ----------------
def get_celeba_loader(
    x_dim: int = 64,
    batch_size: int = 128,
    num_workers: int = 4,
    split: str = "all",
) -> Tuple[DataLoader, List[str]]:
    tfm = transforms.Compose(
        [
            transforms.Resize((x_dim, x_dim)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),  # [-1,1]
        ]
    )
    ds = datasets.CelebA(
        root="data",
        split=split,
        target_type="attr",  # MUST return attribute vector
        download=True,
        transform=tfm,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, list(ds.attr_names)


def build_unet(x_dim: int = 64, num_classes: int = 2) -> UNetModelWrapper:
    """UNet for 64x64 CelebA, class-conditional."""
    model = UNetModelWrapper(
        dim=(3, x_dim, x_dim),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
        num_classes=num_classes,
        class_cond=True,
    )
    return model


"""
@torch.no_grad()
def visualize_train_grid(
    x_dim: int = 64,
    grid_rows: int = 3,
    grid_cols: int = 3,
    out_path: str = "samples/train_examples/train_grid_3x3.png",
    num_workers: int = 4,
):
    batch_size = grid_rows * grid_cols
    loader, _ = get_celeba_loader(x_dim=x_dim, batch_size=batch_size, num_workers=num_workers)
    x, _attrs = next(iter(loader))
    x = x[:batch_size]
    save_grid(x, nrow=grid_cols, out_path=out_path)
    print(f"Saved training grid ({grid_rows}x{grid_cols}) to: {out_path}")
"""

@torch.no_grad()
def visualize_train_grid(
    x_dim: int = 64,
    grid_rows: int = 3,
    grid_cols: int = 3,
    out_path: str = "samples/train_examples/train_grid_3x3.png",
    num_workers: int = 4,
    cond_attrs=("Sex", "Eyeglasses", "Young"),
    label_mode: str = "abbr2line",   # "bits" | "abbr" | "abbr2line"
    font_size: int = None,          # None -> auto
):
    from PIL import ImageDraw, ImageFont
    from torchvision.transforms import ToPILImage
    import os

    batch_size = grid_rows * grid_cols
    loader, attr_names = get_celeba_loader(x_dim=x_dim, batch_size=batch_size, num_workers=num_workers)
    x, attrs = next(iter(loader))
    x = x[:batch_size]
    attrs = attrs[:batch_size]

    cond_attr_indices = get_attr_indices(attr_names, cond_attrs)
    cond_attrs = tuple(public_cond_attrs(cond_attrs))
    bits = (attrs[:, cond_attr_indices] > 0).long()  # [B, K]

    # --- font: prefer truetype (resizable). fallback to default ---
    if font_size is None:
        # 64x64 下建议 7~9
        font_size = max(6, min(9, x_dim // 8))
    font = None
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]:
        try:
            font = ImageFont.truetype(p, font_size)
            break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()  # 不能调大小，但我们会用更短的 label

    def make_label(bit_row):
        bit_str = "".join(str(int(b)) for b in bit_row.tolist())
        if label_mode == "bits":
            # 例: 1011 (顺序就是 cond_attrs)
            return [bit_str]
        # 缩写：Sex->S, Smiling->S, Bangs->B, Young->Y
        abbr = [a[0].upper() for a in cond_attrs]
        if label_mode == "abbr":
            # 例: M1 S0 B1 Y1
            return [" ".join([f"{abbr[i]}{int(bit_row[i])}" for i in range(len(abbr))])]
        # 两行：第一行属性缩写，第二行 bits，更容易塞进 64x64
        return [" ".join(abbr), " ".join(str(int(b)) for b in bit_row.tolist())]

    to_pil = ToPILImage()
    to_tensor = transforms.ToTensor()

    imgs = []
    for i in range(batch_size):
        pil = to_pil(denorm01(x[i]).cpu())
        draw = ImageDraw.Draw(pil)

        lines = make_label(bits[i])
        # 计算文本块大小
        line_h = draw.textbbox((0, 0), "Ag", font=font)[3]
        pad = 2
        max_w = 0
        for ln in lines:
            w = draw.textbbox((0, 0), ln, font=font)[2]
            max_w = max(max_w, w)

        box_w = min(pil.size[0], max_w + 2 * pad)
        box_h = min(pil.size[1], len(lines) * line_h + 2 * pad)

        # 左上角放一个黑底框
        draw.rectangle([0, 0, box_w, box_h], fill=(0, 0, 0))

        y = pad
        for ln in lines:
            draw.text((pad, y), ln, fill=(255, 255, 255), font=font)
            y += line_h

        imgs.append(to_tensor(pil))

    imgs = torch.stack(imgs, dim=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid = make_grid(imgs, nrow=grid_cols, padding=2)
    ToPILImage()(grid).save(out_path)

    print(f"Saved training grid w/ labels ({grid_rows}x{grid_cols}) to: {out_path}")
    print(f"cond_attrs order = {list(cond_attrs)} | label_mode={label_mode} | font_size={font_size}")


# ---------------- training ----------------
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
    cond_attrs: Sequence[str] = ("Sex",),
    num_workers: int = 4,
    dry_run: bool = False,
):
    """
    Train SF2M on CelebA with conditioning on CelebA attributes (first: Sex, where 0=female and 1=male).
    """
    set_seed(seed)
    device = get_device()

    loader, attr_names = get_celeba_loader(
        x_dim=x_dim, batch_size=batch_size, num_workers=num_workers, split="all"
    )
    cond_attr_indices = get_attr_indices(attr_names, cond_attrs)
    cond_attrs = list(public_cond_attrs(cond_attrs))
    num_classes = 2 ** len(cond_attr_indices)

    save_dir = save_dir or "models/celebA_sf2m"
    os.makedirs(save_dir, exist_ok=True)

    drift = build_unet(x_dim, num_classes=num_classes).to(device)
    score = deepcopy(drift).to(device)
    ema_drift = deepcopy(drift).to(device)

    opt = torch.optim.Adam(list(drift.parameters()) + list(score.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: min(step, 5_000) / 5_000)

    FM = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)

    n_epochs = math.ceil(total_steps / len(loader))
    print(f"Training: epochs={n_epochs} | steps={total_steps} | cond_attrs={list(cond_attrs)} | num_classes={num_classes}")
    if dry_run:
        print("[dry_run] Will run exactly 1 iteration and exit after one optimizer step.")

    pbar = tqdm(total=(1 if dry_run else total_steps), unit="step")
    global_step = 0

    for _epoch in range(n_epochs):
        for x1, attrs in loader:
            global_step += 1
            x1 = x1.to(device, non_blocking=True)
            attrs = attrs.to(device, non_blocking=True)

            labels = attrs_to_class_id(attrs, cond_attr_indices).long()  # [B]
            x0 = torch.randn_like(x1)

            t, xt, ut, _, labels, eps = FM.guided_sample_location_and_conditional_flow(x0, x1, y1=labels, return_noise=True)


            lambda_t = FM.compute_lambda(t).to(device)
            vt = drift(t, xt, labels)
            st = score(t, xt, labels)

            flow_loss = torch.mean((vt - ut) ** 2)
            score_loss = torch.mean((lambda_t[:, None, None, None] * st + eps) ** 2)
            loss = flow_loss + score_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(drift.parameters()) + list(score.parameters()), 1.0)
            opt.step()
            sched.step()

            with torch.no_grad():
                for p, ema_p in zip(drift.parameters(), ema_drift.parameters()):
                    ema_p.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            if global_step % log_every == 0 or dry_run:
                tqdm.write(
                    f"[step {global_step}] loss={loss.item():.4f} (flow={flow_loss.item():.4f}, score={score_loss.item():.4f}) "
                    f"| labels unique={torch.unique(labels).tolist()}"
                )

            pbar.update(1)
            if dry_run:
                pbar.close()
                print("[dry_run] OK: one iteration finished.")
                return

            if global_step >= total_steps:
                pbar.close()
                break
        if global_step >= total_steps:
            break

    ckpt_path = os.path.join(save_dir, f"sf2m_celeba_{x_dim}_cond_{'-'.join(cond_attrs)}.pth")
    torch.save(
        {
            "drift": drift.state_dict(),
            "score": score.state_dict(),
            "ema_drift": ema_drift.state_dict(),
            "x_dim": x_dim,
            "sigma": sigma,
            "cond_attrs": list(cond_attrs),
            "cond_attr_indices": list(cond_attr_indices),
            "num_classes": num_classes,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to: {ckpt_path}")


# ---------------- sampling ----------------
@torch.no_grad()
def sample_ode_trajectory(
    ckpt_path: str,
    n_samples: int = 8,
    ts: Optional[Sequence[float]] = None,
    out_dir: str = "samples/ode",
    cond_bits: Optional[Sequence[int]] = None,
):
    device = get_device()
    ckpt_path = resolve_ckpt_path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    x_dim = int(ckpt.get("x_dim", 64))

    cond_attrs = public_cond_attrs(ckpt.get("cond_attrs", ["Sex"]))
    num_classes = int(ckpt.get("num_classes", 2 ** len(cond_attrs)))

    drift = build_unet(x_dim, num_classes=num_classes).to(device).eval()
    drift.load_state_dict(ckpt.get("ema_drift", ckpt["drift"]))

    if ts is None:
        ts = torch.linspace(0.0, 1.0, 10, device=device)
    else:
        ts = torch.tensor(ts, device=device).float()

    if cond_bits is None:
        cond_bits = [1] * len(cond_attrs)
    cond_class = make_class_id_from_bits(cond_bits)
    labels = torch.full((n_samples,), cond_class, device=device, dtype=torch.long)

    y0 = torch.randn(n_samples, 3, x_dim, x_dim, device=device)

    def ode_fn(t, y):
        return drift(t, y, labels)

    traj = torchdiffeq.odeint(ode_fn, y0, ts, atol=1e-4, rtol=1e-4, method="dopri5")

    os.makedirs(out_dir, exist_ok=True)
    nrow = max(1, int(math.sqrt(n_samples)))
    for i, tval in enumerate(ts):
        save_grid(traj[i], nrow=nrow, out_path=os.path.join(out_dir, f"time_{i:02d}_t{float(tval):.2f}.png"))
    for b in range(n_samples):
        save_strip(traj[:, b], out_path=os.path.join(out_dir, f"sample_{b:02d}_strip.png"))

    print(f"Saved ODE samples to: {out_dir} | cond_attrs={cond_attrs} | cond_bits={list(cond_bits)}")


@torch.no_grad()
def sample_sde_trajectory(
    ckpt_path: str,
    n_samples: int = 8,
    ts: Optional[Sequence[float]] = None,
    dt: float = 0.01,
    out_dir: str = "samples/sde",
    cond_bits: Optional[Sequence[int]] = None,
):
    device = get_device()
    ckpt_path = resolve_ckpt_path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    x_dim = int(ckpt.get("x_dim", 64))
    sigma = float(ckpt.get("sigma", 0.1))

    cond_attrs = public_cond_attrs(ckpt.get("cond_attrs", ["Sex"]))
    num_classes = int(ckpt.get("num_classes", 2 ** len(cond_attrs)))

    drift = build_unet(x_dim, num_classes=num_classes).to(device).eval()
    score = build_unet(x_dim, num_classes=num_classes).to(device).eval()
    drift.load_state_dict(ckpt.get("ema_drift", ckpt["drift"]))
    score.load_state_dict(ckpt["score"])

    if ts is None:
        ts = torch.linspace(0.0, 1.0, 10, device=device)
    else:
        ts = torch.tensor(ts, device=device).float()

    if cond_bits is None:
        cond_bits = [1] * len(cond_attrs)
    cond_class = make_class_id_from_bits(cond_bits)
    labels = torch.full((n_samples,), cond_class, device=device, dtype=torch.long)

    class SDE(nn.Module):
        noise_type, sde_type = "diagonal", "ito"

        def __init__(self, drift, score, sigma, labels, c=3, h=64, w=64):
            super().__init__()
            self.drift, self.score = drift, score
            self.sigma = sigma
            self.labels = labels
            self.c, self.h, self.w = c, h, w

        def f(self, t, y):
            y = y.view(-1, self.c, self.h, self.w)
            return (self.drift(t, y, self.labels) + self.score(t, y, self.labels)).flatten(1)

        def g(self, t, y):
            return torch.ones_like(y) * self.sigma

    sde = SDE(drift, score, sigma, labels, c=3, h=x_dim, w=x_dim).to(device)

    y0 = torch.randn(n_samples, 3 * x_dim * x_dim, device=device)
    traj = torchsde.sdeint(sde, y0, ts=ts, dt=dt).view(len(ts), n_samples, 3, x_dim, x_dim)

    os.makedirs(out_dir, exist_ok=True)
    nrow = max(1, int(math.sqrt(n_samples)))
    for i, tval in enumerate(ts):
        save_grid(traj[i], nrow=nrow, out_path=os.path.join(out_dir, f"time_{i:02d}_t{float(tval):.2f}.png"))
    for b in range(n_samples):
        save_strip(traj[:, b], out_path=os.path.join(out_dir, f"sample_{b:02d}_strip.png"))

    print(f"Saved SDE samples to: {out_dir} | cond_attrs={cond_attrs} | cond_bits={list(cond_bits)}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 training iteration to sanity-check shapes/APIs")
    parser.add_argument("--sample_ode", action="store_true")
    parser.add_argument("--sample_sde", action="store_true")
    parser.add_argument("--ckpt", type=str, default="models/celebA_sf2m/sf2m_celeba_64_cond_Sex.pth")

    parser.add_argument("--x_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--total_steps", type=int, default=100_000)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--times", type=float, nargs="*", default=None)
    parser.add_argument("--n_samples", type=int, default=8)

    # sampling conditions
    parser.add_argument("--sex", "--male", dest="sex", type=int, choices=[0, 1], default=1, help="sampling only: 0=female, 1=male")
    parser.add_argument("--cond_bits", type=int, nargs="*", default=None)

    # training conditions
    parser.add_argument("--cond_attrs", type=str, nargs="*", default=["Sex"])
    args = parser.parse_args()

    visualize_train_grid(x_dim=args.x_dim)

    # robust save_dir derivation
    save_dir = os.path.dirname(args.ckpt) or "models/celebA_sf2m"

    if args.train or args.dry_run:
        train_celebA_sf2m(
            x_dim=args.x_dim,
            batch_size=args.batch_size,
            total_steps=args.total_steps,
            sigma=args.sigma,
            ema_decay=args.ema_decay,
            lr=args.lr,
            seed=args.seed,
            num_workers=args.num_workers,
            save_dir=save_dir,
            cond_attrs=args.cond_attrs,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            return

    cond_bits = args.cond_bits if args.cond_bits is not None else [args.sex]

    if args.sample_ode:
        sample_ode_trajectory(args.ckpt, n_samples=args.n_samples, ts=args.times, out_dir="samples/ode", cond_bits=cond_bits)

    if args.sample_sde:
        sample_sde_trajectory(args.ckpt, n_samples=args.n_samples, ts=args.times, out_dir="samples/sde", cond_bits=cond_bits)


if __name__ == "__main__":
    main()
