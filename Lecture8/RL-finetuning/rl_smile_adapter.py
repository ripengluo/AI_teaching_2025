#!/usr/bin/env python3

import argparse
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torchdiffeq
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid


ROOT = Path(__file__).resolve().parents[1]
BASE_REPO = ROOT / "celebA_conditioned"
SURROGATE_REPO = ROOT / "celebA_surrogate"
sys.path.insert(0, str(BASE_REPO))
sys.path.insert(0, str(SURROGATE_REPO))

from celeba_surrogate_cnn import load_surrogate  # noqa: E402
from torchcfm.conditional_flow_matching import SchrodingerBridgeConditionalFlowMatcher  # noqa: E402
from torchcfm.models.unet.unet import UNetModelWrapper  # noqa: E402


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic, warn_only=True)


def get_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_ckpt_path(ckpt_path: str) -> str:
    if os.path.exists(ckpt_path):
        return ckpt_path
    legacy_ckpt_path = ckpt_path.replace("_cond_Sex", "_cond_Male")
    if legacy_ckpt_path != ckpt_path and os.path.exists(legacy_ckpt_path):
        return legacy_ckpt_path
    return ckpt_path


def build_unet(
    x_dim: int,
    num_classes: int | None,
    prop_dim: int = 1,
    prop_adapter_rank: int = 16,
) -> UNetModelWrapper:
    return UNetModelWrapper(
        dim=(3, x_dim, x_dim),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
        num_classes=(num_classes or 1000),
        class_cond=(num_classes is not None),
        prop_dim=prop_dim,
        prop_adapter_rank=prop_adapter_rank,
    )


def freeze_except_adapters(model: torch.nn.Module) -> int:
    trainable = 0
    for name, param in model.named_parameters():
        param.requires_grad = ("prop_encoder" in name) or ("prop_adapter" in name)
        if param.requires_grad:
            trainable += param.numel()
    return trainable


def sample_base_labels(
    batch_size: int,
    num_classes: int | None,
    device: torch.device,
    fixed_label: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor | None:
    if num_classes is None:
        return None
    if fixed_label is not None:
        return torch.full((batch_size,), int(fixed_label), device=device, dtype=torch.long)
    return torch.randint(0, int(num_classes), (batch_size,), generator=generator).to(device=device)


def make_smile_target(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    bits = torch.ones(batch_size, device=device, dtype=torch.long)
    return bits, bits.float().unsqueeze(1)


class OracleSmilePool:
    def __init__(
        self,
        x_dim: int,
        data_root: str,
        split: str,
        num_classes: int | None,
        fixed_label: int | None,
        seed: int,
    ) -> None:
        self.transform = transforms.Compose(
            [
                transforms.Resize((x_dim, x_dim)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * 3, (0.5,) * 3),
            ]
        )
        self.dataset = datasets.CelebA(
            root=data_root,
            split=split,
            target_type="attr",
            download=True,
            transform=None,
        )
        attr_names = list(self.dataset.attr_names)
        if "Smiling" not in attr_names:
            raise ValueError("CelebA attrs do not include Smiling.")

        smile_idx = attr_names.index("Smiling")
        self.male_idx = attr_names.index("Male")
        self.num_classes = num_classes
        keep_mask = self.dataset.attr[:, smile_idx] > 0
        if num_classes is not None and fixed_label is not None:
            male_bits = (self.dataset.attr[:, self.male_idx] > 0).long()
            keep_mask = keep_mask & (male_bits == int(fixed_label))

        self.keep_idx = torch.where(keep_mask)[0].long()
        if self.keep_idx.numel() == 0:
            raise ValueError("Oracle loader found no matching smiling samples for the requested filter.")

        self.generator = torch.Generator().manual_seed(int(seed))
        self.order = torch.empty(0, dtype=torch.long)
        self.cursor = 0

    def _refresh_order(self) -> None:
        perm = torch.randperm(self.keep_idx.numel(), generator=self.generator)
        self.order = self.keep_idx[perm].clone()
        self.cursor = 0

    def next_batch(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("OracleSmilePool.next_batch requires batch_size > 0.")

        images = []
        attrs = []
        while len(images) < batch_size:
            if self.cursor >= self.order.numel():
                self._refresh_order()

            data_idx = int(self.order[self.cursor].item())
            self.cursor += 1
            image, attr = self.dataset[data_idx]
            image = self.transform(image)
            if bool(torch.rand((), generator=self.generator).item() < 0.5):
                image = torch.flip(image, dims=(-1,))
            images.append(image)
            attrs.append(attr if torch.is_tensor(attr) else torch.as_tensor(attr))

        x = torch.stack(images, dim=0).to(device=device, non_blocking=True)
        attr_tensor = torch.stack(attrs, dim=0).to(device=device, non_blocking=True)
        labels = None
        if self.num_classes is not None:
            labels = (attr_tensor[:, self.male_idx] > 0).long()
        prop_cond = torch.ones((x.shape[0], 1), device=device)
        return x, labels, prop_cond

    def state_dict(self) -> dict:
        return {
            "generator_state": self.generator.get_state(),
            "order": self.order.clone(),
            "cursor": int(self.cursor),
        }

    def load_state_dict(self, state_dict: dict | None) -> None:
        if not state_dict:
            return
        self.generator.set_state(state_dict["generator_state"].cpu())
        self.order = state_dict["order"].cpu().clone()
        self.cursor = int(state_dict["cursor"])


def compute_oracle_batch_size(samples_per_step: int, topk_ratio: float, oracle_mix_ratio: float) -> int:
    if oracle_mix_ratio <= 0.0:
        return 0
    if oracle_mix_ratio >= 1.0:
        raise ValueError("oracle_mix_ratio must be in [0, 1).")
    gen_keep = max(1, math.ceil(samples_per_step * topk_ratio))
    return max(1, math.ceil(gen_keep * oracle_mix_ratio / (1.0 - oracle_mix_ratio)))


def get_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in state["torch_cuda"]])


def atomic_torch_save(obj: dict, path: str) -> None:
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


@torch.no_grad()
def sample_ode_images(
    drift: UNetModelWrapper,
    x_dim: int,
    labels: torch.Tensor | None,
    prop_cond: torch.Tensor | None,
    ode_steps: int,
    atol: float,
    rtol: float,
    y0: torch.Tensor | None = None,
) -> torch.Tensor:
    if prop_cond is not None:
        batch_size = prop_cond.shape[0]
        device = prop_cond.device
    elif labels is not None:
        batch_size = labels.shape[0]
        device = labels.device
    else:
        raise ValueError("sample_ode_images needs either labels or prop_cond to infer batch/device.")
    if y0 is None:
        y0 = torch.randn(batch_size, 3, x_dim, x_dim, device=device)
    else:
        y0 = y0.to(device=device)
    ts = torch.linspace(0.0, 1.0, max(2, ode_steps), device=device)

    def ode_fn(t, y):
        return drift(t, y, y=labels, prop_cond=prop_cond)

    traj = torchdiffeq.odeint(ode_fn, y0, ts, atol=atol, rtol=rtol, method="dopri5")
    return traj[-1].clamp(-1.0, 1.0)


@torch.no_grad()
def surrogate_reward(
    surrogate: torch.nn.Module,
    smile_idx: int,
    images: torch.Tensor,
    smile_bits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = surrogate(images)[:, smile_idx]
    smile_prob = torch.sigmoid(logits)
    reward = torch.where(smile_bits > 0, smile_prob, 1.0 - smile_prob)
    return reward, smile_prob


def topk_by_reward(
    images: torch.Tensor,
    labels: torch.Tensor | None,
    prop_cond: torch.Tensor | None,
    rewards: torch.Tensor,
    topk_ratio: float,
 ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    k = max(1, math.ceil(len(rewards) * topk_ratio))
    keep_idx = rewards.topk(k).indices
    return (
        images[keep_idx],
        (labels[keep_idx] if labels is not None else None),
        (prop_cond[keep_idx] if prop_cond is not None else None),
        rewards[keep_idx],
    )


def save_grid(path: str, images: torch.Tensor, nrow: int | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imgs01 = (images.detach().cpu().clamp(-1, 1) + 1.0) * 0.5
    nrow = nrow or max(1, int(math.sqrt(len(imgs01))))
    grid = make_grid(imgs01, nrow=nrow, padding=2)
    to_pil_image(grid).save(path)


def save_comparison_grid(path: str, left_images: torch.Tensor, right_images: torch.Tensor) -> None:
    if left_images.shape != right_images.shape:
        raise ValueError(
            f"Left/right preview shapes must match, got {tuple(left_images.shape)} and {tuple(right_images.shape)}"
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    left01 = (left_images.detach().cpu().clamp(-1, 1) + 1.0) * 0.5
    right01 = (right_images.detach().cpu().clamp(-1, 1) + 1.0) * 0.5
    paired = torch.stack([left01, right01], dim=1).flatten(0, 1)
    grid = make_grid(paired, nrow=2, padding=2)
    to_pil_image(grid).save(path)


def load_base_pair(
    ckpt_path: str,
    device: torch.device,
    prop_adapter_rank: int,
) -> tuple[dict, UNetModelWrapper, UNetModelWrapper | None, UNetModelWrapper, UNetModelWrapper | None]:
    ckpt_path = resolve_ckpt_path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    x_dim = int(ckpt.get("x_dim", 64))
    num_classes = ckpt.get("num_classes", None)
    num_classes = int(num_classes) if num_classes is not None else None
    if num_classes is not None and num_classes <= 0:
        num_classes = None

    agent_drift = build_unet(x_dim, num_classes, prop_adapter_rank=prop_adapter_rank).to(device)
    prior_drift = build_unet(x_dim, num_classes, prop_adapter_rank=prop_adapter_rank).to(device)
    drift_sd = ckpt.get("ema_drift", ckpt["drift"])
    agent_drift.load_state_dict(drift_sd, strict=False)
    prior_drift.load_state_dict(drift_sd, strict=False)

    agent_score = None
    prior_score = None
    if "score" in ckpt:
        agent_score = build_unet(x_dim, num_classes, prop_adapter_rank=prop_adapter_rank).to(device)
        prior_score = build_unet(x_dim, num_classes, prop_adapter_rank=prop_adapter_rank).to(device)
        agent_score.load_state_dict(ckpt["score"], strict=False)
        prior_score.load_state_dict(ckpt["score"], strict=False)

    return ckpt, agent_drift, agent_score, prior_drift, prior_score


def load_agent_pair(
    ckpt_path: str,
    device: torch.device,
) -> tuple[dict, UNetModelWrapper, UNetModelWrapper | None]:
    ckpt = torch.load(ckpt_path, map_location=device)
    x_dim = int(ckpt["x_dim"])
    num_classes = ckpt.get("num_classes", None)
    num_classes = int(num_classes) if num_classes is not None else None
    if num_classes is not None and num_classes <= 0:
        num_classes = None
    prop_adapter_rank = int(ckpt.get("prop_adapter_rank", 16))

    drift = build_unet(x_dim, num_classes, prop_adapter_rank=prop_adapter_rank).to(device)
    drift.load_state_dict(ckpt["drift"], strict=False)

    score = None
    if ckpt.get("score") is not None:
        score = build_unet(x_dim, num_classes, prop_adapter_rank=prop_adapter_rank).to(device)
        score.load_state_dict(ckpt["score"], strict=False)

    return ckpt, drift, score


def resume_ckpt_path(args: argparse.Namespace) -> str:
    if args.resume_ckpt:
        return args.resume_ckpt
    return os.path.join(args.out_dir, "sf2m_smile_adapter_rl_resume.pth")


def validate_resume_args(args: argparse.Namespace, saved_args: dict) -> None:
    keys = [
        "base_ckpt",
        "surrogate_ckpt",
        "seed",
        "deterministic",
        "prop_adapter_rank",
        "fixed_label",
        "samples_per_step",
        "topk_ratio",
        "ft_epochs",
        "lr",
        "kl_coef",
        "oracle_mix_ratio",
        "oracle_data_root",
        "oracle_split",
        "ode_steps",
        "atol",
        "rtol",
    ]
    mismatches = []
    for key in keys:
        current = getattr(args, key)
        saved = saved_args.get(key)
        if current != saved:
            mismatches.append(f"{key}: current={current!r}, saved={saved!r}")
    if mismatches:
        mismatch_text = "\n".join(mismatches)
        raise ValueError(f"Resume args do not match saved training state:\n{mismatch_text}")


def save_resume_checkpoint(
    path: str,
    args: argparse.Namespace,
    step: int,
    best_reward: float,
    best_step: int,
    ckpt: dict,
    drift: UNetModelWrapper,
    score: UNetModelWrapper | None,
    optimizer: torch.optim.Optimizer,
    x_dim: int,
    sigma: float,
    num_classes: int | None,
    oracle_pool: OracleSmilePool | None,
) -> None:
    atomic_torch_save(
        {
            "step": int(step),
            "best_reward": float(best_reward),
            "best_step": int(best_step),
            "base_ckpt": resolve_ckpt_path(args.base_ckpt),
            "drift": drift.state_dict(),
            "score": (score.state_dict() if score is not None else None),
            "optimizer": optimizer.state_dict(),
            "rng_state": get_rng_state(),
            "oracle_state": (oracle_pool.state_dict() if oracle_pool is not None else None),
            "x_dim": x_dim,
            "sigma": sigma,
            "num_classes": num_classes,
            "cond_attrs": ckpt.get("cond_attrs", None),
            "prop_dim": 1,
            "prop_adapter_rank": args.prop_adapter_rank,
            "prop_name": "Smiling",
            "surrogate_ckpt": args.surrogate_ckpt,
            "train_args": vars(args).copy(),
        },
        path,
    )


def load_resume_checkpoint(
    args: argparse.Namespace,
    device: torch.device,
    drift: UNetModelWrapper,
    score: UNetModelWrapper | None,
    optimizer: torch.optim.Optimizer,
    oracle_pool: OracleSmilePool | None,
) -> dict:
    path = resume_ckpt_path(args)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    validate_resume_args(args, state.get("train_args", {}))
    drift.load_state_dict(state["drift"], strict=False)
    if score is not None and state.get("score") is not None:
        score.load_state_dict(state["score"], strict=False)
    optimizer.load_state_dict(state["optimizer"])
    if oracle_pool is not None:
        oracle_pool.load_state_dict(state.get("oracle_state"))
    set_rng_state(state.get("rng_state"))
    return state


def compute_model_losses(
    drift: UNetModelWrapper,
    prior_drift: UNetModelWrapper,
    score: UNetModelWrapper | None,
    prior_score: UNetModelWrapper | None,
    fm: SchrodingerBridgeConditionalFlowMatcher,
    x1: torch.Tensor,
    labels: torch.Tensor | None,
    prop_cond: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x0 = torch.randn_like(x1)
    if labels is not None:
        t, xt, ut, _, y1, eps = fm.guided_sample_location_and_conditional_flow(
            x0, x1, y1=labels, return_noise=True
        )
    else:
        t, xt, ut, eps = fm.sample_location_and_conditional_flow(x0, x1, return_noise=True)
        y1 = None

    vt = drift(t, xt, y=y1, prop_cond=prop_cond)
    with torch.no_grad():
        vt_prior = prior_drift(t, xt, y=y1, prop_cond=prop_cond)
    main_loss = torch.pow(vt - ut, 2).mean(dim=(1, 2, 3))
    kl_loss = torch.pow(vt - vt_prior, 2).mean(dim=(1, 2, 3))

    if score is not None and prior_score is not None:
        st = score(t, xt, y=y1, prop_cond=prop_cond)
        with torch.no_grad():
            st_prior = prior_score(t, xt, y=y1, prop_cond=prop_cond)
        lambda_t = fm.compute_lambda(t).to(x1.device)
        score_loss = torch.pow(lambda_t[:, None, None, None] * st + eps, 2).mean(dim=(1, 2, 3))
        kl_score = torch.pow(st - st_prior, 2).mean(dim=(1, 2, 3))
        main_loss = main_loss + score_loss
        kl_loss = kl_loss + kl_score

    return main_loss, kl_loss


def finetune_batch(
    drift: UNetModelWrapper,
    prior_drift: UNetModelWrapper,
    score: UNetModelWrapper | None,
    prior_score: UNetModelWrapper | None,
    optimizer: torch.optim.Optimizer,
    fm: SchrodingerBridgeConditionalFlowMatcher,
    x1: torch.Tensor,
    labels: torch.Tensor | None,
    prop_cond: torch.Tensor,
    rewards: torch.Tensor,
    oracle_x1: torch.Tensor | None,
    oracle_labels: torch.Tensor | None,
    oracle_prop_cond: torch.Tensor | None,
    ft_epochs: int,
    kl_coef: float,
) -> dict[str, float]:
    drift.train()
    if score is not None:
        score.train()

    reward_min = rewards.min()
    reward_max = rewards.max()
    adv = (rewards - reward_min) / (reward_max - reward_min + 1e-6)
    adv = adv + 0.1
    kl_gate = 1.1 - rewards.clamp(0.0, 1.0)

    total_loss = 0.0
    total_main = 0.0
    total_kl = 0.0
    total_gen = 0.0
    total_oracle = 0.0
    mix_ratio = 0.0
    if oracle_x1 is not None:
        mix_ratio = oracle_x1.shape[0] / float(oracle_x1.shape[0] + x1.shape[0])

    for _ in range(ft_epochs):
        optimizer.zero_grad(set_to_none=True)

        gen_main_loss, gen_kl_loss = compute_model_losses(
            drift=drift,
            prior_drift=prior_drift,
            score=score,
            prior_score=prior_score,
            fm=fm,
            x1=x1,
            labels=labels,
            prop_cond=prop_cond,
        )
        gen_loss = (adv * gen_main_loss + kl_coef * kl_gate * gen_kl_loss).mean()

        oracle_loss = None
        oracle_main_loss = None
        oracle_kl_loss = None
        if oracle_x1 is not None and oracle_prop_cond is not None:
            oracle_main_loss, oracle_kl_loss = compute_model_losses(
                drift=drift,
                prior_drift=prior_drift,
                score=score,
                prior_score=prior_score,
                fm=fm,
                x1=oracle_x1,
                labels=oracle_labels,
                prop_cond=oracle_prop_cond,
            )
            oracle_loss = oracle_main_loss.mean() + kl_coef * oracle_kl_loss.mean()
            loss = (1.0 - mix_ratio) * gen_loss + mix_ratio * oracle_loss
        else:
            loss = gen_loss

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_gen += float(gen_loss.item())
        if oracle_loss is not None:
            total_oracle += float(oracle_loss.item())
            total_main += float(
                ((1.0 - mix_ratio) * gen_main_loss.mean() + mix_ratio * oracle_main_loss.mean()).item()
            )
            total_kl += float(
                ((1.0 - mix_ratio) * gen_kl_loss.mean() + mix_ratio * oracle_kl_loss.mean()).item()
            )
        else:
            total_main += float(gen_main_loss.mean().item())
            total_kl += float(gen_kl_loss.mean().item())

    stats = {
        "loss": total_loss / ft_epochs,
        "main": total_main / ft_epochs,
        "kl": total_kl / ft_epochs,
        "gen_loss": total_gen / ft_epochs,
    }
    if oracle_x1 is not None:
        stats["oracle_loss"] = total_oracle / ft_epochs
        stats["oracle_mix"] = mix_ratio
    return stats


@torch.no_grad()
def save_previews(
    drift: UNetModelWrapper,
    prior_drift: UNetModelWrapper,
    out_dir: str,
    x_dim: int,
    num_classes: int | None,
    fixed_label: int | None,
    step: int,
    n_samples: int,
    ode_steps: int,
    atol: float,
    rtol: float,
    device: torch.device,
    seed: int,
) -> None:
    drift.eval()
    prior_drift.eval()
    preview_gen = torch.Generator().manual_seed(int(seed) + int(step))
    label0 = sample_base_labels(
        n_samples,
        num_classes,
        device,
        fixed_label=fixed_label,
        generator=preview_gen,
    )
    y0 = torch.randn(n_samples, 3, x_dim, x_dim, generator=preview_gen).to(device=device)
    ref_imgs = sample_ode_images(
        prior_drift,
        x_dim,
        label0,
        None,
        ode_steps,
        atol,
        rtol,
        y0=y0,
    )
    smile_imgs = sample_ode_images(
        drift,
        x_dim,
        label0,
        torch.ones(n_samples, 1, device=device),
        ode_steps,
        atol,
        rtol,
        y0=y0,
    )
    save_comparison_grid(
        os.path.join(out_dir, f"step_{step:04d}_comparison.png"),
        ref_imgs,
        smile_imgs,
    )


def train(args) -> None:
    set_seed(args.seed, deterministic=args.deterministic)
    device = get_device(args.device)
    ckpt, drift, score, prior_drift, prior_score = load_base_pair(
        args.base_ckpt,
        device=device,
        prop_adapter_rank=args.prop_adapter_rank,
    )

    for model in (prior_drift, prior_score):
        if model is not None:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

    trainable = freeze_except_adapters(drift)
    if score is not None:
        trainable += freeze_except_adapters(score)

    params = [p for p in list(drift.parameters()) + list(score.parameters() if score is not None else []) if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    surrogate, meta = load_surrogate(args.surrogate_ckpt, device=device)
    if "Smiling" not in meta.attrs:
        raise ValueError(f"Surrogate attrs do not include Smiling: {meta.attrs}")
    smile_idx = meta.attrs.index("Smiling")

    x_dim = int(ckpt.get("x_dim", 64))
    sigma = float(ckpt.get("sigma", 0.1))
    num_classes = ckpt.get("num_classes", None)
    num_classes = int(num_classes) if num_classes is not None else None
    if num_classes is not None and num_classes <= 0:
        num_classes = None
    if int(meta.x_dim) != x_dim:
        raise ValueError(f"Surrogate x_dim={meta.x_dim} does not match SF2M x_dim={x_dim}")
    fm = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)
    oracle_pool = None
    oracle_batch_size = compute_oracle_batch_size(
        samples_per_step=args.samples_per_step,
        topk_ratio=args.topk_ratio,
        oracle_mix_ratio=args.oracle_mix_ratio,
    )
    if args.oracle_mix_ratio > 0.0:
        oracle_pool = OracleSmilePool(
            x_dim=x_dim,
            data_root=args.oracle_data_root,
            split=args.oracle_split,
            num_classes=num_classes,
            fixed_label=args.fixed_label,
            seed=args.seed + 17,
        )

    os.makedirs(args.out_dir, exist_ok=True)
    preview_dir = os.path.join(args.out_dir, "previews")
    best_ckpt = os.path.join(args.out_dir, "sf2m_smile_adapter_rl_best.pth")
    state_ckpt = resume_ckpt_path(args)
    best_reward = float("-inf")
    best_step = 0
    start_step = 1

    if args.resume:
        state = load_resume_checkpoint(
            args=args,
            device=device,
            drift=drift,
            score=score,
            optimizer=optimizer,
            oracle_pool=oracle_pool,
        )
        start_step = int(state["step"]) + 1
        best_reward = float(state.get("best_reward", float("-inf")))
        best_step = int(state.get("best_step", 0))
        print(
            f"[resume] loaded {state_ckpt} | step={state['step']} | best_reward={best_reward:.4f} | "
            f"next_step={start_step}"
        )
    else:
        save_previews(
            drift,
            prior_drift,
            preview_dir,
            x_dim,
            num_classes,
            args.fixed_label,
            0,
            args.preview_samples,
            args.ode_steps,
            args.atol,
            args.rtol,
            device,
            args.seed,
        )
        save_resume_checkpoint(
            path=state_ckpt,
            args=args,
            step=0,
            best_reward=best_reward,
            best_step=best_step,
            ckpt=ckpt,
            drift=drift,
            score=score,
            optimizer=optimizer,
            x_dim=x_dim,
            sigma=sigma,
            num_classes=num_classes,
            oracle_pool=oracle_pool,
        )

    print(
        f"[init] device={device} | trainable_params={trainable} | num_classes={num_classes} | sigma={sigma} | "
        f"oracle_mix_ratio={args.oracle_mix_ratio:.2f} | oracle_batch_size={oracle_batch_size} | "
        f"resume={'yes' if args.resume else 'no'}"
    )

    for step in range(start_step, args.rl_steps + 1):
        drift.eval()
        if score is not None:
            score.eval()

        labels = sample_base_labels(args.samples_per_step, num_classes, device, fixed_label=args.fixed_label)
        smile_bits, prop_cond = make_smile_target(args.samples_per_step, device)
        x1 = sample_ode_images(drift, x_dim, labels, prop_cond, args.ode_steps, args.atol, args.rtol)
        rewards, smile_prob = surrogate_reward(surrogate, smile_idx, x1, smile_bits)

        x_keep, y_keep, prop_keep, reward_keep = topk_by_reward(
            x1,
            labels,
            prop_cond,
            rewards,
            topk_ratio=args.topk_ratio,
        )
        oracle_x = None
        oracle_y = None
        oracle_prop = None
        if oracle_pool is not None:
            oracle_x, oracle_y, oracle_prop = oracle_pool.next_batch(
                batch_size=oracle_batch_size,
                device=device,
            )
        stats = finetune_batch(
            drift=drift,
            prior_drift=prior_drift,
            score=score,
            prior_score=prior_score,
            optimizer=optimizer,
            fm=fm,
            x1=x_keep.detach(),
            labels=(y_keep.detach() if y_keep is not None else None),
            prop_cond=prop_keep.detach(),
            rewards=reward_keep.detach(),
            oracle_x1=oracle_x,
            oracle_labels=oracle_y,
            oracle_prop_cond=oracle_prop,
            ft_epochs=args.ft_epochs,
            kl_coef=args.kl_coef,
        )

        reward1 = rewards.mean().item()
        prob1 = smile_prob.mean().item()
        log_line = (
            f"[step {step:03d}/{args.rl_steps:03d}] "
            f"reward_mean={reward1:.4f} "
            f"| reward(smile=1)={reward1:.4f} "
            f"| p_smile(target1)={prob1:.4f} "
            f"| loss={stats['loss']:.4f} main={stats['main']:.4f} kl={stats['kl']:.4f}"
        )
        if "oracle_loss" in stats:
            log_line += (
                f" | gen_loss={stats['gen_loss']:.4f} "
                f"| oracle_loss={stats['oracle_loss']:.4f} "
                f"| oracle_mix={stats['oracle_mix']:.2f}"
            )
        print(log_line)

        if reward1 > best_reward:
            best_reward = reward1
            best_step = step
            atomic_torch_save(
                {
                    "base_ckpt": resolve_ckpt_path(args.base_ckpt),
                    "drift": drift.state_dict(),
                    "score": (score.state_dict() if score is not None else None),
                    "x_dim": x_dim,
                    "sigma": sigma,
                    "num_classes": num_classes,
                    "cond_attrs": ckpt.get("cond_attrs", None),
                    "prop_dim": 1,
                    "prop_adapter_rank": args.prop_adapter_rank,
                    "prop_name": "Smiling",
                    "surrogate_ckpt": args.surrogate_ckpt,
                    "best_reward": best_reward,
                    "best_step": best_step,
                    "train_args": vars(args),
                },
                best_ckpt,
            )

        if step % args.preview_every == 0 or step == args.rl_steps:
            save_previews(
                drift,
                prior_drift,
                preview_dir,
                x_dim,
                num_classes,
                args.fixed_label,
                step,
                args.preview_samples,
                args.ode_steps,
                args.atol,
                args.rtol,
                device,
                args.seed,
            )

        if step % args.resume_every == 0 or step == args.rl_steps:
            save_resume_checkpoint(
                path=state_ckpt,
                args=args,
                step=step,
                best_reward=best_reward,
                best_step=best_step,
                ckpt=ckpt,
                drift=drift,
                score=score,
                optimizer=optimizer,
                x_dim=x_dim,
                sigma=sigma,
                num_classes=num_classes,
                oracle_pool=oracle_pool,
            )

    out_ckpt = os.path.join(args.out_dir, "sf2m_smile_adapter_rl.pth")
    atomic_torch_save(
        {
            "base_ckpt": resolve_ckpt_path(args.base_ckpt),
            "drift": drift.state_dict(),
            "score": (score.state_dict() if score is not None else None),
            "x_dim": x_dim,
            "sigma": sigma,
            "num_classes": num_classes,
            "cond_attrs": ckpt.get("cond_attrs", None),
            "prop_dim": 1,
            "prop_adapter_rank": args.prop_adapter_rank,
            "prop_name": "Smiling",
            "surrogate_ckpt": args.surrogate_ckpt,
            "best_reward": best_reward,
            "best_step": best_step,
            "train_args": vars(args),
        },
        out_ckpt,
    )
    print(f"[done] saved {out_ckpt}")


def sample_only(args) -> None:
    device = get_device(args.device)
    ckpt, drift, _score = load_agent_pair(args.ckpt, device=device)
    base_ckpt = ckpt.get("base_ckpt")
    if not base_ckpt:
        raise ValueError("The RL checkpoint does not record base_ckpt for reference previews.")
    _, prior_drift, _, _, _ = load_base_pair(
        base_ckpt,
        device=device,
        prop_adapter_rank=int(ckpt.get("prop_adapter_rank", 16)),
    )
    x_dim = int(ckpt["x_dim"])
    num_classes = ckpt.get("num_classes", None)
    num_classes = int(num_classes) if num_classes is not None else None
    if num_classes is not None and num_classes <= 0:
        num_classes = None
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.ckpt), "sample_only")
    os.makedirs(out_dir, exist_ok=True)
    save_previews(
        drift,
        prior_drift,
        out_dir,
        x_dim,
        num_classes,
        args.fixed_label,
        step=0,
        n_samples=args.preview_samples,
        ode_steps=args.ode_steps,
        atol=args.atol,
        rtol=args.rtol,
        device=device,
        seed=args.seed,
    )
    print(f"[done] saved previews to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--base_ckpt", type=str, default=str(BASE_REPO / "models/celebA_sf2m/sf2m_celeba_64_cond_Sex.pth"))
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--resume_ckpt", type=str, default="")
    parser.add_argument(
        "--surrogate_ckpt",
        type=str,
        default=str(SURROGATE_REPO / "checkpoints/surrogate_celeba_64_attrs_Wavy_Hair-Smiling.pth"),
    )
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent / "outputs/smile_adapter"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--prop_adapter_rank", type=int, default=16)
    parser.add_argument("--fixed_label", type=int, default=None)
    parser.add_argument("--samples_per_step", type=int, default=32)
    parser.add_argument("--topk_ratio", type=float, default=0.5)
    parser.add_argument("--rl_steps", type=int, default=1000)
    parser.add_argument("--ft_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl_coef", type=float, default=0.025)
    parser.add_argument("--oracle_mix_ratio", type=float, default=0.0)
    parser.add_argument("--oracle_data_root", type=str, default=str(BASE_REPO / "data"))
    parser.add_argument("--oracle_split", type=str, default="train")
    parser.add_argument("--oracle_num_workers", type=int, default=4)
    parser.add_argument("--resume_every", type=int, default=1)

    parser.add_argument("--ode_steps", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--preview_every", type=int, default=10)
    parser.add_argument("--preview_samples", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume_every <= 0:
        raise ValueError("--resume_every must be >= 1")
    if args.sample_only:
        if not args.ckpt:
            raise ValueError("--sample_only requires --ckpt")
        sample_only(args)
        return
    train(args)


if __name__ == "__main__":
    main()
