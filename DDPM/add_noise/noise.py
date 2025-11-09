import os
from typing import Iterable, List, Optional, Sequence, Union

import torch
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm


def _make_beta_schedule(
    num_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
):
    """
    Linear beta schedule; returns (betas, alphas_cumprod).
    """
    betas = torch.linspace(beta_start, beta_end, num_steps, device=device, dtype=dtype)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas_cumprod


def _ensure_dir(path: str):
    if path is not None:
        os.makedirs(path, exist_ok=True)


def _to_tensor(img: Image.Image, grayscale: Optional[bool] = None):
    """
    Convert PIL image to float tensor in [0,1] shaped (C,H,W).
    Optionally force grayscale (1ch) or RGB (3ch).
    """
    if grayscale is True:
        img = img.convert("L")
    elif grayscale is False:
        img = img.convert("RGB")
    else:
        # keep image mode but normalize to L or RGB for consistency
        if img.mode not in ("L", "RGB"):
            # fallback: convert to RGB
            img = img.convert("RGB")
    return TF.to_tensor(img)  # [C,H,W], float32 in [0,1]


def _to_pil(t: torch.Tensor) -> Image.Image:
    """
    Convert tensor [C,H,W] in [0,1] to PIL Image.
    Clips to [0,1].
    """
    t = t.detach().cpu().clamp(0.0, 1.0)
    return TF.to_pil_image(t)


def q_sample(
    x0: torch.Tensor,
    t: Union[int, torch.Tensor],
    alphas_cumprod: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
):
    """
    Closed-form forward diffusion sample at time index t.

    Args:
        x0: [B,C,H,W] clean image in [0,1].
        t: scalar int or [B] tensor of timestep indices in [0, T-1].
        alphas_cumprod: [T] tensor of ᾱ_t.
        noise: optional noise tensor same shape as x0; default ~N(0,1).

    Returns:
        x_t tensor same shape as x0.
    """
    if noise is None:
        noise = torch.randn_like(x0)
    if isinstance(t, int):
        at = alphas_cumprod[t].view(1, 1, 1, 1).to(x0.device, x0.dtype)
    else:
        at = alphas_cumprod[t].view(-1, 1, 1, 1).to(x0.device, x0.dtype)
    return torch.sqrt(at) * x0 + torch.sqrt(1.0 - at) * noise


def add_diffusion_noise_to_png(
    png_path: str,
    num_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    resize: Optional[Sequence[int]] = None,  # (H,W)
    grayscale: Optional[bool] = None,
    outdir: Optional[str] = None,
    save_every: int = 0,  # 0 disables; 1 saves all 1000; N saves every Nth
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
    return_all: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Forward‑diffuse (add noise) to a PNG image over `num_steps` timesteps.

    The iterative update is:
        x_t = sqrt(1-β_t) * x_{t-1} + sqrt(β_t) * ε_t,
    which yields the same distribution as the closed‑form q_sample with ᾱ_t.

    Args:
        png_path: path to input PNG.
        num_steps: number of diffusion steps (default 1000).
        beta_start, beta_end: linear β schedule endpoints.
        resize: optional (H,W) to resize before diffusion.
        grayscale: force grayscale(True), force RGB(False), or keep(None).
        outdir: if given, intermediate frames saved here.
        save_every: see above.
        device, dtype: tensor placement and type.
        seed: optional RNG seed for reproducibility.
        return_all: if True return list [x_0,...,x_T]; else final x_T.

    Returns:
        final noisy tensor [1,C,H,W] or list of tensors per step.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # --- load image ---
    img = Image.open(png_path)
    if resize is not None:
        img = img.resize(resize[::-1] if len(resize) == 2 else resize)  # PIL wants (W,H)
    x0_chw = _to_tensor(img, grayscale=grayscale)  # [C,H,W]
    x = x0_chw.unsqueeze(0).to(device=device, dtype=dtype)  # [1,C,H,W]

    # --- schedule ---
    betas, alphas_cumprod = _make_beta_schedule(
        num_steps=num_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        device=x.device,
        dtype=x.dtype,
    )

    # --- iterative diffusion ---
    frames: List[torch.Tensor] = [x.clone()] if return_all or save_every == 1 else []
    _ensure_dir(outdir)
    if outdir is not None and save_every == 1:
        _to_pil(x0_chw).save(os.path.join(outdir, f"step0000.png"))

    xt = x
    for t in tqdm(range(num_steps), desc="Diffusing", leave=False):
        eps = torch.randn_like(xt)
        sqrt_one_minus_beta = torch.sqrt(1.0 - betas[t])
        sqrt_beta = torch.sqrt(betas[t])
        xt = sqrt_one_minus_beta * xt + sqrt_beta * eps  # [B,C,H,W]

        # save / collect?
        if save_every and ((t + 1) % save_every == 0 or (t + 1) == num_steps):
            if outdir is not None:
                _to_pil(xt.squeeze(0)).save(
                    os.path.join(outdir, f"step{t+1:04d}.png")
                )
            if return_all:
                frames.append(xt.clone())

    if return_all:
        return frames
    return xt

if __name__ == "__main__":
    noisy_seq = add_diffusion_noise_to_png(
        "Cat.png",
        outdir="noised_frames",
        save_every=100,   # change to 1 to save all 1000 frames
        return_all=True,
        seed=42,
    )
    final = noisy_seq[-1]  # [1,C,H,W]
    print("Final noisy shape:", final.shape)

