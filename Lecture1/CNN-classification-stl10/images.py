import argparse
import csv
import math
import os
import random
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, models, transforms


CLASSES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]
MEAN = (0.4467, 0.4398, 0.4066)
STD = (0.2603, 0.2566, 0.2713)


@dataclass
class DistributedState:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an STL-10 classifier with a ResNet backbone and optional DDP."
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--checkpoint-path", default="stl10_resnet_classifier.pth")
    parser.add_argument("--history-path", default="stl10_history.csv")
    parser.add_argument("--loss-plot-path", default="stl10_loss_curve.png")
    parser.add_argument("--sample-grid-path", default="stl10_train_samples.png")
    parser.add_argument("--prediction-grid-path", default="stl10_predictions.png")
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", default="29500")
    parser.add_argument("--disable-auto-ddp", action="store_true")
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed() -> DistributedState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1

    if enabled:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return DistributedState(
        enabled=enabled,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def set_distributed_env(rank: int, world_size: int, master_addr: str, master_port: str):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def reduce_tensor(values: torch.Tensor, state: DistributedState) -> torch.Tensor:
    if state.enabled:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def effective_num_batches(total_batches: int, max_batches: int) -> int:
    if max_batches <= 0:
        return total_batches
    return min(total_batches, max_batches)


def ensure_parent(path: str):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(96, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), value="random"),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_transform, eval_transform


def maybe_download_dataset(data_dir: str, train_transform, eval_transform, state, download: bool):
    if not download:
        return

    if not state.enabled or state.is_main:
        datasets.STL10(root=data_dir, split="train", download=True, transform=train_transform)
        datasets.STL10(root=data_dir, split="test", download=True, transform=eval_transform)

    if state.enabled:
        dist.barrier()


def build_loaders(args, state: DistributedState):
    train_transform, eval_transform = build_transforms()
    maybe_download_dataset(args.data_dir, train_transform, eval_transform, state, args.download)

    train_dataset = datasets.STL10(
        root=args.data_dir,
        split="train",
        download=False,
        transform=train_transform,
    )
    test_dataset = datasets.STL10(
        root=args.data_dir,
        split="test",
        download=False,
        transform=eval_transform,
    )

    train_sampler = None
    if state.enabled:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=state.world_size,
            rank=state.rank,
            shuffle=True,
            seed=args.seed,
        )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": state.device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )

    test_loader = None
    if state.is_main:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

    return train_loader, test_loader, train_sampler


def build_model(model_name: str, num_classes: int, dropout: float) -> nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
    else:
        model = models.resnet34(weights=None)

    # A CIFAR-style stem is better behaved than the ImageNet 7x7/stride-2 stem on 96x96 images.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def build_scheduler(optimizer, total_steps: int, warmup_steps: int, min_lr_ratio: float):
    total_steps = max(1, total_steps)
    warmup_steps = min(max(0, warmup_steps), total_steps - 1)

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_sample_grid(images: torch.Tensor, labels: torch.Tensor, path: str):
    if not path:
        return
    ensure_parent(path)
    images = denormalize(images[:6].cpu())
    labels = labels[:6].cpu()
    fig = plt.figure(figsize=(12, 6))
    for idx in range(6):
        axis = fig.add_subplot(2, 3, idx + 1)
        axis.imshow(images[idx].permute(1, 2, 0))
        axis.set_title(CLASSES[labels[idx].item()])
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_history(history, path: str):
    if not path:
        return
    ensure_parent(path)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for row in history:
            writer.writerow(row)


def save_loss_plot(history, path: str):
    if not path or not history:
        return

    ensure_parent(path)
    epochs = [row[0] for row in history]
    train_losses = [row[1] for row in history]
    val_losses = [row[3] for row in history]

    fig = plt.figure(figsize=(9, 5))
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(epochs, train_losses, label="Train Loss", linewidth=2.0)
    axis.plot(epochs, val_losses, label="Validation Loss", linewidth=2.0)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Training vs Validation Loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_prediction_grid(model: nn.Module, loader: DataLoader, device: torch.device, path: str):
    if not path:
        return

    ensure_parent(path)
    model.eval()
    images, labels = next(iter(loader))
    images = images[:12].to(device)
    labels = labels[:12].to(device)

    with torch.inference_mode():
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

    images = denormalize(images.cpu())
    labels = labels.cpu()
    predictions = predictions.cpu()

    fig = plt.figure(figsize=(12, 8))
    for idx in range(12):
        axis = fig.add_subplot(3, 4, idx + 1, xticks=[], yticks=[])
        axis.imshow(images[idx].permute(1, 2, 0))
        pred = CLASSES[predictions[idx].item()]
        truth = CLASSES[labels[idx].item()]
        title_color = "green" if predictions[idx].item() == labels[idx].item() else "red"
        axis.set_title(f"{pred} ({truth})", color=title_color)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def mixup_batch(inputs: torch.Tensor, labels: torch.Tensor, alpha: float):
    if alpha <= 0.0:
        return inputs, labels, labels, 1.0

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    lam = max(lam, 1.0 - lam)
    indices = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[indices]
    return mixed_inputs, labels, labels[indices], lam


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch,
    log_interval,
    max_batches,
    mixup_alpha,
    state,
):
    model.train()
    metrics = torch.zeros(3, device=device)
    total_batches = effective_num_batches(len(loader), max_batches)

    for step, (inputs, labels) in enumerate(loader, start=1):
        if step > total_batches:
            break

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mixed_inputs, labels_a, labels_b, lam = mixup_batch(inputs, labels, mixup_alpha)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
            outputs = model(mixed_inputs)
            loss = lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = labels.size(0)
        predictions = outputs.argmax(dim=1)
        metrics[0] += loss.detach() * batch_size
        metrics[1] += lam * (predictions == labels_a).sum() + (1.0 - lam) * (predictions == labels_b).sum()
        metrics[2] += batch_size

        if state.is_main and step % log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch:03d} Step {step:03d}/{total_batches:03d} "
                f"Loss: {loss.item():.4f} LR: {lr:.6f}"
            )

    metrics = reduce_tensor(metrics, state)
    train_loss = (metrics[0] / metrics[2]).item()
    train_acc = (100.0 * metrics[1] / metrics[2]).item()
    return train_loss, train_acc


def evaluate(model, loader, criterion, device, max_batches: int):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    total_batches = effective_num_batches(len(loader), max_batches)

    with torch.inference_mode():
        for step, (inputs, labels) in enumerate(loader, start=1):
            if step > total_batches:
                break

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return loss_sum / total, 100.0 * correct / total


def save_checkpoint(model, path: str, epoch: int, best_acc: float, args):
    ensure_parent(path)
    torch.save(
        {
            "epoch": epoch,
            "best_accuracy": best_acc,
            "model_name": args.model,
            "model_state_dict": unwrap_model(model).state_dict(),
        },
        path,
    )


def load_checkpoint(model: nn.Module, path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def run_training(args):
    state = setup_distributed()
    set_seed(args.seed + state.rank)

    if state.device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    try:
        train_loader, test_loader, train_sampler = build_loaders(args, state)

        if state.is_main:
            sample_images, sample_labels = next(iter(train_loader))
            save_sample_grid(sample_images, sample_labels, args.sample_grid_path)

        if state.enabled:
            dist.barrier()

        model = build_model(args.model, num_classes=len(CLASSES), dropout=args.dropout).to(state.device)
        if state.enabled and state.device.type == "cuda":
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if state.enabled:
            if state.device.type == "cuda":
                model = DDP(model, device_ids=[state.local_rank], output_device=state.local_rank)
            else:
                model = DDP(model)

        if state.is_main:
            print(unwrap_model(model))
            if state.enabled:
                print(f"Running DDP with world_size={state.world_size} on {state.device}.")
            else:
                print(f"Running single-process training on {state.device}.")

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        steps_per_epoch = effective_num_batches(len(train_loader), args.max_train_batches)
        scheduler = build_scheduler(
            optimizer=optimizer,
            total_steps=args.epochs * steps_per_epoch,
            warmup_steps=args.warmup_epochs * steps_per_epoch,
            min_lr_ratio=args.min_lr_ratio,
        )
        scaler = amp.GradScaler("cuda", enabled=state.device.type == "cuda")

        best_acc = -1.0
        history = []
        epochs_without_improvement = 0
        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=state.device,
                epoch=epoch,
                log_interval=args.log_interval,
                max_batches=args.max_train_batches,
                mixup_alpha=args.mixup_alpha,
                state=state,
            )

            if state.is_main:
                val_loss, val_acc = evaluate(
                    model=unwrap_model(model),
                    loader=test_loader,
                    criterion=criterion,
                    device=state.device,
                    max_batches=args.max_val_batches,
                )
                print(
                    f"Epoch {epoch:03d} Summary | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
                )
                history.append((epoch, train_loss, train_acc, val_loss, val_acc))
                save_history(history, args.history_path)

                if val_acc > best_acc:
                    best_acc = val_acc
                    save_checkpoint(model, args.checkpoint_path, epoch, best_acc, args)
                    print(f"Saved new best checkpoint to {args.checkpoint_path}")
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                stop_training = args.patience > 0 and epochs_without_improvement >= args.patience
                if stop_training:
                    print(
                        f"Early stopping at epoch {epoch:03d} after "
                        f"{epochs_without_improvement} epochs without validation improvement."
                    )
            else:
                stop_training = False

            if state.enabled:
                stop_tensor = torch.tensor([int(stop_training)], device=state.device)
                dist.broadcast(stop_tensor, src=0)
                dist.barrier()
                stop_training = bool(stop_tensor.item())

            if stop_training:
                break

        if state.is_main:
            best_model = unwrap_model(model)
            load_checkpoint(best_model, args.checkpoint_path, state.device)
            save_loss_plot(history, args.loss_plot_path)
            save_prediction_grid(best_model, test_loader, state.device, args.prediction_grid_path)
            print(f"Best Test Accuracy: {best_acc:.2f}%")

        if state.enabled:
            dist.barrier()
    finally:
        cleanup_distributed()


def spawned_worker(local_rank: int, world_size: int, args):
    set_distributed_env(
        rank=local_rank,
        world_size=world_size,
        master_addr=args.master_addr,
        master_port=args.master_port,
    )
    run_training(args)


def main():
    args = parse_args()
    visible_gpus = torch.cuda.device_count()
    has_distributed_env = "WORLD_SIZE" in os.environ or "RANK" in os.environ

    if visible_gpus > 1 and not has_distributed_env and not args.disable_auto_ddp:
        print(f"Auto-launching DDP across {visible_gpus} visible GPUs.")
        mp.spawn(spawned_worker, args=(visible_gpus, args), nprocs=visible_gpus, join=True)
        return

    run_training(args)


if __name__ == "__main__":
    main()
