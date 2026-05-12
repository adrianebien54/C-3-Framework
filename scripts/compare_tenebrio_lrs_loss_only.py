#!/usr/bin/env python3

"""Compare learning rates on Tenebrio using loss-only training metrics.

This script is intended for fast LR selection: it logs only train/val Euclidean
loss per epoch and skips extra metric computation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.SCC_Model.CSRNet import CSRNet


@dataclass(frozen=True)
class TenebrioSample:
    image_path: Path
    boxes: list[list[float]]
    width: int
    height: int


class TenebrioDensityDataset(Dataset):
    def __init__(self, image_dir: Path, annotation_file: Path):
        self.image_dir = image_dir
        self.samples = self._load_samples(annotation_file)
        self.to_tensor = transforms.ToTensor()

    def _load_samples(self, annotation_file: Path) -> list[TenebrioSample]:
        with annotation_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        image_index = {}
        for image_entry in data["images"]:
            image_index[image_entry["id"]] = TenebrioSample(
                image_path=self.image_dir / image_entry["file_name"],
                boxes=[],
                width=int(image_entry["width"]),
                height=int(image_entry["height"]),
            )

        grouped_boxes: dict[int, list[list[float]]] = defaultdict(list)
        for annotation in data["annotations"]:
            grouped_boxes[int(annotation["image_id"])].append(annotation["bbox"])

        samples: list[TenebrioSample] = []
        missing_files = 0
        for image_id, sample in image_index.items():
            if not sample.image_path.is_file():
                missing_files += 1
                continue
            samples.append(
                TenebrioSample(
                    image_path=sample.image_path,
                    boxes=grouped_boxes.get(image_id, []),
                    width=sample.width,
                    height=sample.height,
                )
            )

        samples.sort(key=lambda item: item.image_path.name)
        if not samples:
            raise RuntimeError(
                f"No matching images found in {self.image_dir} for annotation file {annotation_file}."
            )
        if missing_files > 0:
            print(
                f"[TenebrioDensityDataset] Skipped {missing_files} annotation entries with missing files in {self.image_dir}."
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        image_tensor = self.to_tensor(image)
        density = self._build_density_map(sample.width, sample.height, sample.boxes)
        density_tensor = torch.from_numpy(density).unsqueeze(0)
        return image_tensor, density_tensor

    @staticmethod
    def _build_density_map(width: int, height: int, boxes: list[list[float]]) -> np.ndarray:
        """Build density map using coordinate scaling approach (Li et al., 2018).
        
        Scales coordinates by 1/8 first, places impulses on small grid,
        then applies Gaussian blur with sigma=15/8. This avoids interpolation loss.
        """
        scale = 1.0 / 8.0
        small_height = int(np.ceil(height * scale))
        small_width = int(np.ceil(width * scale))
        density = np.zeros((small_height, small_width), dtype=np.float32)
        
        for bbox in boxes:
            x, y, w, h = bbox
            center_x = (x + w / 2.0) * scale
            center_y = (y + h / 2.0) * scale
            
            ix = int(round(center_x))
            iy = int(round(center_y))
            if 0 <= ix < small_width and 0 <= iy < small_height:
                density[iy, ix] += 1.0

        if density.sum() > 0:
            sigma_small = 15.0 / 8.0
            density = gaussian_filter(density, sigma=sigma_small, mode="constant")
        return density.astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Tenebrio CSRNet training with loss-only logging.")
    parser.add_argument("--train-image-dir", type=Path, default=Path("datasets/Tenebrio/train/img"))
    parser.add_argument("--val-image-dir", type=Path, default=Path("datasets/Tenebrio/val/img"))
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=Path("datasets/Tenebrio/TenebrioVision_Annotations.json"),
    )
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("exp/tenebrio_lr_compare_loss_only"))
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optional directory for saving model checkpoints after each epoch.",
    )
    parser.add_argument("--seed", type=int, default=3035)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("+", "")


def build_loaders(args: argparse.Namespace):
    train_dataset = TenebrioDensityDataset(args.train_image_dir, args.annotation_file)
    val_dataset = TenebrioDensityDataset(args.val_image_dir, args.annotation_file)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: int,
) -> float:
    model.train()
    running_loss = 0.0
    steps = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        targets = targets[:, :, : predictions.shape[2], : predictions.shape[3]]
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        steps += 1
        if max_steps > 0 and steps >= max_steps:
            break

    return running_loss / max(1, steps)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    max_steps: int,
) -> float:
    model.eval()
    running_loss = 0.0
    steps = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        predictions = model(images)
        targets = targets[:, :, : predictions.shape[2], : predictions.shape[3]]
        loss = criterion(predictions, targets)

        running_loss += float(loss.item())
        steps += 1
        if max_steps > 0 and steps >= max_steps:
            break

    return running_loss / max(1, steps)


def euclidean_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """CSRNet paper loss: (1 / (2N)) * sum_i ||pred_i - target_i||_2^2."""
    batch_size = predictions.shape[0]
    return F.mse_loss(predictions, targets, reduction="sum") / (2.0 * batch_size)


def save_history_csv(path: Path, history: list[tuple[int, float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        writer.writerows(history)


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, lr: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "lr": lr,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def save_plot(path: Path, per_lr_histories: dict[float, list[tuple[int, float, float]]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot generation.")
        return

    plt.figure(figsize=(10, 6))
    for lr, history in per_lr_histories.items():
        epochs = [row[0] for row in history]
        train_losses = [row[1] for row in history]
        val_losses = [row[2] for row in history]
        label_lr = format_lr(lr)
        plt.plot(epochs, train_losses, label=f"train lr={label_lr}", linestyle="-")
        plt.plot(epochs, val_losses, label=f"val lr={label_lr}", linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Loss (Euclidean)")
    plt.title("Tenebrio CSRNet: Train/Val Loss by Learning Rate")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        print("Warning: batch_size != 1 may OOM on full-resolution Tenebrio images.")

    for directory in [args.train_image_dir, args.val_image_dir]:
        if not directory.is_dir():
            raise FileNotFoundError(f"Image directory not found: {directory}")
    if not args.annotation_file.is_file():
        raise FileNotFoundError(f"Annotation file not found: {args.annotation_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for practical runtime at full Tenebrio resolution.")

    train_loader, val_loader = build_loaders(args)
    criterion = euclidean_loss

    all_histories: dict[float, list[tuple[int, float, float]]] = {}
    summary_rows: list[tuple[float, float, int]] = []

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")

    for lr in args.lrs:
        print(f"\n=== LR {lr:.3e} ===")
        model = CSRNet(load_weights=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        lr_checkpoint_dir = None
        if args.checkpoint_dir is not None:
            lr_checkpoint_dir = args.checkpoint_dir / f"lr_{format_lr(lr)}"
            lr_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val = float("inf")
        no_improve = 0
        history: list[tuple[int, float, float]] = []

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                args.max_train_steps,
            )
            val_loss = validate_one_epoch(
                model,
                val_loader,
                criterion,
                device,
                args.max_val_steps,
            )
            history.append((epoch, train_loss, val_loss))

            print(
                f"lr={lr:.3e} epoch={epoch:03d}/{args.epochs:03d} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
            else:
                no_improve += 1

            if args.patience > 0 and no_improve >= args.patience:
                print(
                    f"Early stopping at epoch {epoch} for lr={lr:.3e} "
                    f"(patience={args.patience})."
                )
                break

            if lr_checkpoint_dir is not None:
                save_checkpoint(
                    lr_checkpoint_dir / f"epoch_{epoch:03d}.pth",
                    model,
                    optimizer,
                    epoch,
                    lr,
                )

        all_histories[lr] = history
        summary_rows.append((lr, best_val, len(history)))

        if lr_checkpoint_dir is not None and history:
            final_epoch = history[-1][0]
            save_checkpoint(
                lr_checkpoint_dir / "last.pth",
                model,
                optimizer,
                final_epoch,
                lr,
            )

        csv_path = args.output_dir / f"lr_{format_lr(lr)}.csv"
        save_history_csv(csv_path, history)
        print(f"Saved history to {csv_path}")

    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lr", "best_val_loss", "epochs_ran"])
        writer.writerows(summary_rows)
    print(f"Saved summary to {summary_path}")

    plot_path = args.output_dir / "loss_curves.png"
    save_plot(plot_path, all_histories)
    if plot_path.exists():
        print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()