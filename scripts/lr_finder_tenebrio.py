#!/usr/bin/env python3

"""Learning-rate finder for Tenebrio using CSRNet and batch size 1.

This script performs a Leslie Smith style LR range test:
- start from a very small learning rate
- increase it exponentially over a fixed number of steps
- track the smoothed loss
- report the best LR near the steepest loss descent

The script uses the COCO-style Tenebrio annotations already stored in
datasets/Tenebrio/TenebrioVision_Annotations.json and generates a density
target on the fly from bbox centers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
        return image_tensor, density_tensor, sample.image_path.name

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
    parser = argparse.ArgumentParser(description="Run an LR finder for Tenebrio CSRNet training.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("datasets/Tenebrio/train/img"),
        help="Directory containing the Tenebrio training images.",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=Path("datasets/Tenebrio/TenebrioVision_Annotations.json"),
        help="COCO-style annotation file for Tenebrio.",
    )
    parser.add_argument(
        "--start-lr",
        type=float,
        default=1e-7,
        help="Initial learning rate for the range test.",
    )
    parser.add_argument(
        "--end-lr",
        type=float,
        default=1e-3,
        help="Final learning rate for the range test.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of LR updates to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exp/tenebrio_lr_finder.csv"),
        help="CSV file to store the LR sweep results.",
    )
    return parser.parse_args()


def build_dataloader(image_dir: Path, annotation_file: Path) -> DataLoader:
    dataset = TenebrioDensityDataset(image_dir, annotation_file)
    return DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)


def find_lr(model: nn.Module, dataloader: DataLoader, start_lr: float, end_lr: float, num_steps: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This LR finder is intended to run on CUDA because batch-size 1 at full Tenebrio resolution is memory-heavy.")

    model = model.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)

    lr_mult = (end_lr / start_lr) ** (1.0 / max(1, num_steps - 1))
    best_loss = float("inf")
    best_lr = start_lr
    avg_loss = 0.0
    beta = 0.98
    history = []

    data_iter = iter(dataloader)

    for step in range(num_steps):
        try:
            images, targets, names = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, targets, names = next(data_iter)

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        current_lr = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        targets = targets[:, :, : predictions.shape[2], : predictions.shape[3]]
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        avg_loss = beta * avg_loss + (1 - beta) * loss_value
        smoothed_loss = avg_loss / (1 - beta ** (step + 1))

        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
            best_lr = current_lr

        history.append((step + 1, current_lr, loss_value, smoothed_loss, names[0]))
        print(
            f"step={step + 1:03d} lr={current_lr:.3e} loss={loss_value:.6f} "
            f"smoothed={smoothed_loss:.6f} sample={names[0]}"
        )

        if step > 0 and smoothed_loss > 4.0 * best_loss:
            print("Stopping early because the loss diverged.")
            break

        optimizer.param_groups[0]["lr"] *= lr_mult

    return history, best_lr


def main() -> None:
    args = parse_args()
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    if not args.annotation_file.is_file():
        raise FileNotFoundError(f"Annotation file not found: {args.annotation_file}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    dataloader = build_dataloader(args.image_dir, args.annotation_file)
    model = CSRNet(load_weights=True)

    history, best_lr = find_lr(model, dataloader, args.start_lr, args.end_lr, args.num_steps)

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "lr", "loss", "smoothed_loss", "sample"])
        writer.writerows(history)

    print(f"Best LR candidate: {best_lr:.3e}")
    print(f"Saved LR sweep to {args.output}")


if __name__ == "__main__":
    main()