#!/usr/bin/env python3

"""Evaluate a saved Tenebrio CSRNet checkpoint on the Tenebrio test split.

This computes count-level MAE and R^2 from predicted density-map sums versus
counts recovered from the test density CSV files.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import cfg
from models.CC import CrowdCounter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Tenebrio checkpoint on the test split.")
    parser.add_argument("checkpoint", type=Path, help="Path to a saved model_state_dict .pth file.")
    parser.add_argument("--test-image-dir", type=Path, default=Path("datasets/Tenebrio/test/img"))
    parser.add_argument("--test-density-dir", type=Path, default=Path("datasets/Tenebrio/test/den"))
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional per-image prediction CSV.")
    return parser.parse_args()


def load_image(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path)
    if image.mode == "L":
        image = image.convert("RGB")
    return transforms.ToTensor()(image)


def load_ground_truth_count(density_path: Path) -> float:
    density = pd.read_csv(density_path, sep=",", header=None).values.astype(np.float32, copy=False)
    return float(np.sum(density))


def main() -> None:
    args = parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.test_image_dir.is_dir():
        raise FileNotFoundError(f"Test image directory not found: {args.test_image_dir}")
    if not args.test_density_dir.is_dir():
        raise FileNotFoundError(f"Test density directory not found: {args.test_density_dir}")

    file_names = sorted(path.name for path in args.test_image_dir.glob("*.png"))
    if not file_names:
        raise RuntimeError(f"No PNG files found in {args.test_image_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for practical runtime at full Tenebrio resolution.")

    net = CrowdCounter(cfg.GPU_ID, cfg.NET)
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    net.cuda()
    net.eval()

    predictions: list[tuple[str, float, float, float]] = []
    y_true: list[float] = []
    y_pred: list[float] = []

    for file_name in file_names:
        image_path = args.test_image_dir / file_name
        density_path = args.test_density_dir / f"{Path(file_name).stem}.csv"
        if not density_path.is_file():
            raise FileNotFoundError(f"Missing density CSV for {file_name}: {density_path}")

        gt_count = load_ground_truth_count(density_path)
        image_tensor = load_image(image_path)

        with torch.no_grad():
            batch = Variable(image_tensor[None, :, :, :]).cuda()
            pred_map = net.test_forward(batch)

        pred_count = float(torch.sum(pred_map).item())
        y_true.append(gt_count)
        y_pred.append(pred_count)
        predictions.append((file_name, gt_count, pred_count, pred_count - gt_count))

    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_pred_arr - y_true_arr)))
    ss_res = float(np.sum((y_true_arr - y_pred_arr) ** 2))
    ss_tot = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else float("nan")

    print(f"Samples: {len(predictions)}")
    print(f"MAE: {mae:.6f}")
    print(f"R^2: {r2:.6f}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image", "gt_count", "pred_count", "error"])
            writer.writerows(predictions)
        print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
