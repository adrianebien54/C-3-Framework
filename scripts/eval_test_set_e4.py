#!/usr/bin/env python3
"""Evaluate CSRNet checkpoint on Tenebrio test set and compute MAE + R^2."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from PIL import Image
from torchvision import transforms

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.SCC_Model.CSRNet import CSRNet


def build_annotation_index(annotation_file: Path):
    with annotation_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    id_to_fname = {img["id"]: img["file_name"] for img in data["images"]}
    grouped = {}
    for ann in data["annotations"]:
        fname = id_to_fname.get(ann["image_id"])
        if fname is None:
            continue
        grouped.setdefault(fname, []).append(ann["bbox"])
    return grouped


def build_density_map(width: int, height: int, boxes: list[list[float]]):
    density = np.zeros((height, width), dtype=np.float32)
    for bbox in boxes:
        x, y, w, h = bbox
        cx = int(round(x + w / 2.0))
        cy = int(round(y + h / 2.0))
        if 0 <= cx < width and 0 <= cy < height:
            density[cy, cx] += 1.0
    if density.sum() > 0:
        density = gaussian_filter(density, sigma=15, mode='constant')
    return density.astype(np.float32, copy=False)


def load_image(path: Path):
    img = Image.open(path)
    if img.mode == "L":
        img = img.convert("RGB")
    transform = transforms.ToTensor()
    return transform(img)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--annotation", type=Path, default=Path("datasets/Tenebrio/TenebrioVision_Annotations.json"))
    p.add_argument("--test-img-dir", type=Path, default=Path("datasets/Tenebrio/test/img"))
    p.add_argument("--output-csv", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    model = CSRNet(load_weights=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Load annotation index
    ann_index = build_annotation_index(args.annotation)

    # Get test file names
    file_names = sorted(p.name for p in args.test_img_dir.glob("*.png"))
    if not file_names:
        raise RuntimeError(f"No PNG files in {args.test_img_dir}")

    results = []
    y_true_list = []
    y_pred_list = []

    for name in file_names:
        img_path = args.test_img_dir / name
        boxes = ann_index.get(name, [])

        # Ground truth
        h, w = Image.open(img_path).size[1], Image.open(img_path).size[0]
        gt_den = build_density_map(w, h, boxes)
        gt_count = float(gt_den.sum())

        # Predict
        img = load_image(img_path)
        with torch.no_grad():
            batch = img.unsqueeze(0).to(device)
            pred = model(batch)
            pred_np = pred.squeeze().cpu().numpy()
            pred_count = float(pred_np.sum())

        error = pred_count - gt_count
        results.append((name, gt_count, pred_count, error))
        y_true_list.append(gt_count)
        y_pred_list.append(pred_count)

    # Compute metrics
    y_true = np.array(y_true_list, dtype=np.float64)
    y_pred = np.array(y_pred_list, dtype=np.float64)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else float("nan")

    print(f"Test set evaluation (checkpoint={args.checkpoint.name})")
    print(f"  Samples: {len(results)}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.6f}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(["image", "gt_count", "pred_count", "error"])
            writer.writerows(results)
        print(f"Saved per-image results to {args.output_csv}")


if __name__ == '__main__':
    main()
