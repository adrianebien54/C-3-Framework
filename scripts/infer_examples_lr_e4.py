#!/usr/bin/env python3
"""Infer example test images with the lr=1e-4 checkpoint and save density maps and counts.

Saves PNG visualizations and .npy density arrays plus a CSV with per-image counts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import csv

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.SCC_Model.CSRNet import CSRNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--annotation", type=Path, default=Path("datasets/Tenebrio/TenebrioVision_Annotations.json"))
    p.add_argument("--test-img-dir", type=Path, default=Path("datasets/Tenebrio/test/img"))
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--output-dir", type=Path, default=Path("exp/tenebrio_lr_retrain_e4_074/examples"))
    p.add_argument("--seed", type=int, default=3035)
    return p.parse_args()


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


def visualize_and_save(density: np.ndarray, out_png: Path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.imshow(density, cmap='viridis')
    plt.colorbar()
    plt.axis('off')
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ann_index = build_annotation_index(args.annotation)

    file_names = sorted(p.name for p in args.test_img_dir.glob("*.png"))
    if not file_names:
        raise RuntimeError(f"No PNG files in {args.test_img_dir}")

    sample_names = file_names[: args.n_samples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt

    model = CSRNet(load_weights=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    results = []
    for name in sample_names:
        img_path = args.test_img_dir / name
        img = load_image(img_path)
        h, w = Image.open(img_path).size[1], Image.open(img_path).size[0]
        boxes = ann_index.get(name, [])
        gt_den = build_density_map(w, h, boxes)
        gt_count = float(gt_den.sum())

        with torch.no_grad():
            batch = img.unsqueeze(0).to(device)
            pred = model(batch)
            pred_np = pred.squeeze().cpu().numpy()
            pred_count = float(pred_np.sum())

        # save outputs
        base = args.output_dir / Path(name).stem
        np.save(base.with_suffix('.pred.npy'), pred_np)
        visualize_and_save(pred_np, base.with_suffix('.pred.png'))
        np.save(base.with_suffix('.gt.npy'), gt_den)
        visualize_and_save(gt_den, base.with_suffix('.gt.png'))

        results.append((name, gt_count, pred_count, pred_count - gt_count))

    # write CSV
    out_csv = args.output_dir / "predictions_examples.csv"
    with out_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(["image","gt_count","pred_count","error"])
        writer.writerows(results)

    print(f"Wrote {len(results)} examples to {args.output_dir}")
    print(f"CSV: {out_csv}")


if __name__ == '__main__':
    main()
