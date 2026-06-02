#!/usr/bin/env python3

"""Precompute per-image point-coordinate files for patch-based CSRNet training.

For each image in the dataset, saves a float32 .npy array of shape (N, 2)
containing the (cx, cy) bounding-box centre coordinates scaled to the actual
image resolution stored on disk.  These files are consumed by the Tenebrio
dataset class at train time to generate density maps on-the-fly.

Usage:
  python scripts/precompute_tenebrio_points.py \\
      --data-dir datasets/Tenebrio/386x260 \\
      --annotation-file datasets/Tenebrio/TenebrioVision_Annotations.json

  # Repeat for every resolution you train at:
  python scripts/precompute_tenebrio_points.py \\
      --data-dir datasets/Tenebrio/772x519 \\
      --annotation-file datasets/Tenebrio/TenebrioVision_Annotations.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class TenebrioSample:
    split: str
    image_path: Path
    file_name: str
    width: int
    height: int
    boxes: list[list[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute Tenebrio point-coordinate .npy files.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/Tenebrio/386x260"),
        help="Tenebrio resolution root containing train/val/test splits.",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=Path("datasets/Tenebrio/TenebrioVision_Annotations.json"),
        help="COCO-style Tenebrio annotation JSON (coordinates in original image space).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files.",
    )
    return parser.parse_args()


def load_annotation_samples(data_dir: Path, annotation_file: Path) -> list[TenebrioSample]:
    if not annotation_file.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    with annotation_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    file_name_to_split: dict[str, str] = {}
    for split in SPLITS:
        split_img_dir = data_dir / split / "img"
        if not split_img_dir.is_dir():
            continue
        for img_path in split_img_dir.glob("*.png"):
            file_name_to_split[img_path.name] = split

    image_index: dict[int, dict] = {}
    for entry in data.get("images", []):
        file_name = str(entry["file_name"])
        split = file_name_to_split.get(file_name)
        if split is None:
            continue
        image_path = data_dir / split / "img" / file_name
        if not image_path.is_file():
            continue
        image_index[int(entry["id"])] = {
            "split": split,
            "image_path": image_path,
            "file_name": file_name,
            "width": int(entry["width"]),
            "height": int(entry["height"]),
        }

    grouped_boxes: dict[int, list[list[float]]] = defaultdict(list)
    for ann in data.get("annotations", []):
        img_id = int(ann["image_id"])
        grouped_boxes[img_id].append([float(v) for v in ann["bbox"]])

    samples: list[TenebrioSample] = []
    missing = 0
    for img_id, info in image_index.items():
        if not info["image_path"].is_file():
            missing += 1
            continue
        samples.append(TenebrioSample(
            split=info["split"],
            image_path=info["image_path"],
            file_name=info["file_name"],
            width=info["width"],
            height=info["height"],
            boxes=grouped_boxes.get(img_id, []),
        ))

    samples.sort(key=lambda s: (s.split, s.file_name))
    if not samples:
        raise RuntimeError(f"No matching images found under {data_dir}.")
    if missing:
        print(f"[points] Skipped {missing} images listed in annotations but missing on disk.")
    return samples


def main() -> None:
    args = parse_args()

    samples = load_annotation_samples(args.data_dir, args.annotation_file)

    written = skipped = 0
    for sample in samples:
        out_path = args.data_dir / sample.split / "pts" / f"{Path(sample.file_name).stem}.npy"
        if out_path.is_file() and not args.overwrite:
            skipped += 1
            continue

        with Image.open(sample.image_path) as img:
            img_w, img_h = img.size

        # Scale annotation coordinates (stored in original JSON resolution) to
        # the actual pixel dimensions of this resolution variant.
        sx = img_w / sample.width
        sy = img_h / sample.height

        centers: list[tuple[float, float]] = []
        for x, y, bw, bh in sample.boxes:
            cx = (x + bw / 2.0) * sx
            cy = (y + bh / 2.0) * sy
            centers.append((cx, cy))

        pts = np.array(centers, dtype=np.float32).reshape(-1, 2)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, pts)
        written += 1

    print(f"[points] data_dir={args.data_dir}")
    print(f"[points] Processed {len(samples)} images")
    print(f"[points] Wrote    {written} .npy files")
    if skipped:
        print(f"[points] Skipped  {skipped} existing files (use --overwrite to regenerate)")


if __name__ == "__main__":
    main()
