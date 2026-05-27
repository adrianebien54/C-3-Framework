#!/usr/bin/env python3

"""Precompute Tenebrio ground-truth density maps into split `den` folders.

Workflow:
  1. Build density impulse map at full image resolution.
  2. Apply Gaussian blur with sigma (in full-resolution pixels; default 15).
  3. Downsample to 1/8 of the image resolution (matching CSRNet's output stride).
  4. Renormalise so the pixel sum equals the object count.
  5. Write as a CSV file named after the image stem.

The --sigma flag lets you sweep over different kernel sizes without touching
the model.  The --output-dir flag writes density CSVs to a parallel directory
(and symlinks the img/ folders from --data-dir) so multiple sigma variants can
live side-by-side without duplicating images.

Usage:
  # Default: regenerate sigma=15 CSVs in-place (skip existing)
  python scripts/precompute_tenebrio_densities.py \
      --data-dir datasets/Tenebrio/386x260 \
      --annotation-file datasets/Tenebrio/386x260/TenebrioVision_Annotations_386x260.json

  # Sigma variant stored separately:
  python scripts/precompute_tenebrio_densities.py \
      --data-dir datasets/Tenebrio/386x260 \
      --annotation-file datasets/Tenebrio/386x260/TenebrioVision_Annotations_386x260.json \
      --sigma 5 \
      --output-dir datasets/Tenebrio/386x260_s5 \
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


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
    parser = argparse.ArgumentParser(description="Precompute Tenebrio density CSVs.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/Tenebrio"),
        help="Tenebrio dataset root containing train/val/test splits.",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=Path("datasets/Tenebrio/TenebrioVision_Annotations.json"),
        help="COCO-style Tenebrio annotation JSON.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=15.0,
        help="Gaussian sigma in full-resolution pixels (default 15.0). "
             "The density is blurred at full resolution then downsampled to 1/8.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If set, write density CSVs here instead of --data-dir. "
             "Symlinks for img/ folders are created automatically so the "
             "output dir is a complete dataset root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing density CSV files.",
    )
    return parser.parse_args()


def load_annotation_samples(data_dir: Path, annotation_file: Path) -> list[TenebrioSample]:
    if not annotation_file.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    with annotation_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    image_index: dict[int, dict[str, object]] = {}
    file_name_to_split: dict[str, str] = {}
    for split in SPLITS:
        split_image_dir = data_dir / split / "img"
        if not split_image_dir.is_dir():
            continue
        for image_path in split_image_dir.glob("*.png"):
            file_name_to_split[image_path.name] = split

    for image_entry in data.get("images", []):
        file_name = str(image_entry["file_name"])
        split = file_name_to_split.get(file_name)
        image_path = data_dir / split / "img" / file_name if split else None
        if split is None or image_path is None or not image_path.is_file():
            continue

        image_index[int(image_entry["id"])] = {
            "split": split,
            "image_path": image_path,
            "file_name": file_name,
            "width": int(image_entry["width"]),
            "height": int(image_entry["height"]),
        }

    grouped_boxes: dict[int, list[list[float]]] = defaultdict(list)
    for annotation in data.get("annotations", []):
        grouped_boxes[int(annotation["image_id"])]
        grouped_boxes[int(annotation["image_id"])] = grouped_boxes[int(annotation["image_id"])] + [
            [float(value) for value in annotation["bbox"]]
        ]

    samples: list[TenebrioSample] = []
    missing_files = 0
    for image_id, image_info in image_index.items():
        image_path = image_info["image_path"]
        if not isinstance(image_path, Path) or not image_path.is_file():
            missing_files += 1
            continue

        samples.append(
            TenebrioSample(
                split=str(image_info["split"]),
                image_path=image_path,
                file_name=str(image_info["file_name"]),
                width=int(image_info["width"]),
                height=int(image_info["height"]),
                boxes=grouped_boxes.get(image_id, []),
            )
        )

    samples.sort(key=lambda sample: (sample.split, sample.file_name))
    if not samples:
        raise RuntimeError(
            f"No matching images found under {data_dir} for annotation file {annotation_file}."
        )
    if missing_files > 0:
        print(f"[precompute] Skipped {missing_files} images listed in annotations but missing on disk.")

    return samples


def build_density_map(
    width: int,
    height: int,
    boxes: list[list[float]],
    sigma: float = 15.0,
    downsample: int = 8,
) -> np.ndarray:
    """Build a 1/8-scale density map.

    Steps:
      1. Place fractional impulses at full resolution.
      2. Apply Gaussian blur (sigma in full-resolution pixels).
      3. Downsample to 1/downsample resolution with BICUBIC resampling.
      4. Renormalise so sum == object count.
    """
    density = np.zeros((height, width), dtype=np.float32)
    target_count = float(len(boxes))

    for bbox in boxes:
        x, y, box_width, box_height = bbox
        cx = x + box_width / 2.0
        cy = y + box_height / 2.0

        x0 = int(np.floor(cx))
        y0 = int(np.floor(cy))
        x1 = x0 + 1
        y1 = y0 + 1

        wx1 = cx - x0
        wy1 = cy - y0
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        if 0 <= x0 < width  and 0 <= y0 < height: density[y0, x0] += wx0 * wy0
        if 0 <= x1 < width  and 0 <= y0 < height: density[y0, x1] += wx1 * wy0
        if 0 <= x0 < width  and 0 <= y1 < height: density[y1, x0] += wx0 * wy1
        if 0 <= x1 < width  and 0 <= y1 < height: density[y1, x1] += wx1 * wy1

    if density.sum() > 0:
        density = gaussian_filter(density, sigma=sigma, mode="reflect")
        density *= target_count / density.sum()

    # Downsample to 1/downsample resolution (ceil so edge pixels are covered)
    if downsample > 1:
        out_w = math.ceil(width / downsample)
        out_h = math.ceil(height / downsample)
        density_pil = Image.fromarray(density).resize((out_w, out_h), Image.BICUBIC)
        density = np.array(density_pil, dtype=np.float32)
        # Renormalise after resize to preserve object count
        if density.sum() > 0:
            density *= target_count / density.sum()

    return density.astype(np.float32, copy=False)


def ensure_img_symlink(src_img_dir: Path, dst_img_dir: Path) -> None:
    """Create dst_img_dir as a symlink to src_img_dir (absolute path)."""
    dst_img_dir.parent.mkdir(parents=True, exist_ok=True)
    if dst_img_dir.exists() or dst_img_dir.is_symlink():
        return
    os.symlink(src_img_dir.resolve(), dst_img_dir)


def write_density_csv(output_path: Path, density: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, density, delimiter=",", fmt="%.8f")


def main() -> None:
    args = parse_args()

    out_dir = args.output_dir if args.output_dir is not None else args.data_dir

    # If writing to a separate output dir, create img symlinks for each split
    if args.output_dir is not None:
        for split in SPLITS:
            src_img = args.data_dir / split / "img"
            dst_img = args.output_dir / split / "img"
            if src_img.is_dir():
                ensure_img_symlink(src_img, dst_img)

    samples = load_annotation_samples(args.data_dir, args.annotation_file)

    written = 0
    skipped_existing = 0
    for sample in samples:
        output_path = out_dir / sample.split / "den" / f"{Path(sample.file_name).stem}.csv"
        if output_path.is_file() and not args.overwrite:
            skipped_existing += 1
            continue

        with Image.open(sample.image_path) as image:
            image_width, image_height = image.size

        # Scale annotation coordinates (in original image space) to actual image space.
        if (image_width, image_height) != (sample.width, sample.height):
            sx = image_width / sample.width
            sy = image_height / sample.height
            scaled_boxes = [
                [x * sx, y * sy, bw * sx, bh * sy]
                for x, y, bw, bh in sample.boxes
            ]
        else:
            scaled_boxes = sample.boxes

        density = build_density_map(image_width, image_height, scaled_boxes,
                                    sigma=args.sigma, downsample=8)
        write_density_csv(output_path, density)
        written += 1

    print(f"[precompute] sigma={args.sigma}  output={out_dir}")
    print(f"[precompute] Processed {len(samples)} images")
    print(f"[precompute] Wrote {written} density CSV files")
    if skipped_existing > 0:
        print(f"[precompute] Skipped {skipped_existing} existing density CSV files")


if __name__ == "__main__":
    main()
