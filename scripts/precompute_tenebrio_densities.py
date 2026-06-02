#!/usr/bin/env python3

"""Precompute Tenebrio ground-truth density maps into split `den` folders.

Workflow:
  1. Build density impulse map at full image resolution.
  2. Apply Gaussian blur with the given sigma (in full-resolution pixels).
  3. Downsample to 1/8 using sum-pooling (reshape into 8×8 blocks and sum),
     which preserves the total count exactly without renormalisation.
  4. Write as an HDF5 file (.h5) with dataset key 'density'.

The --output-dir flag writes density CSVs to a parallel directory
(and symlinks the img/ folders from --data-dir) so multiple variants can
live side-by-side without duplicating images.

Usage:
  # Default: regenerate density CSVs in-place (sigma auto-computed from bbox sizes)
  python scripts/precompute_tenebrio_densities.py \
      --data-dir datasets/Tenebrio/386x260 \
      --annotation-file datasets/Tenebrio/386x260/TenebrioVision_Annotations_386x260.json

  # Override sigma explicitly:
  python scripts/precompute_tenebrio_densities.py \
      --data-dir datasets/Tenebrio/386x260 \
      --annotation-file datasets/Tenebrio/386x260/TenebrioVision_Annotations_386x260.json \
      --sigma 11 \
      --output-dir datasets/Tenebrio/386x260_s11 \
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

import h5py
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
        default=None,
        help="Gaussian sigma in full-resolution pixels. "
             "If omitted, computed automatically as mean((bbox_w + bbox_h) / 2) / 2 "
             "from the annotation bounding boxes.",
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


def compute_sigma_from_boxes(boxes_list: list[list[list[float]]]) -> float:
    """Compute sigma as mean((bbox_w + bbox_h) / 2) / 2 across all boxes."""
    sizes = [(bw + bh) / 2.0 for boxes in boxes_list for _, _, bw, bh in boxes]
    return (sum(sizes) / len(sizes)) / 2.0 if sizes else 15.0


def build_density_map(
    width: int,
    height: int,
    boxes: list[list[float]],
    sigma: float,
    downsample: int = 8,
) -> np.ndarray:
    """Build a 1/8-scale density map.

    Steps:
      1. Place fractional impulses at full resolution.
      2. Apply Gaussian blur (sigma in full-resolution pixels).
      3. Downsample to 1/downsample resolution with BICUBIC resampling.
      4. Renormalise so sum == object count.
    """
    # Pad canvas to the nearest multiple of downsample before placing impulses,
    # so the Gaussian tails extend naturally into the padded region during blur.
    pad_h = (downsample - height % downsample) % downsample
    pad_w = (downsample - width  % downsample) % downsample
    H = height + pad_h
    W = width  + pad_w

    density = np.zeros((H, W), dtype=np.float32)

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

        if 0 <= x0 < W and 0 <= y0 < H: density[y0, x0] += wx0 * wy0
        if 0 <= x1 < W and 0 <= y0 < H: density[y0, x1] += wx1 * wy0
        if 0 <= x0 < W and 0 <= y1 < H: density[y1, x0] += wx0 * wy1
        if 0 <= x1 < W and 0 <= y1 < H: density[y1, x1] += wx1 * wy1

    if density.sum() > 0:
        density = gaussian_filter(density, sigma=sigma, mode="reflect")

    # Sum-pool: exact count preservation since all impulses were placed on the
    # padded canvas and blurred without boundary trimming.
    if downsample > 1:
        density = density.reshape(H // downsample, downsample,
                                  W // downsample, downsample).sum(axis=(1, 3))

    return density.astype(np.float32, copy=False)


def ensure_img_symlink(src_img_dir: Path, dst_img_dir: Path) -> None:
    """Create dst_img_dir as a symlink to src_img_dir (absolute path)."""
    dst_img_dir.parent.mkdir(parents=True, exist_ok=True)
    if dst_img_dir.exists() or dst_img_dir.is_symlink():
        return
    os.symlink(src_img_dir.resolve(), dst_img_dir)


def write_density_h5(output_path: Path, density: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("density", data=density, compression="gzip", compression_opts=1)


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

    if args.sigma is not None:
        sigma = args.sigma
    else:
        # Scale boxes to actual image resolution before computing sigma
        scaled_boxes_all = []
        for sample in samples:
            with Image.open(sample.image_path) as image:
                iw, ih = image.size
            if (iw, ih) != (sample.width, sample.height):
                sx, sy = iw / sample.width, ih / sample.height
                scaled_boxes_all.append([[x*sx, y*sy, bw*sx, bh*sy] for x,y,bw,bh in sample.boxes])
            else:
                scaled_boxes_all.append(sample.boxes)
        sigma = compute_sigma_from_boxes(scaled_boxes_all)
        print(f"[precompute] sigma auto-computed: {sigma:.2f} (mean bbox avg-dim / 2)")

    written = 0
    skipped_existing = 0
    for sample in samples:
        output_path = out_dir / sample.split / "den" / f"{Path(sample.file_name).stem}.h5"
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
                                    sigma=sigma, downsample=8)
        write_density_h5(output_path, density)
        written += 1

    print(f"[precompute] sigma={sigma:.2f}  output={out_dir}")
    print(f"[precompute] Processed {len(samples)} images")
    print(f"[precompute] Wrote {written} density HDF5 files (.h5)")
    if skipped_existing > 0:
        print(f"[precompute] Skipped {skipped_existing} existing density CSV files")


if __name__ == "__main__":
    main()
