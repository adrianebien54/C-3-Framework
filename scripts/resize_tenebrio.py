#!/usr/bin/env python3
"""Resize all Tenebrio images to 386x260 and update annotations accordingly."""

import json
from pathlib import Path
from PIL import Image
import sys

TARGET_WIDTH = 386
TARGET_HEIGHT = 260


def main():
    # Paths
    dataset_root = Path("datasets/Tenebrio")
    ann_file = dataset_root / "TenebrioVision_Annotations.json"
    
    # Load annotations
    with ann_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Loaded annotations with {len(data['images'])} images and {len(data['annotations'])} annotations")
    
    # Process each image
    resized_count = 0
    for img_entry in data["images"]:
        fname = img_entry["file_name"]
        
        # Try to find image in train/val/test
        img_path = None
        for subdir in ["train/img", "val/img", "test/img"]:
            candidate = dataset_root / subdir / fname
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path is None:
            print(f"Warning: Could not find {fname}")
            continue
        
        # Get original dimensions
        orig_img = Image.open(img_path)
        orig_width, orig_height = orig_img.size
        
        # Resize and save
        resized_img = orig_img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
        resized_img.save(img_path)
        
        # Update image dimensions in annotation
        img_entry["width"] = TARGET_WIDTH
        img_entry["height"] = TARGET_HEIGHT
        
        # Scale bbox coordinates for this image's annotations
        scale_x = TARGET_WIDTH / orig_width
        scale_y = TARGET_HEIGHT / orig_height
        
        for ann in data["annotations"]:
            if ann["image_id"] == img_entry["id"]:
                # bbox format: [x, y, width, height]
                x, y, w, h = ann["bbox"]
                ann["bbox"] = [
                    x * scale_x,
                    y * scale_y,
                    w * scale_x,
                    h * scale_y
                ]
        
        resized_count += 1
        if resized_count % 100 == 0:
            print(f"Resized {resized_count}/{len(data['images'])} images...")
    
    print(f"Resized {resized_count} images to {TARGET_WIDTH}x{TARGET_HEIGHT}")
    
    # Save updated annotations
    with ann_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Updated annotations saved to {ann_file}")


if __name__ == "__main__":
    main()
