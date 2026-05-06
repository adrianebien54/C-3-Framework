#!/usr/bin/env python3
"""
Augment Tenebrio dataset with horizontal and vertical flips.
Creates 4x the dataset: original + h-flip + v-flip + both flips.
Saves augmented images and updates annotations JSON.
"""

import json
import argparse
from pathlib import Path
from PIL import Image
import shutil


def transform_bbox_hflip(bbox, width):
    """Transform bbox for horizontal flip. bbox = [x, y, w, h]"""
    x, y, w, h = bbox
    new_x = width - x - w
    return [new_x, y, w, h]


def transform_bbox_vflip(bbox, height):
    """Transform bbox for vertical flip. bbox = [x, y, w, h]"""
    x, y, w, h = bbox
    new_y = height - y - h
    return [x, new_y, w, h]


def transform_bbox_both(bbox, width, height):
    """Transform bbox for both flips."""
    x, y, w, h = bbox
    new_x = width - x - w
    new_y = height - y - h
    return [new_x, new_y, w, h]


def augment_dataset(data_dir="datasets/Tenebrio", output_dir="datasets/Tenebrio_augmented"):
    """
    Augment dataset with flips.
    
    Args:
        data_dir: Original dataset directory
        output_dir: Output directory for augmented dataset
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Load original annotations
    anno_path = data_dir / "TenebrioVision_Annotations.json"
    with open(anno_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['images'])} images, {len(data['annotations'])} annotations")
    
    # Create output directories
    output_dir.mkdir(exist_ok=True)
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'img').mkdir(parents=True, exist_ok=True)
    
    # Copy original images first (to preserve file structure)
    print("\nCopying original images...")
    for split in ['train', 'val', 'test']:
        src_dir = data_dir / split / 'img'
        dst_dir = output_dir / split / 'img'
        if src_dir.exists():
            files = list(src_dir.glob('*.png'))
            for idx, img_file in enumerate(files, 1):
                shutil.copy2(img_file, dst_dir / img_file.name)
                if idx % 100 == 0:
                    print(f"  {split}: {idx}/{len(files)}")
                shutil.copy2(img_file, dst_dir / img_file.name)
    
    # Create augmented images and build new annotation data
    new_images = []
    new_annotations = []
    
    image_id_map = {img['id']: img for img in data['images']}
    anno_by_image = {}
    for anno in data['annotations']:
        img_id = anno['image_id']
        if img_id not in anno_by_image:
            anno_by_image[img_id] = []
        anno_by_image[img_id].append(anno)
    
    # Add original images and annotations
    print("\nAdding original images to augmented dataset...")
    new_image_id = 0
    next_anno_id = 0
    orig_to_new_id = {}
    
    for idx, orig_img in enumerate(data['images'], 1):
        if idx % 100 == 0:
            print(f"  Processing original: {idx}/{len(data['images'])}")
        orig_img_id = orig_img['id']
        orig_to_new_id[orig_img_id] = new_image_id
        
        new_img = orig_img.copy()
        new_img['id'] = new_image_id
        new_images.append(new_img)
        
        # Add annotations for this image
        if orig_img_id in anno_by_image:
            for anno in anno_by_image[orig_img_id]:
                new_anno = anno.copy()
                new_anno['id'] = next_anno_id
                new_anno['image_id'] = new_image_id
                new_annotations.append(new_anno)
                next_anno_id += 1
        
        new_image_id += 1
    
    # Create augmented versions
    print("\nCreating horizontally flipped images...")
    for idx, orig_img in enumerate(data['images'], 1):
        if idx % 100 == 0:
            print(f"  H-flip: {idx}/{len(data['images'])}")
        orig_img_id = orig_img['id']
        width, height = orig_img['width'], orig_img['height']
        file_name = orig_img['file_name']
        
        # Find split (train/val/test)
        split = None
        for s in ['train', 'val', 'test']:
            if (data_dir / s / 'img' / file_name).exists():
                split = s
                break
        
        if split is None:
            print(f"Warning: Could not find {file_name} in any split")
            continue
        
        # Load original image
        img_path = data_dir / split / 'img' / file_name
        img = Image.open(img_path)
        
        # Horizontal flip
        img_hflip = img.transpose(Image.FLIP_LEFT_RIGHT)
        hflip_name = file_name.replace('.png', '_hflip.png')
        hflip_path = output_dir / split / 'img' / hflip_name
        img_hflip.save(hflip_path)
        
        new_img = orig_img.copy()
        new_img['id'] = new_image_id
        new_img['file_name'] = hflip_name
        new_images.append(new_img)
        
        if orig_img_id in anno_by_image:
            for anno in anno_by_image[orig_img_id]:
                new_anno = anno.copy()
                new_anno['id'] = next_anno_id
                new_anno['image_id'] = new_image_id
                new_anno['bbox'] = transform_bbox_hflip(anno['bbox'], width)
                new_annotations.append(new_anno)
                next_anno_id += 1
        
        new_image_id += 1
    
    print("\nCreating vertically flipped images...")
    for idx, orig_img in enumerate(data['images'], 1):
        if idx % 100 == 0:
            print(f"  V-flip: {idx}/{len(data['images'])}")
        orig_img_id = orig_img['id']
        width, height = orig_img['width'], orig_img['height']
        file_name = orig_img['file_name']
        
        # Find split
        split = None
        for s in ['train', 'val', 'test']:
            if (data_dir / s / 'img' / file_name).exists():
                split = s
                break
        
        if split is None:
            continue
        
        # Load original image
        img_path = data_dir / split / 'img' / file_name
        img = Image.open(img_path)
        
        # Vertical flip
        img_vflip = img.transpose(Image.FLIP_TOP_BOTTOM)
        vflip_name = file_name.replace('.png', '_vflip.png')
        vflip_path = output_dir / split / 'img' / vflip_name
        img_vflip.save(vflip_path)
        
        new_img = orig_img.copy()
        new_img['id'] = new_image_id
        new_img['file_name'] = vflip_name
        new_images.append(new_img)
        
        if orig_img_id in anno_by_image:
            for anno in anno_by_image[orig_img_id]:
                new_anno = anno.copy()
                new_anno['id'] = next_anno_id
                new_anno['image_id'] = new_image_id
                new_anno['bbox'] = transform_bbox_vflip(anno['bbox'], height)
                new_annotations.append(new_anno)
                next_anno_id += 1
        
        new_image_id += 1
    
    print("\nCreating both-flipped images...")
    for idx, orig_img in enumerate(data['images'], 1):
        if idx % 100 == 0:
            print(f"  Both-flip: {idx}/{len(data['images'])}")
        orig_img_id = orig_img['id']
        width, height = orig_img['width'], orig_img['height']
        file_name = orig_img['file_name']
        
        # Find split
        split = None
        for s in ['train', 'val', 'test']:
            if (data_dir / s / 'img' / file_name).exists():
                split = s
                break
        
        if split is None:
            continue
        
        # Load original image
        img_path = data_dir / split / 'img' / file_name
        img = Image.open(img_path)
        
        # Both flips
        img_both = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
        both_name = file_name.replace('.png', '_both.png')
        both_path = output_dir / split / 'img' / both_name
        img_both.save(both_path)
        
        new_img = orig_img.copy()
        new_img['id'] = new_image_id
        new_img['file_name'] = both_name
        new_images.append(new_img)
        
        if orig_img_id in anno_by_image:
            for anno in anno_by_image[orig_img_id]:
                new_anno = anno.copy()
                new_anno['id'] = next_anno_id
                new_anno['image_id'] = new_image_id
                new_anno['bbox'] = transform_bbox_both(anno['bbox'], width, height)
                new_annotations.append(new_anno)
                next_anno_id += 1
        
        new_image_id += 1
    
    # Save augmented annotations
    print("\nSaving augmented annotations...")
    output_anno_path = output_dir / "TenebrioVision_Annotations_augmented.json"
    
    data['images'] = new_images
    data['annotations'] = new_annotations
    
    with open(output_anno_path, 'w') as f:
        json.dump(data, f)
    
    print(f"\n✅ Augmentation complete!")
    print(f"   Original: {len(data['images']) // 4} images, {len(data['annotations']) // 4} annotations")
    print(f"   Augmented: {len(data['images'])} images, {len(data['annotations'])} annotations")
    print(f"   Saved to: {output_dir}")
    print(f"   Annotations: {output_anno_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augment Tenebrio dataset with flips')
    parser.add_argument('--data-dir', default='datasets/Tenebrio', help='Original dataset directory')
    parser.add_argument('--output-dir', default='datasets/Tenebrio_augmented', help='Output directory')
    args = parser.parse_args()
    
    augment_dataset(args.data_dir, args.output_dir)
