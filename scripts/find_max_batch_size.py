#!/usr/bin/env python3
"""Find maximum batch size for lr=1e-5 on resized Tenebrio dataset."""

import torch
import sys
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms
import numpy as np
import json
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.SCC_Model.CSRNet import CSRNet


def build_density_map(width: int, height: int, boxes: list) -> np.ndarray:
    density = np.zeros((height, width), dtype=np.float32)
    for bbox in boxes:
        x, y, w, h = bbox
        center_x = int(round(x + w / 2.0))
        center_y = int(round(y + h / 2.0))
        if 0 <= center_x < width and 0 <= center_y < height:
            density[center_y, center_x] += 1.0
    if density.sum() > 0:
        density = gaussian_filter(density, sigma=15, mode="constant")
    return density.astype(np.float32, copy=False)


def load_sample_images(n_samples=4):
    """Load n sample images and their annotations, cycling through if needed."""
    dataset_root = Path("datasets/Tenebrio")
    ann_file = dataset_root / "TenebrioVision_Annotations.json"
    img_dir = dataset_root / "train/img"
    
    with ann_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    id_to_fname = {img["id"]: img["file_name"] for img in data["images"]}
    grouped_boxes = defaultdict(list)
    for ann in data["annotations"]:
        grouped_boxes[int(ann["image_id"])].append(ann["bbox"])
    
    samples = []
    to_tensor = transforms.ToTensor()
    
    # Cycle through images if needed
    img_entries = [img for img in data["images"] if (img_dir / img["file_name"]).is_file()]
    
    for i in range(n_samples):
        img_entry = img_entries[i % len(img_entries)]
        fname = img_entry["file_name"]
        img_path = img_dir / fname
        
        img = Image.open(img_path).convert("RGB")
        img_tensor = to_tensor(img)
        
        boxes = grouped_boxes.get(img_entry["id"], [])
        density = build_density_map(img_entry["width"], img_entry["height"], boxes)
        density_tensor = torch.from_numpy(density).unsqueeze(0)
        
        samples.append((img_tensor, density_tensor))
    
    return samples


def test_batch_size(batch_size, lr=1e-5):
    """Test a specific batch size. Returns (success, peak_memory_mb)."""
    device = torch.device("cuda")
    
    # Load sample images
    samples = load_sample_images(n_samples=batch_size)
    if len(samples) < batch_size:
        return False, 0, "Not enough samples"
    
    # Stack into batch
    images = torch.stack([s[0] for s in samples]).to(device)
    targets = torch.stack([s[1] for s in samples]).to(device)
    
    # Create model
    model = CSRNet(load_weights=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    try:
        # Forward pass
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        predictions = model(images)
        targets_resized = targets[:, :, :predictions.shape[2], :predictions.shape[3]]
        loss = criterion(predictions, targets_resized)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return True, peak_memory, "OK"
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return False, 0, "OOM"
        raise
    finally:
        del model
        del optimizer
        torch.cuda.empty_cache()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available")
        return
    
    print(f"Testing batch sizes with lr=1e-5 on resized Tenebrio (386x260)")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GiB")
    print()
    
    batch_sizes = [88, 89, 90, 91, 92, 93, 94, 95]
    max_batch = 0
    
    for bs in batch_sizes:
        print(f"Testing batch_size={bs}...", end=" ", flush=True)
        success, peak_mem, msg = test_batch_size(bs, lr=1e-5)
        
        if success:
            print(f"✓ Peak VRAM: {peak_mem:.1f} MiB ({peak_mem/1024:.1f} GiB)")
            max_batch = bs
        else:
            print(f"✗ {msg}")
            break
    
    print()
    print(f"Maximum batch size: {max_batch}")


if __name__ == "__main__":
    main()
