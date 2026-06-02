#!/usr/bin/env python3
"""Visualise CSRNet predictions vs ground truth on a handful of val images."""

import os, sys
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.CC import CrowdCounter
from datasets.Tenebrio.setting import cfg_data

# ── config ────────────────────────────────────────────────────────────────────
CKPT   = ROOT / 'exp/05-31_20-26_Tenebrio_CSRNet_1e-05_386x260_decay0.995/all_ep_488_mae_0.6_mse_0.9.pth'
VAL_IMG = ROOT / 'datasets/Tenebrio/386x260/val/img'
VAL_DEN = ROOT / 'datasets/Tenebrio/386x260/val/den'
OUT     = ROOT / 'scripts/val_preview.png'
N_SHOW  = 6   # number of images to visualise
LOG_PARA = cfg_data.LOG_PARA
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
# ──────────────────────────────────────────────────────────────────────────────

def pad8(img_pil):
    w, h = img_pil.size
    pw = (8 - w % 8) % 8
    ph = (8 - h % 8) % 8
    if pw or ph:
        from PIL import ImageOps
        img_pil = ImageOps.expand(img_pil, border=(0, 0, pw, ph), fill=0)
    return img_pil

img_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

# load model
net = CrowdCounter([0], 'CSRNet').cuda()
state = torch.load(str(CKPT), map_location='cuda', weights_only=False)
if isinstance(state, dict) and 'net' in state:
    net.load_state_dict(state['net'])
else:
    net.load_state_dict(state)
net.eval()
print(f'Loaded checkpoint: {CKPT.name}')

# pick val images
fnames = sorted(os.listdir(VAL_IMG))[:N_SHOW]

fig, axes = plt.subplots(N_SHOW, 3, figsize=(13, 3.2 * N_SHOW))
fig.suptitle(f'CSRNet val preview  |  checkpoint: {CKPT.name}', fontsize=11)

for row, fname in enumerate(fnames):
    # --- load image ---
    img_pil = Image.open(VAL_IMG / fname).convert('RGB')
    img_padded = pad8(img_pil)
    img_t = img_tf(img_padded).unsqueeze(0).cuda()

    # --- load GT density ---
    stem = Path(fname).stem
    with h5py.File(VAL_DEN / f'{stem}.h5', 'r') as hf:
        gt_den = np.array(hf['density'], dtype=np.float32)
    # .h5 stores un-scaled density (sum ≈ count); model outputs density × LOG_PARA
    gt_count = gt_den.sum()
    gt_den_display = gt_den * LOG_PARA   # scale up to match model output space

    # --- predict ---
    with torch.no_grad():
        pred_map = net.CCN(img_t)          # (1,1,H/8,W/8)
    pred_den = pred_map.squeeze().cpu().numpy()
    pred_count = pred_den.sum() / LOG_PARA

    # --- plot ---
    ax_img, ax_gt, ax_pred = axes[row]

    ax_img.imshow(img_pil)
    ax_img.set_title(fname, fontsize=7)
    ax_img.axis('off')

    vmax = max(gt_den_display.max(), pred_den.max(), 1e-6)
    ax_gt.imshow(gt_den_display, cmap='hot', vmin=0, vmax=vmax)
    ax_gt.set_title(f'GT  count={gt_count:.1f}', fontsize=8)
    ax_gt.axis('off')

    ax_pred.imshow(pred_den, cmap='hot', vmin=0, vmax=vmax)
    ax_pred.set_title(f'Pred  count={pred_count:.1f}  err={abs(pred_count-gt_count):.2f}', fontsize=8)
    ax_pred.axis('off')

plt.tight_layout()
plt.savefig(str(OUT), dpi=130, bbox_inches='tight')
print(f'Saved → {OUT}')
