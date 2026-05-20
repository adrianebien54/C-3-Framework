#!/usr/bin/env python3
"""Compare optimizer runs: Adam WD=1e-4 vs Adam no-WD (vs AdamW when done)."""

import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    "Adam  wd=1e-4": ROOT / "exp/05-20_13-07_Tenebrio_CSRNet_0.0001",
    "Adam  wd=0":    ROOT / "exp/05-20_15-15_Tenebrio_CSRNet_0.0001",
    "AdamW wd=1e-4": ROOT / "exp/05-20_15-31_Tenebrio_CSRNet_0.0001",
}

COLORS = ["#e07b39", "#3a6ea5", "#2ca02c"]


def load(exp_dir, tag):
    steps, values = [], []
    for tf in sorted(glob.glob(str(exp_dir / "events.out.tfevents.*"))):
        ea = EventAccumulator(tf); ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            continue
        for e in ea.Scalars(tag):
            steps.append(e.step); values.append(e.value)
    if not steps:
        return None, None
    order = np.argsort(steps)
    return np.array(steps)[order].astype(float), np.array(values)[order].astype(float)


def smooth(v, w=15):
    if len(v) < w:
        return v
    return np.convolve(np.pad(v, (w//2, w//2), "edge"), np.ones(w)/w, "valid")[:len(v)]


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Optimizer comparison  (LR=1e-4, 193×130 Tenebrio)", fontsize=11)

for ax, tag, ylabel in zip(axes, ["mae", "val_loss"], ["MAE (insects)", "Val loss (MSE)"]):
    for (label, exp_dir), color in zip(RUNS.items(), COLORS):
        steps, vals = load(exp_dir, tag)
        if steps is None:
            continue
        ax.plot(steps, vals,         color=color, alpha=0.25, linewidth=0.8)
        ax.plot(steps, smooth(vals), color=color, linewidth=1.8, label=label)
    ax.set_title(tag); ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
out = ROOT / "exp/optimizer_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
