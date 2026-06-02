#!/usr/bin/env python3
"""Find the most reliable best checkpoint for an experiment using smoothed MAE.

Raw per-epoch MAE is noisy. This script smooths it with a rolling window and
finds the epoch with the minimum smoothed MAE, then matches it to the nearest
saved .pth checkpoint file.
"""

import argparse
import glob
import re
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]

_parser = argparse.ArgumentParser()
_parser.add_argument("exp_dir", help="Path to experiment directory (absolute or relative to repo root).")
_parser.add_argument("--window", type=int, default=20, help="Smoothing window in epochs (default: 20).")
_args = _parser.parse_args()

_p = Path(_args.exp_dir)
EXP_DIR = _p if _p.is_absolute() else ROOT / _p


def load_tag(exp_dir, tag):
    tf_files = sorted(glob.glob(str(exp_dir / "events.out.tfevents.*")))
    steps, values = [], []
    for tf_file in tf_files:
        ea = EventAccumulator(tf_file)
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            continue
        for e in ea.Scalars(tag):
            steps.append(e.step)
            values.append(e.value)
    order = np.argsort(steps)
    return np.array(steps, dtype=float)[order], np.array(values, dtype=float)[order]


def smooth(values, window):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    pad = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(pad, kernel, mode="valid")[: len(values)]


def parse_checkpoints(exp_dir):
    """Return dict of {epoch: Path} from all_ep_*.pth files."""
    ckpts = {}
    for f in exp_dir.glob("all_ep_*.pth"):
        m = re.match(r"all_ep_(\d+)_", f.name)
        if m:
            ckpts[int(m.group(1))] = f
    return ckpts


def main():
    mae_steps, mae_vals = load_tag(EXP_DIR, "mae")
    vl_steps, vl_vals = load_tag(EXP_DIR, "val_loss")

    smoothed = smooth(mae_vals, _args.window)
    best_idx = int(np.argmin(smoothed))
    best_epoch = int(mae_steps[best_idx])
    best_smoothed_mae = smoothed[best_idx]
    raw_mae_at_best = mae_vals[best_idx]

    # val_loss at best epoch
    vl_at_best = float("nan")
    vl_idx = np.where(vl_steps == best_epoch)[0]
    if len(vl_idx):
        vl_at_best = vl_vals[vl_idx[0]]

    ckpts = parse_checkpoints(EXP_DIR)
    if ckpts:
        nearest_epoch = min(ckpts.keys(), key=lambda e: abs(e - best_epoch))
        nearest_ckpt = ckpts[nearest_epoch]
    else:
        nearest_epoch, nearest_ckpt = None, None

    print(f"Experiment : {EXP_DIR.name}")
    print(f"Smooth window : {_args.window} epochs")
    print(f"Best epoch (smoothed MAE) : {best_epoch}")
    print(f"  Smoothed MAE : {best_smoothed_mae:.4f}")
    print(f"  Raw MAE      : {raw_mae_at_best:.4f}")
    print(f"  Val loss     : {vl_at_best:.4f}")
    if nearest_ckpt:
        print(f"Nearest checkpoint : {nearest_ckpt.name}  (epoch {nearest_epoch})")
    else:
        print("No .pth checkpoints found in experiment directory.")

    # Also print top-5 smoothed epochs for context
    print()
    print("Top 5 epochs by smoothed MAE:")
    top5 = np.argsort(smoothed)[:5]
    for i in top5:
        ep = int(mae_steps[i])
        vl = float("nan")
        vi = np.where(vl_steps == ep)[0]
        if len(vi):
            vl = vl_vals[vi[0]]
        print(f"  epoch {ep:4d}  smoothed_mae={smoothed[i]:.4f}  raw_mae={mae_vals[i]:.4f}  val_loss={vl:.4f}")


if __name__ == "__main__":
    main()
