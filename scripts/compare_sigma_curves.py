#!/usr/bin/env python3
"""Compare MAE curves across multiple sigma experiment directories.

Usage:
    python scripts/compare_sigma_curves.py \
        --dirs exp/YYYY-MM-DD_*_386x260_s* exp/YYYY-MM-DD_*_386x260 \
        --sigmas 1 5 11 15 20 \
        --output sigma_comparison.png

If --sigmas is omitted, sigma labels are inferred from directory names
(looks for _s<number> suffix, else uses the full dir name).
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_mae(exp_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, mae_values) from all tfevents files in exp_dir."""
    tf_files = sorted(glob.glob(str(exp_dir / "events.out.tfevents.*")))
    steps, values = [], []
    for tf_file in tf_files:
        ea = EventAccumulator(tf_file)
        ea.Reload()
        if "mae" not in ea.Tags().get("scalars", []):
            continue
        for e in ea.Scalars("mae"):
            steps.append(e.step)
            values.append(e.value)
    if not steps:
        return np.array([]), np.array([])
    order = np.argsort(steps)
    return np.array(steps, dtype=float)[order], np.array(values, dtype=float)[order]


def infer_label(exp_dir: Path) -> str:
    """Guess a short label from the experiment directory name."""
    name = exp_dir.name
    # Look for _s<digits> at the end of the data-path component
    import re
    m = re.search(r'_s(\d+)(?:_|$)', name)
    if m:
        return f"σ={m.group(1)}"
    # Fallback: use the LR part of the exp name
    parts = name.split("_")
    for p in parts:
        try:
            float(p)
            return f"lr={p}"
        except ValueError:
            pass
    return name[-20:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MAE across sigma experiments.")
    parser.add_argument("--dirs", nargs="+", required=True,
                        help="Experiment directories to compare (glob patterns OK).")
    parser.add_argument("--sigmas", nargs="+", type=str, default=None,
                        help="Labels for each dir (e.g. '1 5 11 15 20'). "
                             "Inferred from dir names if omitted.")
    parser.add_argument("--output", type=str, default="sigma_comparison.png",
                        help="Output PNG path.")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Moving-average window for MAE curves (default 5).")
    args = parser.parse_args()

    # Expand globs
    exp_dirs: list[Path] = []
    for pattern in args.dirs:
        matches = sorted(glob.glob(pattern))
        if matches:
            exp_dirs.extend(Path(m) for m in matches)
        else:
            exp_dirs.append(Path(pattern))

    if not exp_dirs:
        print("No experiment directories found.")
        return

    labels = args.sigmas if args.sigmas and len(args.sigmas) == len(exp_dirs) \
             else [infer_label(d) for d in exp_dirs]

    # Colour palette
    colours = plt.cm.tab10(np.linspace(0, 0.9, len(exp_dirs)))

    fig, ax = plt.subplots(figsize=(10, 5))

    best_final: list[tuple[str, float]] = []

    for exp_dir, label, colour in zip(exp_dirs, labels, colours):
        steps, maes = load_mae(exp_dir)
        if len(steps) == 0:
            print(f"  [skip] No MAE data in {exp_dir}")
            continue

        # Raw (faint) + smoothed (bold)
        ax.plot(steps, maes, color=colour, linewidth=0.6, alpha=0.25)

        window = min(args.smooth, len(maes))
        kernel = np.ones(window) / window
        pad = np.pad(maes, (window // 2, window // 2), mode="edge")
        smoothed = np.convolve(pad, kernel, mode="valid")[:len(maes)]
        ax.plot(steps, smoothed, color=colour, linewidth=1.8, label=label)

        best_final.append((label, float(np.min(maes))))

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("MAE", fontsize=10)
    ax.set_title("MAE vs Epoch — Gaussian σ comparison (386×260, lr=1e-4)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # Annotate best MAE per sigma
    summary = "  |  ".join(f"{lbl}: best={mae:.2f}" for lbl, mae in sorted(best_final, key=lambda x: x[1]))
    fig.text(0.5, 0.01, summary, ha="center", fontsize=8, color="#444")

    out = Path(args.output)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    print("\nBest MAE per sigma:")
    for label, mae in sorted(best_final, key=lambda x: x[1]):
        print(f"  {label:>8s}  →  {mae:.3f}")


if __name__ == "__main__":
    main()
