#!/usr/bin/env python3

"""Generate zoomed linear loss plots for augmented Tenebrio experiment.

Creates individual plots for each learning rate, zoomed to a specified epoch range.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def _lr_sort_key(csv_path: Path) -> tuple[float, str]:
    match = re.search(r"lr_([0-9eE+\-.]+)$", csv_path.stem)
    if match is None:
        return (float("inf"), csv_path.as_posix())
    try:
        return (float(match.group(1)), csv_path.as_posix())
    except ValueError:
        return (float("inf"), csv_path.as_posix())


def _discover_csv_paths(input_dir: Path) -> list[Path]:
    """Find lr_*.csv files, first in flat directory, then recursively."""
    direct_paths = sorted(input_dir.glob("lr_*.csv"), key=_lr_sort_key)
    if direct_paths:
        return direct_paths
    return sorted(input_dir.rglob("lr_*.csv"), key=_lr_sort_key)


def read_history(csv_path: Path, start_epoch: int | None = None):
    """Read history from CSV, optionally filtering by start epoch."""
    rows = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epoch = int(row["epoch"])
            if start_epoch is not None and epoch < start_epoch:
                continue
            rows.append(
                (
                    epoch,
                    float(row["train_loss"]),
                    float(row["val_loss"]),
                )
            )
    return rows


def save_individual_plot(
    csv_path: Path,
    output_path: Path,
    start_epoch: int | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    """Save individual plot for a single learning rate."""
    import matplotlib.pyplot as plt

    lr_label = csv_path.stem.replace("lr_", "")
    history = read_history(csv_path, start_epoch)

    if not history:
        print(f"  ⚠ No data for {csv_path.stem}")
        return

    epochs = [row[0] for row in history]
    train_losses = [row[1] for row in history]
    val_losses = [row[2] for row in history]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, train_losses, label=f"train lr={lr_label}", linewidth=2)
    ax.plot(epochs, val_losses, label=f"val lr={lr_label}", linestyle="--", linewidth=2)

    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (MSE)", fontsize=12)
    
    title = f"Tenebrio CSRNet Augmented: LR={lr_label} Linear Loss"
    if start_epoch is not None:
        title += f" (Epochs {start_epoch}+)"
    ax.set_title(title, fontsize=13)
    
    ax.grid(alpha=0.2)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate individual zoomed linear plots for augmented Tenebrio experiment."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("exp/tenebrio_batch1_1200ep_augmented"),
        help="Directory containing augmented experiment data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots. Defaults to <input-dir>/plots_linear_zoom/",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=200,
        help="Start epoch for zoom. Defaults to 200.",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Optional lower y-axis bound.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional upper y-axis bound.",
    )

    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    output_dir = args.output_dir or (args.input_dir / "plots_linear_zoom")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = _discover_csv_paths(args.input_dir)
    if not csv_paths:
        raise RuntimeError(f"No lr_*.csv files found in {args.input_dir}")

    print(f"\nGenerating zoomed linear plots for {len(csv_paths)} learning rates...")
    print(f"Start epoch: {args.start_epoch}")
    print(f"Output directory: {output_dir}\n")

    for csv_path in csv_paths:
        lr_label = csv_path.stem.replace("lr_", "")
        output_path = output_dir / f"loss_curves_linear_zoom_augmented_lr_{lr_label}.png"
        save_individual_plot(
            csv_path,
            output_path,
            start_epoch=args.start_epoch,
            y_min=args.y_min,
            y_max=args.y_max,
        )

    print(f"\n✅ All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
