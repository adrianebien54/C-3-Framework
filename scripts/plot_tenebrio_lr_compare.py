#!/usr/bin/env python3

"""Plot Tenebrio LR comparison curves from existing CSV outputs.

This script reads files named lr_*.csv from a comparison output directory and
writes both linear and log-scale plots.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LR comparison curves from CSV files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing lr_*.csv files. Nested lr_*.csv files are supported when needed.",
    )
    parser.add_argument(
        "--linear-output",
        type=Path,
        default=None,
        help="Output path for linear-scale plot. Defaults to <input-dir>/loss_curves_linear.png",
    )
    parser.add_argument(
        "--log-output",
        type=Path,
        default=None,
        help="Output path for log-scale plot. Defaults to <input-dir>/loss_curves_logy.png",
    )
    parser.add_argument(
        "--drop-first-epoch",
        action="store_true",
        help="Exclude epoch 1 from plotted curves.",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=None,
        help="Only plot epochs >= this value (applied after --drop-first-epoch).",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Optional lower y-axis bound for both plots.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional upper y-axis bound for both plots.",
    )
    return parser.parse_args()


def _lr_sort_key(csv_path: Path) -> tuple[float, str]:
    match = re.search(r"lr_([0-9eE+\-.]+)$", csv_path.stem)
    if match is None:
        return (float("inf"), csv_path.as_posix())
    try:
        return (float(match.group(1)), csv_path.as_posix())
    except ValueError:
        return (float("inf"), csv_path.as_posix())


def _discover_csv_paths(input_dir: Path) -> list[Path]:
    direct_paths = sorted(input_dir.glob("lr_*.csv"), key=_lr_sort_key)
    if direct_paths:
        return direct_paths
    return sorted(input_dir.rglob("lr_*.csv"), key=_lr_sort_key)


def read_histories(input_dir: Path, drop_first_epoch: bool, start_epoch: int | None):
    histories = {}
    for csv_path in _discover_csv_paths(input_dir):
        lr_label = csv_path.stem.replace("lr_", "")
        rows = []
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(
                    (
                        int(row["epoch"]),
                        float(row["train_loss"]),
                        float(row["val_loss"]),
                    )
                )
        if drop_first_epoch:
            rows = [entry for entry in rows if entry[0] > 1]
        if start_epoch is not None:
            rows = [entry for entry in rows if entry[0] >= start_epoch]
        if rows:
            histories[lr_label] = rows
    return histories


def save_plot(path: Path, histories, log_scale: bool, y_min: float | None, y_max: float | None) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for lr_label, history in histories.items():
        epochs = [row[0] for row in history]
        train_losses = [max(row[1], 1e-12) for row in history]
        val_losses = [max(row[2], 1e-12) for row in history]
        plt.plot(epochs, train_losses, label=f"train lr={lr_label}", linestyle="-")
        plt.plot(epochs, val_losses, label=f"val lr={lr_label}", linestyle="--")

    if log_scale:
        plt.yscale("log")

    if y_min is not None or y_max is not None:
        lower = y_min
        upper = y_max
        if log_scale and lower is not None and lower <= 0:
            lower = 1e-12
        plt.ylim(bottom=lower, top=upper)

    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Tenebrio CSRNet: Train/Val Loss by Learning Rate")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    histories = read_histories(args.input_dir, args.drop_first_epoch, args.start_epoch)
    if not histories:
        raise RuntimeError(f"No lr_*.csv files found in {args.input_dir}")

    linear_output = args.linear_output or (args.input_dir / "loss_curves_linear.png")
    log_output = args.log_output or (args.input_dir / "loss_curves_logy.png")

    save_plot(linear_output, histories, log_scale=False, y_min=args.y_min, y_max=args.y_max)
    save_plot(log_output, histories, log_scale=True, y_min=args.y_min, y_max=args.y_max)

    print(f"Saved linear plot to {linear_output}")
    print(f"Saved log-scale plot to {log_output}")


if __name__ == "__main__":
    main()