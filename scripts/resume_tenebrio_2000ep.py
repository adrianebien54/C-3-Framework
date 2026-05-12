#!/usr/bin/env python3
"""Resume Tenebrio training from epoch 1200 for 800 more epochs (reaching 2000 total).

Resumes three models:
1. Non-augmented lr=1e-5
2. Non-augmented lr=1e-6  
3. Augmented lr=1e-6

Note: This script will merge the resumption history with existing history.
"""

import sys
import subprocess
import csv
from pathlib import Path


def merge_csv_history(existing_csv_path: Path, new_csv_path: Path, output_csv_path: Path) -> None:
    """Merge existing CSV history with new resume history."""
    all_rows = []
    
    # Read existing history
    if existing_csv_path.exists():
        with existing_csv_path.open("r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                all_rows.append(row)
    
    # Read new history from resume run
    if new_csv_path.exists():
        with new_csv_path.open("r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                all_rows.append(row)
    
    # Write merged history
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        writer.writerows(all_rows)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    python_exe = script_dir.parent / ".venv" / "bin" / "python3"
    
    configs = [
        {
            "name": "non-augmented lr=1e-5",
            "train_dir": "datasets/Tenebrio/train/img",
            "val_dir": "datasets/Tenebrio/val/img",
            "annotation": "datasets/Tenebrio/TenebrioVision_Annotations.json",
            "checkpoint": "exp/tenebrio_batch1_1200ep/checkpoints/lr_1e-5/epoch_1200.pth",
            "checkpoint_dir": "exp/tenebrio_batch1_1200ep/checkpoints/lr_1e-5",
            "existing_csv": "exp/tenebrio_batch1_1200ep/checkpoints/lr_1e-05.csv",
            "lr": "1e-05",
        },
        {
            "name": "non-augmented lr=1e-6",
            "train_dir": "datasets/Tenebrio/train/img",
            "val_dir": "datasets/Tenebrio/val/img",
            "annotation": "datasets/Tenebrio/TenebrioVision_Annotations.json",
            "checkpoint": "exp/tenebrio_batch1_1200ep/checkpoints/lr_1e-6/epoch_1200.pth",
            "checkpoint_dir": "exp/tenebrio_batch1_1200ep/checkpoints/lr_1e-6",
            "existing_csv": "exp/tenebrio_batch1_1200ep/checkpoints/lr_1e-06.csv",
            "lr": "1e-06",
        },
        {
            "name": "augmented lr=1e-6",
            "train_dir": "datasets/Tenebrio_augmented/train/img",
            "val_dir": "datasets/Tenebrio_augmented/val/img",
            "annotation": "datasets/Tenebrio_augmented/TenebrioVision_Annotations_augmented.json",
            "checkpoint": "exp/tenebrio_batch1_1200ep_augmented/checkpoints/lr_1e-6/epoch_1200.pth",
            "checkpoint_dir": "exp/tenebrio_batch1_1200ep_augmented/checkpoints/lr_1e-6",
            "existing_csv": "exp/tenebrio_batch1_1200ep_augmented/lr_1e-6/lr_1e-06.csv",
            "lr": "1e-06",
        },
    ]

    print("=" * 80)
    print("RESUMING TENEBRIO TRAINING FROM EPOCH 1200 TO 2000 (800 MORE EPOCHS)")
    print("=" * 80)
    print(f"\nEstimated time: 20-40 minutes (3 models × ~800 epochs)\n")

    for i, config in enumerate(configs, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/3] {config['name']}")
        print(f"{'='*80}")
        
        checkpoint = Path(config["checkpoint"])
        if not checkpoint.exists():
            print(f"❌ Checkpoint not found: {checkpoint}")
            print("   Skipping this config.")
            continue

        # Build resume command
        cmd = [
            str(python_exe),
            "-u",
            "scripts/resume_tenebrio_e4.py",
            "--train-image-dir", config["train_dir"],
            "--val-image-dir", config["val_dir"],
            "--annotation", config["annotation"],
            "--checkpoint", config["checkpoint"],
            "--checkpoint-dir", config["checkpoint_dir"],
            "--epochs", "2000",
        ]

        print(f"Command: {' '.join(cmd)}\n")

        result = subprocess.run(cmd, env={
            **dict(__import__('os').environ),
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'
        })

        if result.returncode != 0:
            print(f"\n❌ Training failed for {config['name']} with exit code {result.returncode}")
            sys.exit(1)

        # Merge CSV histories
        existing_csv = Path(config["existing_csv"])
        temp_csv = Path(config["checkpoint_dir"]) / f"lr_{config['lr']}.csv"
        if temp_csv.exists():
            merge_csv_history(existing_csv, temp_csv, existing_csv)
            print(f"✅ Merged CSV history to {existing_csv}")

        print(f"✅ Completed {config['name']}")

    print("\n" + "=" * 80)
    print("✅ ALL RESUME TRAINING COMPLETE!")
    print("=" * 80)
