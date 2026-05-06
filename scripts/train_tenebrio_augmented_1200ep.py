#!/usr/bin/env python3
"""
Train CSRNet models on augmented Tenebrio dataset to 1200 epochs.
Uses horizontal/vertical flipped versions for 4x the training data.
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    print("=" * 80)
    print("TRAINING ON AUGMENTED TENEBRIO DATASET (4x images with H/V flips)")
    print("=" * 80)
    
    import subprocess
    
    # Base configuration
    base_cmd = [
        "python", "-u",
        "scripts/compare_tenebrio_lrs_loss_only.py",
        "--train-image-dir", "datasets/Tenebrio_augmented/train/img",
        "--val-image-dir", "datasets/Tenebrio_augmented/val/img",
        "--annotation-file", "datasets/Tenebrio_augmented/TenebrioVision_Annotations_augmented.json",
        "--epochs", "1200",
        "--batch-size", "1",
    ]
    
    learning_rates = ["1e-4", "1e-5", "1e-6", "1e-7"]
    
    for lr in learning_rates:
        output_dir = f"exp/tenebrio_batch1_1200ep_augmented/lr_{lr}"
        checkpoint_dir = f"exp/tenebrio_batch1_1200ep_augmented/checkpoints/lr_{lr}"
        
        cmd = base_cmd + [
            "--lrs", lr,
            "--output-dir", output_dir,
            "--checkpoint-dir", checkpoint_dir,
        ]
        
        print(f"\n{'='*80}")
        print(f"Training lr={lr} on augmented dataset to 1200 epochs...")
        print(f"{'='*80}")
        print(f"Command: {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd, env={**dict(__import__('os').environ), 
                                          'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'})
        
        if result.returncode != 0:
            print(f"\n❌ Training failed for lr={lr} with exit code {result.returncode}")
            sys.exit(1)
        
        print(f"\n✅ Completed lr={lr}")
    
    print("\n" + "=" * 80)
    print("✅ ALL AUGMENTED TRAINING COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Compare baseline vs augmented loss curves")
    print("2. Analyze overfitting patterns")
    print("3. Generate final convergence plots")
