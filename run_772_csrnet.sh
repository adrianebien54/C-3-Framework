#!/bin/bash
set -e
cd /home/umrobotics/C-3-Framework
source .venv/bin/activate

echo "[$(date)] === Run 1: CSRNet 772x519 AdamW lr=1e-6 bs=1 wd=1e-4 ==="
python train.py --lr 1e-5 --batch-size 1 --weight-decay 1e-4 \
    2>&1 | tee /tmp/csrnet_772_lr1e5_bs1.log

echo "[$(date)] === Run 2: CSRNet 772x519 AdamW lr=1e-4 bs=6 wd=1e-4 ==="
python train.py --lr 1e-4 --batch-size 6 --weight-decay 1e-4 \
    2>&1 | tee /tmp/csrnet_772_lr1e4_bs6.log

echo "[$(date)] === All runs complete ==="
