#!/bin/bash
set -e
cd /home/umrobotics/C-3-Framework
source .venv/bin/activate

echo "[$(date)] === CSRNet Aug Set 1: Full suite (lr=1e-4, bs=6, wd=1e-4) ==="
python train.py --lr 1e-4 --batch-size 6 --weight-decay 1e-4 --aug-set 1 \
    2>&1 | tee /tmp/csrnet_772_lr1e4_bs6_aug1.log

echo "[$(date)] === CSRNet Aug Set 2: Flips only (lr=1e-4, bs=6, wd=1e-4) ==="
python train.py --lr 1e-4 --batch-size 6 --weight-decay 1e-4 --aug-set 2 \
    2>&1 | tee /tmp/csrnet_772_lr1e4_bs6_aug2.log

echo "[$(date)] === CSRNet aug runs complete ==="
