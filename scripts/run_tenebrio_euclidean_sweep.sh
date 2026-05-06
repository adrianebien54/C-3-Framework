#!/usr/bin/env bash
set -euo pipefail

cd /home/umrobotics/C-3-Framework
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_ROOT="exp/tenebrio_euclidean_sweep_1e6_to_1e3"
mkdir -p "$OUT_ROOT"

.venv/bin/python -u scripts/compare_tenebrio_lrs_loss_only.py \
  --lrs 1e-6 \
  --epochs 100 \
  --batch-size 1 \
  --output-dir "$OUT_ROOT/e6_100" \
  > "$OUT_ROOT/run.log" 2>&1

.venv/bin/python -u scripts/compare_tenebrio_lrs_loss_only.py \
  --lrs 1e-5 \
  --epochs 100 \
  --batch-size 1 \
  --output-dir "$OUT_ROOT/e5_100" \
  >> "$OUT_ROOT/run.log" 2>&1

.venv/bin/python -u scripts/compare_tenebrio_lrs_loss_only.py \
  --lrs 1e-4 \
  --epochs 100 \
  --batch-size 1 \
  --output-dir "$OUT_ROOT/e4_100" \
  >> "$OUT_ROOT/run.log" 2>&1

.venv/bin/python -u scripts/compare_tenebrio_lrs_loss_only.py \
  --lrs 1e-3 \
  --epochs 100 \
  --batch-size 1 \
  --output-dir "$OUT_ROOT/e3_100" \
  >> "$OUT_ROOT/run.log" 2>&1

.venv/bin/python scripts/plot_tenebrio_lr_compare.py \
  --input-dir "$OUT_ROOT/e6_100" \
  --start-epoch 6 \
  --linear-output "$OUT_ROOT/e6_100/loss_curves_linear_after_epoch5.png" \
  --log-output "$OUT_ROOT/e6_100/loss_curves_logy_after_epoch5.png" \
  >> "$OUT_ROOT/run.log" 2>&1

.venv/bin/python scripts/plot_tenebrio_lr_compare.py \
  --input-dir "$OUT_ROOT/e5_100" \
  --start-epoch 6 \
  --linear-output "$OUT_ROOT/e5_100/loss_curves_linear_after_epoch5.png" \
  --log-output "$OUT_ROOT/e5_100/loss_curves_logy_after_epoch5.png" \
  >> "$OUT_ROOT/run.log" 2>&1

.venv/bin/python scripts/plot_tenebrio_lr_compare.py \
  --input-dir "$OUT_ROOT/e4_100" \
  --start-epoch 6 \
  --linear-output "$OUT_ROOT/e4_100/loss_curves_linear_after_epoch5.png" \
  --log-output "$OUT_ROOT/e4_100/loss_curves_logy_after_epoch5.png" \
  >> "$OUT_ROOT/run.log" 2>&1

.venv/bin/python scripts/plot_tenebrio_lr_compare.py \
  --input-dir "$OUT_ROOT/e3_100" \
  --start-epoch 6 \
  --linear-output "$OUT_ROOT/e3_100/loss_curves_linear_after_epoch5.png" \
  --log-output "$OUT_ROOT/e3_100/loss_curves_logy_after_epoch5.png" \
  >> "$OUT_ROOT/run.log" 2>&1
