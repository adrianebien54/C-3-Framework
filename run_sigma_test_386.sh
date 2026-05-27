#!/bin/bash
# Quick sigma test: bbox/4 (σ=6) vs bbox/2 (σ=11) at 386×260, 100 epochs each at lr=1e-4.
# σ=11 density maps are already in datasets/Tenebrio/386x260 (generated earlier).
# σ=6  density maps are generated into datasets/Tenebrio/386x260_s6.
set -e
cd /home/umrobotics/C-3-Framework
source .venv/bin/activate

ANNOTATION=datasets/Tenebrio/386x260/TenebrioVision_Annotations_386x260.json
BASE_DIR=datasets/Tenebrio/386x260

# ── 1. Generate σ=6 density maps ──────────────────────────────────────────────
echo ""
echo "[$(date)] Generating σ=6 density maps (bbox/4)..."
python scripts/precompute_tenebrio_densities.py \
    --data-dir "$BASE_DIR" \
    --annotation-file "$ANNOTATION" \
    --sigma 6 \
    --output-dir datasets/Tenebrio/386x260_s6 \
    --overwrite

# Spot-check
python - <<'PYCHECK'
import pandas as pd, pathlib
f = next(pathlib.Path("datasets/Tenebrio/386x260_s6/train/den").glob("*.csv"))
d = pd.read_csv(f, header=None).values
print(f"  σ=6  CSV: shape={d.shape}  sum={d.sum():.2f}  max={d.max():.4f}")
PYCHECK

# ── 2. Train σ=6 (100 epochs, lr=1e-4) ───────────────────────────────────────
echo ""
echo "[$(date)] Training σ=6 (bbox/4) — 100 epochs..."
python train.py \
    --lr 1e-4 --batch-size 1 --weight-decay 1e-4 \
    --data-path datasets/Tenebrio/386x260_s6 \
    --max-epoch 100 \
    > /tmp/csrnet_s6.log 2>&1
echo "[$(date)] σ=6 done."
EXP_S6=$(ls -td exp/*_386x260_s6* 2>/dev/null | head -1)
[ -n "$EXP_S6" ] && python scripts/plot_training_curves.py "$EXP_S6"

# ── 3. Train σ=11 (100 epochs, lr=1e-4) ──────────────────────────────────────
echo ""
echo "[$(date)] Training σ=11 (bbox/2) — 100 epochs..."
python train.py \
    --lr 1e-4 --batch-size 1 --weight-decay 1e-4 \
    --data-path "$BASE_DIR" \
    --max-epoch 100 \
    > /tmp/csrnet_s11.log 2>&1
echo "[$(date)] σ=11 done."
EXP_S11=$(ls -td exp/*_386x260 2>/dev/null | grep -v '_s[0-9]' | head -1)
[ -n "$EXP_S11" ] && python scripts/plot_training_curves.py "$EXP_S11"

# ── 4. Comparison plot ────────────────────────────────────────────────────────
echo ""
echo "[$(date)] Generating comparison plot..."
EXP_S6=$(ls -td exp/*_386x260_s6* 2>/dev/null | head -1)
EXP_S11=$(ls -td exp/*_386x260 2>/dev/null | grep -v '_s[0-9]' | head -1)
python scripts/compare_sigma_curves.py \
    --dirs "$EXP_S6" "$EXP_S11" \
    --sigmas "σ=6 (bbox/4)" "σ=11 (bbox/2)" \
    --output sigma_comparison_386.png

echo ""
echo "[$(date)] DONE. See sigma_comparison_386.png"
echo "  σ=6  log: /tmp/csrnet_s6.log"
echo "  σ=11 log: /tmp/csrnet_s11.log"
