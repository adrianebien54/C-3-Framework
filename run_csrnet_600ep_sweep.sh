#!/bin/bash
# CSRNet 386x260 LR sweep — 4 LRs each to 600 epochs, fresh runs.
# Uses 1/8-scale GT (no upsample in CSRNet, no GT resize in Tenebrio.py).
# Usage: bash run_csrnet_600ep_sweep.sh
set -e
cd /home/umrobotics/C-3-Framework
source .venv/bin/activate

DATA_PATH=datasets/Tenebrio/386x260

run_one() {
    LR=$1
    OPT=${2:-adamw}
    OPT_FLAG=""
    [ "$OPT" = "adam" ] && OPT_FLAG="--optimizer adam"
    LOG=/tmp/csrnet_386_lr${LR}_${OPT}_600ep.log
    EXP_SUFFIX="386x260"
    [ "$OPT" = "adam" ] && EXP_SUFFIX="386x260_adam"

    echo ""
    echo "========================================"
    echo "[$(date)] CSRNet lr=${LR} opt=${OPT} bs=1 wd=1e-4 (600 epochs)"
    echo "========================================"

    python train.py \
        --lr "$LR" \
        --batch-size 1 \
        --weight-decay 1e-4 \
        --data-path "$DATA_PATH" \
        $OPT_FLAG \
        > "$LOG" 2>&1

    echo "[$(date)] lr=${LR} opt=${OPT} done."

    EXP_DIR=$(ls -td exp/*_"${EXP_SUFFIX}" 2>/dev/null | head -1)
    if [ -n "$EXP_DIR" ]; then
        echo "Plotting $EXP_DIR ..."
        python scripts/plot_training_curves.py "$EXP_DIR"
    fi
}

run_one 1e-6           # AdamW
run_one 1e-6 adam      # plain Adam (comparison)
run_one 1e-5           # AdamW
run_one 1e-4           # AdamW
run_one 1e-3           # AdamW

# ── Git push C-3-Framework ─────────────────────────────────────────────────
echo ""
echo "[$(date)] Committing C-3-Framework..."
git add \
    train.py \
    misc/utils.py \
    scripts/plot_training_curves.py \
    datasets/Tenebrio/Tenebrio.py \
    models/SCC_Model/CSRNet.py \
    run_csrnet_600ep_sweep.sh \
    exp/

git commit -m "CSRNet 386x260 sweep: 1/8-scale GT, no upsample, 600 epochs

Key fix: CSRNet now outputs at 1/8 resolution (removed F.upsample) and
GT density maps are kept at native 1/8 scale. This gives ~64x stronger
per-pixel gradient signal vs full-resolution MSE loss.

- models/SCC_Model/CSRNet.py: removed F.upsample(scale_factor=8)
- datasets/Tenebrio/Tenebrio.py: GT no longer upsampled to full res
- train.py: --resume / --resume-path CLI args added
- misc/utils.py: exclude datasets/exp/.venv from code snapshot
- exp/: training_curves.png for lr=1e-6, 1e-5, 1e-4, 1e-3

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git push

# ── MobileCount sweep ──────────────────────────────────────────────────────
echo ""
echo "[$(date)] Starting MobileCount LR sweep..."
cd /home/umrobotics/MobileCount
bash run_mobilecount_386_lrsweep.sh

# ── Git push MobileCount ───────────────────────────────────────────────────
echo ""
echo "[$(date)] Committing MobileCount..."
cd /home/umrobotics/MobileCount
git add \
    train.py \
    misc/utils.py \
    datasets/Tenebrio/Tenebrio.py \
    scripts/plot_training_curves.py \
    run_mobilecount_386_plateau.sh \
    run_mobilecount_386_lrsweep.sh \
    exp/

git commit -m "MobileCount 386x260 sweep: 600 epochs, 4 LRs

- train.py: --resume/--resume-path/--data-path CLI args
- misc/utils.py: exclude datasets/exp/.venv from code snapshot
- Tenebrio.py: density map normalisation fix (area_ratio)
- exp/: training_curves.png for lr=1e-6, 1e-5, 1e-4, 1e-3

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git push

echo ""
echo "[$(date)] ALL DONE."
