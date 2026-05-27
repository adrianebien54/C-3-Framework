#!/bin/bash
# CSRNet — 386×260 bbox/4 (σ=6) — LR sweep — 4 LRs × 600 epochs
# AdamW (default), wd=1e-4, bs=1
# Usage: nohup bash run_csrnet_386_bbox4_lrsweep.sh > /tmp/csrnet_386_bbox4_lrsweep.log 2>&1 &
set -e
cd /home/umrobotics/C-3-Framework
source .venv/bin/activate

DATA=datasets/Tenebrio/386x260_s6
LR_LIST="1e-6 1e-5 1e-4 1e-3"
declare -A LR_EXP  # filled during run

run_one() {
    local LR=$1
    local LOG="/tmp/csrnet_386x260_s6_lr${LR}_600ep.log"
    echo ""
    echo "──────────────────────────────────────────────────────"
    echo "[$(date)] CSRNet  res=386x260  lr=$LR  bs=1  wd=1e-4  600ep  σ=6 (bbox/4)"
    echo "  data: $DATA → $LOG"
    echo "──────────────────────────────────────────────────────"
    python train.py \
        --lr "$LR" \
        --batch-size 1 \
        --weight-decay 1e-4 \
        --max-epoch 600 \
        --data-path "$DATA" \
        > "$LOG" 2>&1
    echo "[$(date)] Done lr=$LR"
    EXP_DIR=$(ls -td exp/*_386x260_s6* 2>/dev/null | head -1)
    LR_EXP["$LR"]="$EXP_DIR"
    [ -n "$EXP_DIR" ] && python scripts/plot_training_curves.py "$EXP_DIR" 2>/dev/null || true
}

echo ""
echo "════════════════════════════════════════════════════════"
echo "[$(date)] CSRNet 386×260 bbox/4 LR sweep — START"
echo "LRs: $LR_LIST"
echo "════════════════════════════════════════════════════════"

for LR in $LR_LIST; do run_one "$LR"; done

# Final comparison plot (all 4 LRs on one chart)
DIRS=(); LABELS=()
for LR in $LR_LIST; do
    [ -n "${LR_EXP[$LR]}" ] && DIRS+=("${LR_EXP[$LR]}") && LABELS+=("lr=$LR")
done
if [ ${#DIRS[@]} -gt 1 ]; then
    python scripts/compare_sigma_curves.py \
        --dirs "${DIRS[@]}" --sigmas "${LABELS[@]}" \
        --output lr_comparison_csrnet_386x260_s6.png
    echo "→ lr_comparison_csrnet_386x260_s6.png"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "[$(date)] ALL DONE — CSRNet 386×260 bbox/4 LR sweep"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Logs: /tmp/csrnet_386x260_s6_lr*_600ep.log"
