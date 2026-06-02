#!/bin/bash
# CSRNet — 386×260 σ=11 — batch-size sweep — bs=2,4,6,8 × 1200 epochs
# AdamW, lr=1e-4, wd=1e-4
# Usage: nohup bash run_csrnet_386_bssweep.sh > /tmp/csrnet_386_bssweep.log 2>&1 &
set -e
cd /home/umrobotics/C-3-Framework
source .venv/bin/activate

DATA=datasets/Tenebrio/386x260
BS_LIST="2 4 6 8"
declare -A BS_EXP  # filled during run

run_one() {
    local BS=$1
    local LOG="/tmp/csrnet_386x260_bs${BS}_1200ep.log"
    echo ""
    echo "──────────────────────────────────────────────────────"
    echo "[$(date)] CSRNet  res=386x260  lr=1e-4  bs=$BS  wd=1e-4  1200ep  σ=11"
    echo "  data: $DATA → $LOG"
    echo "──────────────────────────────────────────────────────"
    python train.py \
        --lr 1e-4 \
        --batch-size "$BS" \
        --weight-decay 1e-4 \
        --max-epoch 1200 \
        --data-path "$DATA" \
        > "$LOG" 2>&1
    echo "[$(date)] Done bs=$BS"
    EXP_DIR=$(ls -td exp/*_386x260 2>/dev/null | head -1)
    BS_EXP["$BS"]="$EXP_DIR"
    [ -n "$EXP_DIR" ] && python scripts/plot_training_curves.py "$EXP_DIR" 2>/dev/null || true
}

echo ""
echo "════════════════════════════════════════════════════════"
echo "[$(date)] CSRNet 386×260 batch-size sweep — START"
echo "Batch sizes: $BS_LIST"
echo "════════════════════════════════════════════════════════"

for BS in $BS_LIST; do run_one "$BS"; done

# Final comparison plot (all 4 batch sizes on one chart)
DIRS=(); LABELS=()
for BS in $BS_LIST; do
    [ -n "${BS_EXP[$BS]}" ] && DIRS+=("${BS_EXP[$BS]}") && LABELS+=("bs=$BS")
done
if [ ${#DIRS[@]} -gt 1 ]; then
    python scripts/compare_sigma_curves.py \
        --dirs "${DIRS[@]}" --sigmas "${LABELS[@]}" \
        --output bs_comparison_csrnet_386x260.png
    echo "→ bs_comparison_csrnet_386x260.png"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "[$(date)] ALL DONE — CSRNet 386×260 batch-size sweep"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Logs: /tmp/csrnet_386x260_bs*_1200ep.log"
