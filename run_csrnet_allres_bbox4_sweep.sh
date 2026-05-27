#!/bin/bash
# CSRNet — bbox/4 sigma — all resolutions — 4 LRs × 600 epochs each
#
# Sigma table (bbox/4 rule):
#   49x33      → σ=1  → datasets/Tenebrio/49x33_s1
#   97x65      → σ=2  → datasets/Tenebrio/97x65_s2
#   193x130    → σ=3  → datasets/Tenebrio/193x130_s3
#   386x260    → σ=6  → datasets/Tenebrio/386x260_s6
#   772x519    → σ=12 → datasets/Tenebrio/772x519_s12
#   1544x1038  → σ=24 → datasets/Tenebrio/1544x1038_s24
#
# Total: 6 resolutions × 4 LRs = 24 sequential jobs.
#
# Usage: nohup bash run_csrnet_allres_bbox4_sweep.sh > /tmp/csrnet_allres_bbox4.log 2>&1 &
set -e
cd /home/umrobotics/C-3-Framework
source .venv/bin/activate

TENEBRIO=datasets/Tenebrio

declare -A BBOX4_DIR
BBOX4_DIR[49x33]="49x33_s1"
BBOX4_DIR[97x65]="97x65_s2"
BBOX4_DIR[193x130]="193x130_s3"
BBOX4_DIR[386x260]="386x260_s6"
BBOX4_DIR[772x519]="772x519_s12"
BBOX4_DIR[1544x1038]="1544x1038_s24"

RESOLUTIONS="49x33 97x65 193x130 386x260 772x519 1544x1038"
LR_LIST="1e-6 1e-5 1e-4 1e-3"

run_one() {
    local DATA_DIR=$1
    local LR=$2
    local RES=$3
    local SUFFIX=$(basename "$DATA_DIR")
    local LOG="/tmp/csrnet_${SUFFIX}_lr${LR}_600ep.log"

    echo ""
    echo "──────────────────────────────────────────────────────"
    echo "[$(date)] CSRNet  res=$RES  lr=$LR  bs=1  wd=1e-4  600ep"
    echo "  data: $DATA_DIR → $LOG"
    echo "──────────────────────────────────────────────────────"

    python train.py \
        --lr "$LR" \
        --batch-size 1 \
        --weight-decay 1e-4 \
        --data-path "$DATA_DIR" \
        > "$LOG" 2>&1

    echo "[$(date)] done."

    # Plot individual curves for most recent exp matching this data suffix
    EXP_DIR=$(ls -td exp/*_"${SUFFIX}" 2>/dev/null | head -1)
    [ -n "$EXP_DIR" ] && python scripts/plot_training_curves.py "$EXP_DIR" 2>/dev/null || true
}

echo ""
echo "════════════════════════════════════════════════════════"
echo "[$(date)] CSRNet bbox/4 all-resolution sweep — START"
echo "Resolutions: $RESOLUTIONS"
echo "LRs: $LR_LIST"
echo "════════════════════════════════════════════════════════"

for RES in $RESOLUTIONS; do
    DIR="${TENEBRIO}/${BBOX4_DIR[$RES]}"

    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  Resolution: $RES  ($(basename $DIR))"
    echo "╚══════════════════════════════════════════════════╝"

    for LR in $LR_LIST; do
        run_one "$DIR" "$LR" "$RES"
    done

    echo ""
    echo "[$(date)] ✓ $RES complete — all LRs done."
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "[$(date)] ALL DONE — CSRNet bbox/4 all-resolution sweep"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Per-resolution best MAE (check training_curves.png in each exp/):"
for RES in $RESOLUTIONS; do
    SUFFIX="${BBOX4_DIR[$RES]}"
    echo "  $RES:"
    ls -td exp/*_"${SUFFIX}" 2>/dev/null | head -4 | sed "s/^/    /"
done
echo ""
echo "Logs: /tmp/csrnet_*_600ep.log"
