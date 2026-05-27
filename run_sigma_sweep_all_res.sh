#!/bin/bash
# Full sigma sweep across all resolutions.
# For each resolution, tests 3 sigma values: bbox/4, bbox/2, and original sigma=15.
# 386x260 is SKIPPED here — that comparison runs in run_sigma_test_386.sh / sigma_s15_run.
#
# Sigma table (bbox/4 | bbox/2 | 15):
#   49x33      :  1 |  2 | 15
#   97x65      :  2 |  3 | 15
#   193x130    :  3 |  6 | 15
#   772x519    : 12 | 24 | 15
#   1544x1038  : 24 | 48 | 15
#
# Phase 1 (CPU, runs first): generate all density maps
# Phase 2 (GPU, sequential): train 100 epochs per sigma per resolution
#
# Usage: bash run_sigma_sweep_all_res.sh
set -e
cd /home/umrobotics/C-3-Framework
source .venv/bin/activate

TENEBRIO=datasets/Tenebrio

# ── Sigma lookup ──────────────────────────────────────────────────────────────
sigma_bbox4() {
    case $1 in
        49x33)     echo 1  ;;
        97x65)     echo 2  ;;
        193x130)   echo 3  ;;
        772x519)   echo 12 ;;
        1544x1038) echo 24 ;;
    esac
}
sigma_bbox2() {
    case $1 in
        49x33)     echo 2  ;;
        97x65)     echo 3  ;;
        193x130)   echo 6  ;;
        772x519)   echo 24 ;;
        1544x1038) echo 48 ;;
    esac
}

RESOLUTIONS="49x33 97x65 193x130 772x519 1544x1038"

# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Generate density maps (CPU only, no GPU needed)
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════"
echo "[$(date)] PHASE 1: Generating all density maps..."
echo "════════════════════════════════════════════════════════"

for RES in $RESOLUTIONS; do
    BASE="$TENEBRIO/$RES"
    ANN="$BASE/TenebrioVision_Annotations_${RES}.json"
    S4=$(sigma_bbox4 $RES)
    S2=$(sigma_bbox2 $RES)

    echo ""
    echo "--- $RES  (σ_bbox4=$S4  σ_bbox2=$S2  σ_orig=15) ---"

    # sigma = bbox/4
    python scripts/precompute_tenebrio_densities.py \
        --data-dir "$BASE" --annotation-file "$ANN" \
        --sigma "$S4" --output-dir "${BASE}_s${S4}" --overwrite
    echo "  σ=$S4 done"

    # sigma = bbox/2
    python scripts/precompute_tenebrio_densities.py \
        --data-dir "$BASE" --annotation-file "$ANN" \
        --sigma "$S2" --output-dir "${BASE}_s${S2}" --overwrite
    echo "  σ=$S2 done"

    # sigma = 15 (original baseline)
    python scripts/precompute_tenebrio_densities.py \
        --data-dir "$BASE" --annotation-file "$ANN" \
        --sigma 15 --output-dir "${BASE}_s15" --overwrite
    echo "  σ=15 done"

done

echo ""
echo "[$(date)] PHASE 1 complete — all density maps generated."

# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Training (GPU, sequential)
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════"
echo "[$(date)] PHASE 2: Training comparisons..."
echo "════════════════════════════════════════════════════════"

train_one() {
    local DATA_DIR=$1
    local LOG=$2
    echo "  [$(date)] Training $DATA_DIR → $LOG"
    python train.py \
        --lr 1e-4 --batch-size 1 --weight-decay 1e-4 \
        --data-path "$DATA_DIR" \
        --max-epoch 100 \
        > "$LOG" 2>&1
    echo "  [$(date)] done."
    EXP_DIR=$(ls -td exp/*_$(basename $DATA_DIR)* 2>/dev/null | head -1)
    [ -n "$EXP_DIR" ] && python scripts/plot_training_curves.py "$EXP_DIR" 2>/dev/null || true
}

for RES in $RESOLUTIONS; do
    BASE="$TENEBRIO/$RES"
    S4=$(sigma_bbox4 $RES)
    S2=$(sigma_bbox2 $RES)

    echo ""
    echo "════ $RES ════"

    train_one "${BASE}_s${S4}" "/tmp/sigma_${RES}_s${S4}.log"
    train_one "${BASE}_s${S2}" "/tmp/sigma_${RES}_s${S2}.log"
    train_one "${BASE}_s15"    "/tmp/sigma_${RES}_s15.log"

    # Per-resolution comparison plot
    EXP_S4=$(ls -td exp/*_${RES}_s${S4}* 2>/dev/null | head -1)
    EXP_S2=$(ls -td exp/*_${RES}_s${S2}* 2>/dev/null | head -1)
    EXP_S15=$(ls -td exp/*_${RES}_s15* 2>/dev/null | head -1)
    if [ -n "$EXP_S4" ] && [ -n "$EXP_S2" ] && [ -n "$EXP_S15" ]; then
        python scripts/compare_sigma_curves.py \
            --dirs "$EXP_S4" "$EXP_S2" "$EXP_S15" \
            --sigmas "σ=${S4} (bbox/4)" "σ=${S2} (bbox/2)" "σ=15 (orig)" \
            --output "sigma_comparison_${RES}.png"
        echo "  → sigma_comparison_${RES}.png"
    fi
done

echo ""
echo "[$(date)] ALL DONE."
echo "Comparison plots:"
ls sigma_comparison_*.png 2>/dev/null
