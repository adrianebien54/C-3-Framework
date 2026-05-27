#!/bin/bash
# Sigma sweep: test 5 Gaussian sigma values at 386x260, 100 epochs each at lr=1e-4.
# Generates density maps with each sigma, trains CSRNet, plots individual curves,
# then produces a combined comparison plot.
#
# Usage: bash run_sigma_sweep_386.sh
# Runtime: ~5 runs x ~17 min = ~85 min total
set -e
cd /home/umrobotics/C-3-Framework
source .venv/bin/activate

ANNOTATION=datasets/Tenebrio/386x260/TenebrioVision_Annotations_386x260.json
BASE_DIR=datasets/Tenebrio/386x260

run_sigma() {
    SIGMA=$1
    echo ""
    echo "========================================"
    echo "[$(date)] sigma=${SIGMA}: generating density maps..."
    echo "========================================"

    if [ "$SIGMA" = "15" ]; then
        # Baseline: regenerate in-place so it matches the new 1/8-scale format
        DATA_DIR="$BASE_DIR"
        python scripts/precompute_tenebrio_densities.py \
            --data-dir "$DATA_DIR" \
            --annotation-file "$ANNOTATION" \
            --sigma "$SIGMA" \
            --overwrite
    else
        DATA_DIR="datasets/Tenebrio/386x260_s${SIGMA}"
        python scripts/precompute_tenebrio_densities.py \
            --data-dir "$BASE_DIR" \
            --annotation-file "$ANNOTATION" \
            --sigma "$SIGMA" \
            --output-dir "$DATA_DIR" \
            --overwrite
    fi

    # Spot-check: print shape of one CSV
    python - "$DATA_DIR" <<'PYCHECK'
import sys, pandas as pd, pathlib
data_dir = pathlib.Path(sys.argv[1])
sample = next((data_dir / "train" / "den").glob("*.csv"), None)
if sample:
    d = pd.read_csv(sample, header=None)
    print(f"  CSV shape: {d.shape}  sum={d.values.sum():.2f}")
else:
    print("  WARNING: no CSV found in train/den")
PYCHECK

    echo "[$(date)] sigma=${SIGMA}: training 100 epochs at lr=1e-4..."
    LOG="/tmp/csrnet_sigma${SIGMA}.log"

    python train.py \
        --lr 1e-4 \
        --batch-size 1 \
        --weight-decay 1e-4 \
        --data-path "$DATA_DIR" \
        --max-epoch 100 \
        > "$LOG" 2>&1

    echo "[$(date)] sigma=${SIGMA}: training done."

    # Plot individual curves
    EXP_DIR=$(ls -td exp/*_386x260* 2>/dev/null | head -1)
    if [ -n "$EXP_DIR" ]; then
        echo "  Plotting $EXP_DIR ..."
        python scripts/plot_training_curves.py "$EXP_DIR"
    fi
}

run_sigma 1
run_sigma 5
run_sigma 11
run_sigma 15
run_sigma 20

# Combined comparison plot
echo ""
echo "[$(date)] Generating comparison plot..."
python scripts/compare_sigma_curves.py \
    --dirs \
        "$(ls -td exp/*_386x260_s1   2>/dev/null | head -1)" \
        "$(ls -td exp/*_386x260_s5   2>/dev/null | head -1)" \
        "$(ls -td exp/*_386x260_s11  2>/dev/null | head -1)" \
        "$(ls -td exp/*_386x260      2>/dev/null | grep -v '_s[0-9]' | head -1)" \
        "$(ls -td exp/*_386x260_s20  2>/dev/null | head -1)" \
    --sigmas "σ=1" "σ=5" "σ=11" "σ=15 (current)" "σ=20" \
    --output sigma_comparison_386.png

echo ""
echo "[$(date)] ALL DONE. Results: sigma_comparison_386.png"
echo "Logs: /tmp/csrnet_sigma{1,5,11,15,20}.log"
