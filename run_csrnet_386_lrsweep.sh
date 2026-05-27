#!/bin/bash
# CSRNet 386x260 LR sweep: lr=1e-5, 1e-4, 1e-3  (AdamW wd=1e-4, bs=1, plateau-based)
# Run after the lr=1e-6 run has finished.
# Usage: bash run_csrnet_386_lrsweep.sh
set -e
cd /home/umrobotics/C-3-Framework
source .venv/bin/activate

PLATEAU_WINDOW=80
OVERSHOOT=50
DATA_PATH=datasets/Tenebrio/386x260

run_one() {
    LR=$1
    LOG=/tmp/csrnet_386_lr${LR}_bs1.log

    echo ""
    echo "========================================"
    echo "[$(date)] Starting CSRNet 386x260 lr=${LR} bs=1 wd=1e-4"
    echo "========================================"

    python train.py \
        --lr "$LR" \
        --batch-size 1 \
        --weight-decay 1e-4 \
        --data-path "$DATA_PATH" \
        > "$LOG" 2>&1 &
    TRAIN_PID=$!
    echo "Training PID: $TRAIN_PID  (log: $LOG)"

    # Inline plateau monitor
    python3 - "$TRAIN_PID" "$LOG" "$PLATEAU_WINDOW" "$OVERSHOOT" <<'PYEOF'
import sys, time, re, os, signal

train_pid    = int(sys.argv[1])
log_path     = sys.argv[2]
plat_window  = int(sys.argv[3])
overshoot    = int(sys.argv[4])

best_mae         = float('inf')
best_mae_ep      = 0
plateau_start_ep = None

print(f"Monitor started (plateau_window={plat_window}, overshoot={overshoot})", flush=True)

while True:
    if not os.path.exists(f'/proc/{train_pid}'):
        print("Training process ended on its own.", flush=True)
        break

    try:
        lines = open(log_path).readlines()
    except FileNotFoundError:
        time.sleep(15)
        continue

    bests    = [l for l in lines if '[best]' in l]
    ep_lines = [l for l in lines if re.match(r'\[ep \d+\]', l)]

    if not ep_lines:
        time.sleep(15)
        continue

    cur_ep = int(re.search(r'\[ep (\d+)\]', ep_lines[-1]).group(1))

    if bests:
        m_mae = re.search(r'\[mae ([\d.]+)\]', bests[-1])
        m_ep  = re.search(r'all_ep_(\d+)',     bests[-1])
        if m_mae and m_ep:
            mae    = float(m_mae.group(1))
            bst_ep = int(m_ep.group(1))

            if mae < best_mae - 0.005:
                print(f"ep {cur_ep}: new best MAE {mae:.3f}", flush=True)
                best_mae         = mae
                best_mae_ep      = bst_ep
                plateau_start_ep = None
            elif plateau_start_ep is None:
                plateau_start_ep = bst_ep
                print(f"ep {cur_ep}: plateau clock started (best MAE {best_mae:.3f} at ep {best_mae_ep})", flush=True)

    if plateau_start_ep is not None:
        since = cur_ep - plateau_start_ep
        if since >= plat_window + overshoot:
            print(
                f"\nStopping: {overshoot} overshoot epochs complete "
                f"(plateau at ep {plateau_start_ep}, best MAE {best_mae:.3f}, "
                f"current ep {cur_ep})",
                flush=True,
            )
            os.kill(train_pid, signal.SIGTERM)
            break
        elif since == plat_window:
            print(
                f"ep {cur_ep}: plateau confirmed — running {overshoot} more epochs before stopping.",
                flush=True,
            )

    time.sleep(15)

print("Monitor done.", flush=True)
PYEOF

    wait $TRAIN_PID 2>/dev/null || true
    echo "[$(date)] Training with lr=${LR} finished."

    # Find the most-recently-modified 386x260 experiment directory for this LR
    EXP_DIR=$(ls -td exp/*_386x260 2>/dev/null | head -1)
    if [ -n "$EXP_DIR" ]; then
        echo "Saving training curves for ${EXP_DIR} ..."
        python scripts/plot_training_curves.py "${EXP_DIR}"
    else
        echo "WARNING: could not determine EXP_DIR; skipping plot."
    fi
}

run_one 1e-5
run_one 1e-4
run_one 1e-3

echo ""
echo "[$(date)] CSRNet LR sweep complete."
