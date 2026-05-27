#!/bin/bash
# Run CSRNet on 386x260 with lr=1e-6, bs=1, AdamW wd=1e-4
# Stops automatically when MAE plateaus + 50 overshoot epochs.
set -e
cd /home/umrobotics/C-3-Framework
source .venv/bin/activate

LOG=/tmp/csrnet_386_lr1e6_bs1.log
PLATEAU_WINDOW=80   # epochs with no improvement = plateau
OVERSHOOT=50        # continue this many epochs after plateau before stopping

echo "[$(date)] Starting CSRNet 386x260 lr=1e-6 bs=1 wd=1e-4"

python train.py \
    --lr 1e-6 \
    --batch-size 1 \
    --weight-decay 1e-4 \
    --data-path datasets/Tenebrio/386x260 \
    > "$LOG" 2>&1 &
TRAIN_PID=$!
echo "Training PID: $TRAIN_PID  (log: $LOG)"

# Plateau monitor
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
    # Stop if training process is gone
    if not os.path.exists(f'/proc/{train_pid}'):
        print("Training process ended on its own.", flush=True)
        break

    try:
        lines = open(log_path).readlines()
    except FileNotFoundError:
        time.sleep(15)
        continue

    bests   = [l for l in lines if '[best]' in l]
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
                plateau_start_ep = None   # reset plateau clock
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
echo "[$(date)] Done."
