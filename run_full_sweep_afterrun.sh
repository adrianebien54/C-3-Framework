#!/bin/bash
# Full sweep runner — waits for the current CSRNet lr=1e-6 run to finish,
# then: plot lr=1e-6 → CSRNet LR sweep → MobileCount LR sweep → git push both repos.
# Usage: bash run_full_sweep_afterrun.sh [plateau_runner_pid]
set -e
cd /home/umrobotics/C-3-Framework
source .venv/bin/activate

PLATEAU_PID=${1:-""}

# ── 1. Wait for the current lr=1e-6 run to finish ──────────────────────────
if [ -n "$PLATEAU_PID" ]; then
    echo "[$(date)] Waiting for plateau runner PID $PLATEAU_PID to finish..."
    while kill -0 "$PLATEAU_PID" 2>/dev/null; do
        sleep 30
    done
    echo "[$(date)] Plateau runner finished."
fi

# ── 2. Plot lr=1e-6 ────────────────────────────────────────────────────────
LR1E6_EXP=$(ls -td exp/*_386x260 2>/dev/null | grep "1e-06\|1e-6" | head -1)
if [ -n "$LR1E6_EXP" ]; then
    echo "[$(date)] Plotting lr=1e-6 experiment: $LR1E6_EXP"
    python scripts/plot_training_curves.py "$LR1E6_EXP"
else
    echo "WARNING: Could not find lr=1e-6 386x260 experiment dir; skipping plot."
fi

# ── 3. CSRNet LR sweep (1e-5, 1e-4, 1e-3) ─────────────────────────────────
echo ""
echo "[$(date)] Starting CSRNet LR sweep..."
bash run_csrnet_386_lrsweep.sh

# ── 4. MobileCount LR sweep (1e-6, 1e-5, 1e-4, 1e-3) ─────────────────────
echo ""
echo "[$(date)] Starting MobileCount LR sweep..."
cd /home/umrobotics/MobileCount
bash run_mobilecount_386_lrsweep.sh

# ── 5. Git push C-3-Framework ──────────────────────────────────────────────
echo ""
echo "[$(date)] Committing and pushing C-3-Framework..."
cd /home/umrobotics/C-3-Framework
git add \
    scripts/plot_training_curves.py \
    run_csrnet_386_lrsweep.sh \
    run_full_sweep_afterrun.sh \
    datasets/Tenebrio/Tenebrio.py \
    exp/

git commit -m "Add LR sweep results and plot script for CSRNet 386x260

- plot_training_curves.py: add CLI arg for exp_dir
- run_csrnet_386_lrsweep.sh: sequential LR sweep (1e-5, 1e-4, 1e-3)
- Tenebrio.py: density map normalisation fix (area_ratio)
- exp/: training curves PNG for lr=1e-6, 1e-5, 1e-4, 1e-3

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

git push

# ── 6. Git push MobileCount ────────────────────────────────────────────────
echo ""
echo "[$(date)] Committing and pushing MobileCount..."
cd /home/umrobotics/MobileCount
git add \
    train.py \
    datasets/Tenebrio/Tenebrio.py \
    scripts/plot_training_curves.py \
    run_mobilecount_386_plateau.sh \
    run_mobilecount_386_lrsweep.sh \
    exp/

git commit -m "Add LR sweep results and plot script for MobileCount 386x260

- train.py: add --data-path CLI arg; include resolution in EXP_NAME
- Tenebrio.py: density map normalisation fix (area_ratio)
- plot_training_curves.py: add CLI arg for exp_dir
- run_mobilecount_386_plateau.sh: single-LR plateau runner
- run_mobilecount_386_lrsweep.sh: sequential LR sweep (1e-6 to 1e-3)
- exp/: training curves PNG for lr=1e-6, 1e-5, 1e-4, 1e-3

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

git push

echo ""
echo "[$(date)] All done — full LR sweep complete, both repos pushed."
