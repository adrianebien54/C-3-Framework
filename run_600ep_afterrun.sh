#!/bin/bash
# Sequential runner: waits for running lr=1e-4 to finish, then resumes
# lr=1e-5, lr=1e-6, lr=1e-3 to epoch 600. Then runs MobileCount sweep.
# Finally git-pushes both repos.
# Usage: bash run_600ep_afterrun.sh <lr1e4_train_pid>
set -e
cd /home/umrobotics/C-3-Framework
source .venv/bin/activate

LR1E4_PID=${1:?"Usage: $0 <lr=1e-4 training PID>"}
EXP_BASE=/home/umrobotics/C-3-Framework/exp

# ── Helper: plot the most-recent 386x260 exp ──────────────────────────────
plot_latest() {
    EXP_DIR=$(ls -td "$EXP_BASE"/*_386x260 2>/dev/null | head -1)
    if [ -n "$EXP_DIR" ]; then
        echo "Plotting $EXP_DIR ..."
        python scripts/plot_training_curves.py "$EXP_DIR"
    else
        echo "WARNING: no 386x260 exp dir found; skipping plot."
    fi
}

# ── 1. Wait for lr=1e-4 to finish ────────────────────────────────────────
echo "[$(date)] Waiting for lr=1e-4 training (PID $LR1E4_PID) to finish..."
while kill -0 "$LR1E4_PID" 2>/dev/null; do sleep 30; done
echo "[$(date)] lr=1e-4 done."
plot_latest

# ── 2. Resume lr=1e-5 (ep 400 → 600) ────────────────────────────────────
echo ""
echo "[$(date)] Resuming CSRNet lr=1e-5 from ep 400..."
python train.py \
    --resume \
    --resume-path "$EXP_BASE/05-22_19-47_Tenebrio_CSRNet_1e-05_386x260/latest_state.pth"
echo "[$(date)] lr=1e-5 done."
plot_latest

# ── 3. Resume lr=1e-6 (ep 485 → 600) ────────────────────────────────────
echo ""
echo "[$(date)] Resuming CSRNet lr=1e-6 from ep 485..."
python train.py \
    --resume \
    --resume-path "$EXP_BASE/05-22_18-16_Tenebrio_CSRNet_1e-06_386x260/latest_state.pth"
echo "[$(date)] lr=1e-6 done."
plot_latest

# ── 4. Resume lr=1e-3 (ep 137 → 600) ────────────────────────────────────
echo ""
echo "[$(date)] Resuming CSRNet lr=1e-3 from ep 137..."
python train.py \
    --resume \
    --resume-path "$EXP_BASE/05-26_12-43_Tenebrio_CSRNet_0.001_386x260/latest_state.pth"
echo "[$(date)] lr=1e-3 done."
plot_latest

# ── 5. MobileCount sweep (all 4 LRs, fresh, 600 epochs each) ─────────────
echo ""
echo "[$(date)] Starting MobileCount LR sweep..."
cd /home/umrobotics/MobileCount
bash run_mobilecount_386_lrsweep.sh

# ── 6. Git push C-3-Framework ─────────────────────────────────────────────
echo ""
echo "[$(date)] Committing C-3-Framework..."
cd /home/umrobotics/C-3-Framework
git add \
    train.py \
    misc/utils.py \
    scripts/plot_training_curves.py \
    datasets/Tenebrio/Tenebrio.py \
    run_csrnet_386_lrsweep.sh \
    run_csrnet_386_plateau.sh \
    run_full_sweep_afterrun.sh \
    run_600ep_afterrun.sh \
    exp/

git commit -m "CSRNet 386x260 LR sweep to 600 epochs + resume support

- train.py: add --resume / --resume-path CLI args
- misc/utils.py: exclude datasets/exp/.venv from code snapshot
- All four LRs (1e-6, 1e-5, 1e-4, 1e-3) run to 600 epochs
- exp/: training_curves.png for each LR

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git push

# ── 7. Git push MobileCount ───────────────────────────────────────────────
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

git commit -m "MobileCount 386x260 LR sweep to 600 epochs + resume support

- train.py: add --resume / --resume-path CLI args, --data-path
- misc/utils.py: exclude datasets/exp/.venv from code snapshot
- Tenebrio.py: density map normalisation fix (area_ratio)
- All four LRs (1e-6, 1e-5, 1e-4, 1e-3) run to 600 epochs
- exp/: training_curves.png for each LR

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git push

echo ""
echo "[$(date)] ALL DONE."
