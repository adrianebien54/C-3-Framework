#!/bin/bash
# Extends bbox/2 overnight run from 600 → 1200 epochs for all 8 experiments.
# Resumes each run from its latest_state.pth.
#
# Chained automatically — do not launch manually until run_overnight_bbox2.sh finishes.
set -e

MC_PYTHON=/home/umrobotics/C-3-Framework/.venv/bin/python3
LR_LIST="1e-6 1e-5 1e-4 1e-3"

# Python formats LR floats in dir names like this:
declare -A LR_PAT
LR_PAT["1e-6"]="1e-06"
LR_PAT["1e-5"]="1e-05"
LR_PAT["1e-4"]="0.0001"
LR_PAT["1e-3"]="0.001"

echo ""
echo "████████████████████████████████████████████████████████"
echo "[$(date)] BBOX/2 EXTEND 600→1200 — START"
echo "████████████████████████████████████████████████████████"

# ── STEP 1: CSRNet 600→1200 ───────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "[$(date)] STEP 1/4 — CSRNet 386×260 bbox/2 extend to 1200 ep"
echo "════════════════════════════════════════════════════════"

cd /home/umrobotics/C-3-Framework
source .venv/bin/activate
declare -A CSRNET_EXP

for LR in $LR_LIST; do
    PAT="${LR_PAT[$LR]}"
    EXP_DIR=$(ls -td exp/*_CSRNet_${PAT}_386x260 2>/dev/null | grep -v "s[0-9]" | head -1)

    if [ -z "$EXP_DIR" ] || [ ! -f "$EXP_DIR/latest_state.pth" ]; then
        echo "  [WARN] No resumable exp found for CSRNet lr=$LR (looked for *_CSRNet_${PAT}_386x260)"
        continue
    fi

    LOG="/tmp/csrnet_386x260_bbox2_lr${LR}_extend1200.log"
    echo ""
    echo "──────────────────────────────────────────────────────"
    echo "[$(date)] CSRNet extend  lr=$LR  600→1200ep  $(basename $EXP_DIR)"
    echo "  → $LOG"
    echo "──────────────────────────────────────────────────────"
    python train.py \
        --lr "$LR" \
        --batch-size 1 \
        --weight-decay 1e-4 \
        --max-epoch 1200 \
        --data-path datasets/Tenebrio/386x260 \
        --resume \
        --resume-path "$EXP_DIR/latest_state.pth" \
        > "$LOG" 2>&1
    echo "[$(date)] Done lr=$LR"
    CSRNET_EXP["$LR"]="$EXP_DIR"
    python scripts/plot_training_curves.py "$EXP_DIR" 2>/dev/null || true
done

# Comparison plot
DIRS=(); LABELS=()
for LR in $LR_LIST; do
    [ -n "${CSRNET_EXP[$LR]}" ] && DIRS+=("${CSRNET_EXP[$LR]}") && LABELS+=("lr=$LR")
done
[ ${#DIRS[@]} -gt 1 ] && \
    python scripts/compare_sigma_curves.py \
        --dirs "${DIRS[@]}" --sigmas "${LABELS[@]}" \
        --output lr_comparison_csrnet_386x260_bbox2_1200ep.png && \
    echo "→ lr_comparison_csrnet_386x260_bbox2_1200ep.png"

echo "[$(date)] STEP 1 complete."

# ── STEP 2: Push CSRNet ───────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "[$(date)] STEP 2/4 — Git push C-3-Framework"
echo "════════════════════════════════════════════════════════"

git add -f lr_comparison_csrnet_386x260_bbox2_1200ep.png 2>/dev/null || true
git diff --cached --quiet && echo "Nothing new to commit." || \
    git commit -m "CSRNet 386×260 bbox/2 LR sweep — 1200 epoch results

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git push origin python3.x
echo "[$(date)] STEP 2 complete."

# ── STEP 3: MobileCount 600→1200 ─────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "[$(date)] STEP 3/4 — MobileCount 386×260 bbox/2 extend to 1200 ep"
echo "════════════════════════════════════════════════════════"

cd /home/umrobotics/MobileCount
declare -A MC_EXP

for LR in $LR_LIST; do
    PAT="${LR_PAT[$LR]}"
    EXP_DIR=$(ls -td exp/*_MobileCount_${PAT}_386x260 2>/dev/null | grep -v "s[0-9]" | head -1)

    if [ -z "$EXP_DIR" ] || [ ! -f "$EXP_DIR/latest_state.pth" ]; then
        echo "  [WARN] No resumable exp found for MobileCount lr=$LR (looked for *_MobileCount_${PAT}_386x260)"
        continue
    fi

    LOG="/tmp/mobilecount_386x260_bbox2_lr${LR}_extend1200.log"
    echo ""
    echo "──────────────────────────────────────────────────────"
    echo "[$(date)] MobileCount extend  lr=$LR  600→1200ep  $(basename $EXP_DIR)"
    echo "  → $LOG"
    echo "──────────────────────────────────────────────────────"
    $MC_PYTHON train.py \
        --lr "$LR" \
        --batch-size 6 \
        --weight-decay 1e-4 \
        --max-epoch 1200 \
        --data-path /home/umrobotics/C-3-Framework/datasets/Tenebrio/386x260 \
        --resume \
        --resume-path "$EXP_DIR/latest_state.pth" \
        > "$LOG" 2>&1
    echo "[$(date)] Done lr=$LR"
    MC_EXP["$LR"]="$EXP_DIR"
    $MC_PYTHON scripts/plot_training_curves.py "$EXP_DIR" 2>/dev/null || true
done

# Comparison plot
DIRS=(); LABELS=()
for LR in $LR_LIST; do
    [ -n "${MC_EXP[$LR]}" ] && DIRS+=("${MC_EXP[$LR]}") && LABELS+=("lr=$LR")
done
[ ${#DIRS[@]} -gt 1 ] && \
    $MC_PYTHON /home/umrobotics/C-3-Framework/scripts/compare_sigma_curves.py \
        --dirs "${DIRS[@]}" --sigmas "${LABELS[@]}" \
        --output lr_comparison_mobilecount_386x260_bbox2_1200ep.png && \
    echo "→ lr_comparison_mobilecount_386x260_bbox2_1200ep.png"

echo "[$(date)] STEP 3 complete."

# ── STEP 4: Push MobileCount ─────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "[$(date)] STEP 4/4 — Git push MobileCount"
echo "════════════════════════════════════════════════════════"

cd /home/umrobotics/MobileCount
git add -f lr_comparison_mobilecount_386x260_bbox2_1200ep.png 2>/dev/null || true
git diff --cached --quiet && echo "Nothing new to commit." || \
    git commit -m "MobileCount 386×260 bbox/2 LR sweep — 1200 epoch results

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git push origin master
echo "[$(date)] STEP 4 complete."

echo ""
echo "████████████████████████████████████████████████████████"
echo "[$(date)] BBOX/2 EXTEND 600→1200 — ALL DONE"
echo "████████████████████████████████████████████████████████"
