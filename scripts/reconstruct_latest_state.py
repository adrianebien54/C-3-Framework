#!/usr/bin/env python3
"""Reconstruct latest_state.pth from a weights-only epoch checkpoint.

Use when latest_state.pth is lost but an all_ep_*.pth checkpoint exists.
The optimizer momentum is reset to zero (unavoidable); all other values
are exact or accurately computed.

Usage:
    python scripts/reconstruct_latest_state.py \
        --weights exp/05-31_.../all_ep_488_mae_0.6_mse_0.9.pth \
        --epoch 488 \
        --best-mae 0.5822 \
        --best-mse 0.8550 \
        --lr 1e-5 \
        --lr-decay 0.995 \
        --weight-decay 1e-4 \
        --iters-per-epoch 14 \
        --optimizer adamw
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.CC import CrowdCounter
from config import cfg

parser = argparse.ArgumentParser()
parser.add_argument('--weights',          required=True, help='Path to all_ep_*.pth (weights-only checkpoint)')
parser.add_argument('--epoch',            required=True, type=int, help='Epoch number (1-indexed display, e.g. 488)')
parser.add_argument('--best-mae',         required=True, type=float)
parser.add_argument('--best-mse',         required=True, type=float)
parser.add_argument('--lr',               default=1e-5, type=float)
parser.add_argument('--lr-decay',         default=0.995, type=float)
parser.add_argument('--weight-decay',     default=1e-4, type=float)
parser.add_argument('--iters-per-epoch',  default=14, type=int, help='Approx training iterations per epoch (for i_tb)')
parser.add_argument('--optimizer',        default='adamw', choices=['adam', 'adamw'])
parser.add_argument('--output',           default=None, help='Output path (default: same dir as weights)')
args = parser.parse_args()

weights_path = Path(args.weights)
epoch_0idx = args.epoch - 1  # convert to 0-indexed (stored value)

# Build model on CPU
cfg.GPU_ID = [0]
net = CrowdCounter(cfg.GPU_ID, 'CSRNet')
weights = torch.load(weights_path, map_location='cpu', weights_only=True)
net.load_state_dict(weights)
print(f"Loaded weights from {weights_path.name}")

# Build optimizer (fresh — no momentum state recoverable)
params = net.CCN.parameters()
if args.optimizer == 'adam':
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
else:
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

# Build scheduler and fast-forward to correct step count
# scheduler.step() is called once per epoch before training, so after
# epoch_0idx+1 epochs (0..epoch_0idx) it has been stepped epoch+1 times
# but PyTorch StepLR tracks last_epoch starting at 0 after first step.
scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)
scheduler_state = scheduler.state_dict()
n_steps = args.epoch  # step() was called once per epoch for epochs 0..epoch-1
scheduler_state['last_epoch'] = n_steps
scheduler_state['_step_count'] = n_steps + 1
scheduler_state['_last_lr'] = [args.lr * args.lr_decay ** n_steps]
scheduler.load_state_dict(scheduler_state)

# Update optimizer LR to match the decayed value
decayed_lr = args.lr * args.lr_decay ** n_steps
for pg in optimizer.param_groups:
    pg['lr'] = decayed_lr
print(f"LR at ep {args.epoch}: {decayed_lr:.4e}  (base {args.lr} × {args.lr_decay}^{n_steps})")

# Reconstruct train_record
best_model_name = weights_path.stem  # e.g. all_ep_488_mae_0.6_mse_0.9
train_record = {
    'best_mae':        args.best_mae,
    'best_mse':        args.best_mse,
    'best_model_name': best_model_name,
}

# i_tb: TensorBoard global step for train_loss (approx, only affects plot x-axis)
i_tb = args.epoch * args.iters_per_epoch

# exp_name / exp_path inferred from weights path
weights_abs = weights_path.resolve()
exp_name = weights_abs.parent.name
exp_path = str(weights_abs.parent.parent.relative_to(ROOT))

state = {
    'net':          net.state_dict(),
    'optimizer':    optimizer.state_dict(),
    'scheduler':    scheduler.state_dict(),
    'epoch':        epoch_0idx,
    'i_tb':         i_tb,
    'train_record': train_record,
    'exp_path':     exp_path,
    'exp_name':     exp_name,
}

out = Path(args.output) if args.output else weights_path.parent / 'latest_state.pth'
torch.save(state, out)
print(f"Saved reconstructed state -> {out}")
print(f"  epoch={epoch_0idx} (resumes from ep {args.epoch + 1})")
print(f"  best_mae={args.best_mae}, best_mse={args.best_mse}")
print(f"  i_tb={i_tb}, exp_name={exp_name}")
