"""
Generate updated MobileCount 386x260 bbox/2 plots — 1500 epoch runs.
Individual 4-panel (rolling best) + combined rolling mean plots.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

MC_EXP_BASE = '/home/umrobotics/MobileCount/exp'

CONFIGS = [
    ('lr=1e-3', '05-29_17-28_Tenebrio_MobileCount_0.001_386x260'),
    ('lr=1e-4', '05-29_16-30_Tenebrio_MobileCount_0.0001_386x260'),
    ('lr=1e-5', '05-29_15-33_Tenebrio_MobileCount_1e-05_386x260'),
    ('lr=1e-6', '05-29_14-36_Tenebrio_MobileCount_1e-06_386x260'),
]

COLORS = {
    'lr=1e-3': '#e6194b',
    'lr=1e-4': '#3cb44b',
    'lr=1e-5': '#4363d8',
    'lr=1e-6': '#f58231',
}

WINDOW = 50

def load_mae(exp_dir):
    acc = EventAccumulator(exp_dir)
    acc.Reload()
    events = acc.Scalars('mae')
    epochs = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return epochs, values

def rolling_best(values, window=50):
    rb = np.array([values[max(0, i - window):i + 1].min() for i in range(len(values))])
    return rb

# ── Load all data ─────────────────────────────────────────────────────────────
data = {}
for label, exp_name in CONFIGS:
    exp_dir = os.path.join(MC_EXP_BASE, exp_name)
    epochs, values = load_mae(exp_dir)
    best_idx = np.argmin(values)
    data[label] = {
        'epochs': epochs,
        'values': values,
        'best_epoch': epochs[best_idx],
        'best_mae': values[best_idx],
        'rb': rolling_best(values, WINDOW),
        'cum_min': np.minimum.accumulate(values),
    }
    print(f"{label}: best MAE={values[best_idx]:.4f} @ ep {epochs[best_idx]}, total eps={epochs[-1]}")

# ── PLOT 1: Individual 4-panel ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=False)
fig.suptitle('ROLLING BEST (50 EPOCH WINDOW)', fontsize=14, fontweight='bold', y=1.01)

# Compute global y range per panel (each has its own scale since sharey=False)
for ax, (label, _) in zip(axes, CONFIGS):
    d = data[label]
    epochs = d['epochs']
    rb = d['rb']
    cum_min = d['cum_min']
    best_ep = d['best_epoch']
    best_mae = d['best_mae']
    color = COLORS[label]

    ymin = best_mae * 0.85
    ymax = np.percentile(d['values'], 95) * 1.05

    ax.plot(epochs, rb, color=color, linewidth=1.8, label='Rolling best', zorder=3)
    ax.plot(epochs, cum_min, color=color, linewidth=1.2, linestyle='--',
            alpha=0.7, label='Running best', zorder=2)
    ax.axvline(best_ep, color='black', linewidth=0.8, linestyle=':', zorder=4)

    # Label to the LEFT of dotted line
    ax.text(best_ep - (epochs[-1] * 0.02), ymin + (ymax - ymin) * 0.05,
            f'ep {best_ep}\nMAE {best_mae:.3f}',
            fontsize=8, ha='right', va='bottom', color='black')

    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, epochs[-1] + 10)

axes[0].set_ylabel('Validation MAE')

# Legend upper center on first panel
handles, labels_leg = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_leg, loc='upper center', ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, 1.0), framealpha=0.9)

plt.tight_layout()
out1 = '/home/umrobotics/C-3-Framework/mobilecount_386x260_bbox2_mae_individual.png'
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out1}')

# ── PLOT 2: All 4 LRs rolling mean on one graph ───────────────────────────────
fig2, ax2 = plt.subplots(figsize=(12, 6))

for label, _ in CONFIGS:
    d = data[label]
    epochs = d['epochs']
    values = d['values']
    color = COLORS[label]

    # raw faint
    ax2.plot(epochs, values, color=color, alpha=0.15, linewidth=0.8)

    # rolling mean
    rm = np.convolve(values, np.ones(WINDOW) / WINDOW, mode='valid')
    rm_epochs = epochs[WINDOW - 1:]
    ax2.plot(rm_epochs, rm, color=color, linewidth=2.0, label=label)

ax2.set_title('ROLLING MEAN (50 EPOCH WINDOW)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation MAE')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0)

plt.tight_layout()
out2 = '/home/umrobotics/C-3-Framework/mobilecount_386x260_bbox2_mae_alllrs.png'
fig2.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f'Saved {out2}')

# ── PLOT 3: 3-LR rolling mean (exclude lr=1e-3) ───────────────────────────────
fig3, ax3 = plt.subplots(figsize=(12, 6))

for label, _ in CONFIGS[1:]:  # skip 1e-3
    d = data[label]
    epochs = d['epochs']
    values = d['values']
    color = COLORS[label]

    ax3.plot(epochs, values, color=color, alpha=0.15, linewidth=0.8)
    rm = np.convolve(values, np.ones(WINDOW) / WINDOW, mode='valid')
    rm_epochs = epochs[WINDOW - 1:]
    ax3.plot(rm_epochs, rm, color=color, linewidth=2.0, label=label)

ax3.set_title('ROLLING MEAN (50 EPOCH WINDOW)', fontsize=13, fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Validation MAE')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0)

plt.tight_layout()
out3 = '/home/umrobotics/C-3-Framework/mobilecount_386x260_bbox2_mae_3lrs.png'
fig3.savefig(out3, dpi=150, bbox_inches='tight')
plt.close(fig3)
print(f'Saved {out3}')

print('\nAll plots done.')
