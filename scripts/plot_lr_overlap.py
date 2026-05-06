import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_csv(path):
    return pd.read_csv(path)


def smooth(series, window=11):
    return series.rolling(window=window, center=True, min_periods=1).mean()


def detect_overlap_and_diverge(train, val, rel_tol=0.05, abs_tol=1e-6):
    # compute smoothed diff
    diff = val - train
    tol = np.maximum(abs_tol, rel_tol * (np.abs(train) + 1e-12))
    overlap_mask = np.abs(diff) <= tol
    diverge_mask = diff > tol
    return overlap_mask, diverge_mask


def plot_for_lr(ax, df, lr_label):
    epochs = df['epoch']
    train = df['train_loss']
    val = df['val_loss']

    train_s = smooth(train, window=21)
    val_s = smooth(val, window=21)

    ax.plot(epochs, train_s, label=f'train lr={lr_label}', color='C0')
    ax.plot(epochs, val_s, '--', label=f'val lr={lr_label}', color='C1')

    overlap_mask, diverge_mask = detect_overlap_and_diverge(train_s, val_s, rel_tol=0.05, abs_tol=1e-6)

    # mark overlap points (where curves are effectively equal)
    ax.scatter(epochs[overlap_mask], train_s[overlap_mask], c='green', s=18, label='overlap', zorder=5)

    # mark divergence start points (first epoch where val > train in a run)
    # find runs of diverge_mask and mark the first epoch of each run
    diverge_idxs = np.where(diverge_mask)[0]
    if diverge_idxs.size > 0:
        runs = np.split(diverge_idxs, np.where(np.diff(diverge_idxs) != 1)[0] + 1)
        starts = [r[0] for r in runs if r.size > 0]
        ax.scatter(epochs.iloc[starts], val_s.iloc[starts], c='red', s=30, marker='x', label='diverge start', zorder=6)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.25)


def main():
    base = os.path.join('exp', 'tenebrio_batch1_1200ep', 'checkpoints')
    f1 = os.path.join(base, 'lr_1e-04.csv')
    f2 = os.path.join(base, 'lr_1e-05.csv')

    if not os.path.exists(f1) or not os.path.exists(f2):
        print('Required CSVs not found:')
        print(f1)
        print(f2)
        return

    df1 = load_csv(f1)
    df2 = load_csv(f2)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    plot_for_lr(axs[0], df1, '1e-4')
    plot_for_lr(axs[1], df2, '1e-5')

    fig.suptitle('Train vs Val (smoothed) — overlap (green) and divergence starts (red x)')
    out = os.path.join(base, 'lr_1e-4_1e-5_overlap_diverge.png')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150)
    print('Saved plot to', out)


if __name__ == '__main__':
    main()
