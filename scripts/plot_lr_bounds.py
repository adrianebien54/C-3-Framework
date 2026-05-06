import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_csv(path):
    return pd.read_csv(path)


def compute_bounds(series, window=21, k=1.0):
    mean = series.rolling(window=window, center=True, min_periods=1).mean()
    std = series.rolling(window=window, center=True, min_periods=1).std().fillna(0.0)
    lower = mean - k * std
    upper = mean + k * std
    return mean, lower, upper


def detect_overlap_and_diverge(train_m, val_m, rel_tol=0.05, abs_tol=1e-6):
    diff = val_m - train_m
    tol = np.maximum(abs_tol, rel_tol * (np.abs(train_m) + 1e-12))
    overlap_mask = np.abs(diff) <= tol
    diverge_mask = diff > tol
    return overlap_mask, diverge_mask


def plot_lr_with_bounds(ax, df, lr_label, k=1.0, window=21):
    epochs = df['epoch']
    train = df['train_loss']
    val = df['val_loss']

    train_m, train_low, train_high = compute_bounds(train, window=window, k=k)
    val_m, val_low, val_high = compute_bounds(val, window=window, k=k)

    ax.plot(epochs, train_m, label=f'train lr={lr_label}', color='C0')
    ax.fill_between(epochs, train_low, train_high, color='C0', alpha=0.18)

    ax.plot(epochs, val_m, '--', label=f'val lr={lr_label}', color='C1')
    ax.fill_between(epochs, val_low, val_high, color='C1', alpha=0.18)

    overlap_mask, diverge_mask = detect_overlap_and_diverge(train_m, val_m, rel_tol=0.05, abs_tol=1e-6)
    ax.scatter(epochs[overlap_mask], train_m[overlap_mask], c='green', s=18, label='overlap', zorder=5)

    diverge_idxs = np.where(diverge_mask)[0]
    if diverge_idxs.size > 0:
        runs = np.split(diverge_idxs, np.where(np.diff(diverge_idxs) != 1)[0] + 1)
        starts = [r[0] for r in runs if r.size > 0]
        ax.scatter(epochs.iloc[starts], val_m.iloc[starts], c='red', s=30, marker='x', label='diverge start', zorder=6)

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

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    plot_lr_with_bounds(axs[0], df1, '1e-4', k=1.0, window=21)
    plot_lr_with_bounds(axs[1], df2, '1e-5', k=1.0, window=21)

    fig.suptitle('Train/Val with rolling mean ± std bounds (k=1)')
    out = os.path.join(base, 'lr_1e-4_1e-5_bounds.png')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150)
    print('Saved bounds plot to', out)


if __name__ == '__main__':
    main()
