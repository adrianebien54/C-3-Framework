import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter


def parse_log_file(log_path):
    data = []
    pattern = r'lr=[\d.e-]+ epoch=(\d+)/\d+ train_loss=([\d.]+) val_loss=([\d.]+)'
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                data.append(
                    {
                        'epoch': int(match.group(1)),
                        'train_loss': float(match.group(2)),
                        'val_loss': float(match.group(3)),
                    }
                )
    return pd.DataFrame(data)


def apply_median_filter(series, kernel_size=51):
    return medfilt(series, kernel_size=kernel_size)


def apply_savgol(series, window=51, polyorder=3):
    return savgol_filter(series, window_length=window, polyorder=polyorder)


def compute_gap_derivative(epochs, train_series, val_series, med_kernel=51, savgol_window=51, derivative_window=9):
    train_median = apply_median_filter(train_series, kernel_size=med_kernel)
    val_median = apply_median_filter(val_series, kernel_size=med_kernel)
    
    train_filtered = apply_savgol(train_median, window=savgol_window, polyorder=3)
    val_filtered = apply_savgol(val_median, window=savgol_window, polyorder=3)

    gap = val_filtered - train_filtered
    gap_smooth = pd.Series(gap).rolling(window=derivative_window, center=True, min_periods=1).mean().values

    epoch_step = np.diff(epochs, prepend=epochs[0])
    epoch_step[0] = 1  # handle first epoch
    gap_derivative = np.diff(gap_smooth, prepend=gap_smooth[0]) / epoch_step
    gap_derivative_smooth = pd.Series(gap_derivative).rolling(window=derivative_window, center=True, min_periods=1).mean().values
    gap_acceleration = np.diff(gap_derivative_smooth, prepend=gap_derivative_smooth[0])

    return train_filtered, val_filtered, gap_smooth, gap_derivative_smooth, gap_acceleration


def detect_divergence_point(epochs, gap_derivative, positive_threshold=20):
    """Find first epoch where gap derivative stays positive for at least positive_threshold consecutive epochs."""
    positive = gap_derivative > 0
    sustained = pd.Series(positive).rolling(window=positive_threshold, min_periods=positive_threshold).sum() == positive_threshold
    idx = sustained[sustained].index
    if len(idx) == 0:
        return None
    return int(epochs[int(idx.values[0])])


def main():
    log_path = os.path.join('exp', 'tenebrio_batch1_1200ep', 'tenebrio_batch1_200ep', 'train_1e-4.log')
    csv_path = os.path.join('exp', 'tenebrio_batch1_1200ep', 'checkpoints', 'lr_1e-04.csv')

    if not os.path.exists(log_path) or not os.path.exists(csv_path):
        print(f'Required files not found')
        return

    df_log = parse_log_file(log_path)
    df_csv = pd.read_csv(csv_path)
    if df_log.empty or df_csv.empty:
        print('Could not parse training data.')
        return

    df = pd.concat([df_log, df_csv], ignore_index=True)
    df = df[(df['epoch'] >= 0) & (df['epoch'] <= 1200)].reset_index(drop=True)

    epochs = df['epoch'].values
    train = df['train_loss'].values
    val = df['val_loss'].values

    train_filtered, val_filtered, gap_smooth, gap_derivative_smooth, gap_acceleration = compute_gap_derivative(
        epochs, train, val, med_kernel=51, savgol_window=51, derivative_window=9
    )
    divergence_epoch = detect_divergence_point(
        epochs, gap_derivative_smooth, positive_threshold=20
    )

    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axs[0].plot(epochs, gap_smooth, color='C2', linewidth=2, label='gap = val - train (median + Savitzky-Golay)')
    axs[0].axhline(0.0, color='black', linewidth=1, alpha=0.6)
    if divergence_epoch is not None:
        axs[0].axvline(divergence_epoch, color='red', linestyle='--', linewidth=2,
                       label=f'divergence epoch ~ {divergence_epoch}')
    axs[0].set_ylabel('Gap', fontsize=12)
    axs[0].set_title('LR 1e-4: Epochs 0-1200 — Gap Analysis (Median + Savitzky-Golay)', fontsize=13)
    axs[0].legend(fontsize=10, loc='upper right')
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(epochs, gap_derivative_smooth, color='C3', linewidth=2,
                label='d(gap)/d(epoch) (smoothed)')
    axs[1].plot(epochs, gap_acceleration, color='C4', linewidth=1.8, alpha=0.9,
                label='acceleration of gap derivative')
    axs[1].axhline(0.0, color='black', linewidth=1, alpha=0.6)
    if divergence_epoch is not None:
        axs[1].axvline(divergence_epoch, color='red', linestyle='--', linewidth=2,
                       label=f'divergence epoch ~ {divergence_epoch}')
    axs[1].set_xlabel('Epoch', fontsize=12)
    axs[1].set_ylabel('Rate / Acceleration', fontsize=12)
    axs[1].legend(fontsize=10, loc='upper right')
    axs[1].grid(True, alpha=0.3)

    out = os.path.join('exp', 'tenebrio_batch1_1200ep', 'checkpoints',
                       'lr_1e-4_epochs0-1200_median_gap_analysis.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f'Saved plot to {out}')
    print(f'Data: {len(df)} epochs (range {epochs.min()}-{epochs.max()})')
    if divergence_epoch is not None:
        print(f'Detected divergence epoch (median-filtered): {divergence_epoch}')
    else:
        print('No sustained positive accelerating divergence point detected')


if __name__ == '__main__':
    main()
