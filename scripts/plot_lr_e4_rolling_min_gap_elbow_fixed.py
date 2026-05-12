import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


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


def apply_rolling_min(series, window=51):
    series = pd.Series(series)
    return series.rolling(window=window, center=True, min_periods=1).min()


def apply_savgol(series, window=51, polyorder=3):
    return savgol_filter(series, window_length=window, polyorder=polyorder)


def normalize_to_01(series):
    """Normalize series to 0-1 range."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return np.zeros_like(series)
    return (series - min_val) / (max_val - min_val)


def compute_smoothed_curves(train_series, val_series, rolling_window=51, savgol_window=51):
    train_min = apply_rolling_min(train_series, window=rolling_window)
    val_min = apply_rolling_min(val_series, window=rolling_window)
    
    train_filtered = apply_savgol(train_min.values, window=savgol_window, polyorder=3)
    val_filtered = apply_savgol(val_min.values, window=savgol_window, polyorder=3)
    
    return train_filtered, val_filtered


def find_elbow_point(epochs, gap, derivative_window=9):
    """Find elbow point where second derivative of gap is maximized."""
    gap_smooth = pd.Series(gap).rolling(window=derivative_window, center=True, min_periods=1).mean().values
    
    # First derivative
    first_deriv = np.diff(gap_smooth, prepend=gap_smooth[0])
    
    # Second derivative
    second_deriv = np.diff(first_deriv, prepend=first_deriv[0])
    second_deriv_smooth = pd.Series(second_deriv).rolling(window=derivative_window, center=True, min_periods=1).mean().values
    
    # Find maximum of second derivative (sharpest point of curvature)
    max_idx = np.argmax(second_deriv_smooth)
    elbow_epoch = int(epochs[max_idx])
    max_curvature = second_deriv_smooth[max_idx]
    
    return elbow_epoch, second_deriv_smooth, max_curvature


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
    df = df[(df['epoch'] >= 50) & (df['epoch'] <= 1200)].reset_index(drop=True)

    epochs = df['epoch'].values
    train = df['train_loss'].values
    val = df['val_loss'].values

    # Apply rolling minimum + Savitzky-Golay smoothing
    train_filtered, val_filtered = compute_smoothed_curves(train, val, rolling_window=30, savgol_window=51)

    # Normalize to 0-1
    train_norm = normalize_to_01(pd.Series(train_filtered))
    val_norm = normalize_to_01(pd.Series(val_filtered))

    # Calculate normalized gap
    gap = val_norm - train_norm

    # Find elbow point
    elbow_epoch, second_deriv_smooth, max_curvature = find_elbow_point(epochs, gap.values, derivative_window=9)

    # Create plot
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: Normalized curves
    axs[0].plot(epochs, train_norm, label='train loss (normalized, rolling min w30 + Savitzky-Golay)', color='C0', linewidth=2)
    axs[0].plot(epochs, val_norm, label='val loss (normalized, rolling min w30 + Savitzky-Golay)', color='C1', linewidth=2, linestyle='--')
    axs[0].set_ylabel('Normalized Loss (0-1)', fontsize=12)
    axs[0].set_title('LR 1e-4: Epochs 50-1200 — Normalized Rolling Min Curves', fontsize=13)
    axs[0].legend(fontsize=10, loc='upper right')
    axs[0].grid(True, alpha=0.3)

    # Panel 2: Gap
    axs[1].plot(epochs, gap, color='C2', linewidth=2, label='Gap = Val_norm - Train_norm')
    axs[1].axhline(0.0, color='black', linewidth=1, alpha=0.6)
    axs[1].axvline(elbow_epoch, color='red', linestyle='--', linewidth=2.5, label=f'elbow epoch ~ {elbow_epoch}')
    axs[1].set_ylabel('Gap', fontsize=12)
    axs[1].legend(fontsize=10, loc='upper right')
    axs[1].grid(True, alpha=0.3)

    # Panel 3: Second derivative (curvature)
    axs[2].plot(epochs, second_deriv_smooth, color='C4', linewidth=2, label='d²(Gap)/d(epoch)² (curvature)')
    axs[2].axhline(0.0, color='black', linewidth=1, alpha=0.6)
    axs[2].axvline(elbow_epoch, color='red', linestyle='--', linewidth=2.5, 
                   label=f'max curvature at epoch {elbow_epoch}')
    axs[2].set_xlabel('Epoch', fontsize=12)
    axs[2].set_ylabel('Second Derivative', fontsize=12)
    axs[2].legend(fontsize=10, loc='upper right')
    axs[2].grid(True, alpha=0.3)

    out = os.path.join('exp', 'tenebrio_batch1_1200ep', 'checkpoints',
                       'lr_1e-4_epochs50-1200_rolling_min_normalized_gap_elbow.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f'Saved plot to {out}')
    print(f'Data: {len(df)} epochs (range {epochs.min()}-{epochs.max()})')
    print(f'\nRolling Minimum Normalized Gap Analysis:')
    print(f'  Elbow point (max gap curvature): epoch {elbow_epoch}')
    print(f'  Max curvature value: {max_curvature:.6f}')
    gap_at_elbow_idx = np.where(epochs == elbow_epoch)[0]
    if len(gap_at_elbow_idx) > 0:
        print(f'  Gap at elbow: {gap.iloc[gap_at_elbow_idx[0]]:.6f}')


if __name__ == '__main__':
    main()
