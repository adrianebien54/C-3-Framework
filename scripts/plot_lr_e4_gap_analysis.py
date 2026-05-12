import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def compute_smooth(series, window=21):
    return series.rolling(window=window, center=True, min_periods=1).mean()


def compute_gap_derivative(epochs, train_series, val_series, smooth_window=21, derivative_window=9):
    train_s = compute_smooth(train_series, window=smooth_window)
    val_s = compute_smooth(val_series, window=smooth_window)

    gap = val_s - train_s
    gap_s = compute_smooth(gap, window=derivative_window)

    epoch_step = epochs.diff().replace(0, np.nan)
    gap_derivative = gap_s.diff() / epoch_step
    gap_derivative = gap_derivative.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gap_derivative_s = compute_smooth(gap_derivative, window=derivative_window)
    gap_acceleration = gap_derivative_s.diff().fillna(0.0)

    return train_s, val_s, gap_s, gap_derivative_s, gap_acceleration


def detect_divergence_point(epochs, gap_derivative, gap_acceleration, positive_window=7):
    positive = (gap_derivative > 0) & (gap_acceleration > 0)
    sustained = positive.rolling(window=positive_window, min_periods=positive_window).sum() == positive_window
    idx = sustained[sustained].index
    if len(idx) == 0:
        return None
    return int(epochs.iloc[int(idx[0])])


def main():
    log_path = os.path.join('exp', 'tenebrio_batch1_1200ep', 'tenebrio_batch1_200ep', 'train_1e-4.log')
    csv_path = os.path.join('exp', 'tenebrio_batch1_1200ep', 'checkpoints', 'lr_1e-04.csv')

    if not os.path.exists(log_path):
        print(f'Log file not found: {log_path}')
        return
    if not os.path.exists(csv_path):
        print(f'CSV file not found: {csv_path}')
        return

    df_log = parse_log_file(log_path)
    df_csv = pd.read_csv(csv_path)
    if df_log.empty or df_csv.empty:
        print('Could not parse training data.')
        return

    df = pd.concat([df_log, df_csv], ignore_index=True)
    df = df[(df['epoch'] >= 50) & (df['epoch'] <= 1200)].reset_index(drop=True)

    epochs = df['epoch']
    train = df['train_loss']
    val = df['val_loss']

    train_s, val_s, gap_s, gap_derivative_s, gap_acceleration = compute_gap_derivative(
        epochs, train, val, smooth_window=21, derivative_window=9
    )
    divergence_epoch = detect_divergence_point(
        epochs, gap_derivative_s, gap_acceleration, positive_window=7
    )

    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axs[0].plot(epochs, gap_s, color='C2', linewidth=2, label='gap = val - train (smoothed)')
    axs[0].axhline(0.0, color='black', linewidth=1, alpha=0.6)
    if divergence_epoch is not None:
        axs[0].axvline(divergence_epoch, color='red', linestyle='--', linewidth=2,
                       label=f'divergence epoch ~ {divergence_epoch}')
    axs[0].set_ylabel('Gap', fontsize=12)
    axs[0].set_title('LR 1e-4: Epochs 50-1200 — Gap Analysis', fontsize=13)
    axs[0].legend(fontsize=10, loc='upper right')
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(epochs, gap_derivative_s, color='C3', linewidth=2,
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
                       'lr_1e-4_epochs50-1200_gap_derivative.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f'Saved plot to {out}')
    print(f'Data: {len(df)} epochs (range {epochs.min()}-{epochs.max()})')
    if divergence_epoch is not None:
        print(f'Detected divergence epoch: {divergence_epoch}')
    else:
        print('No sustained positive accelerating divergence point detected')


if __name__ == '__main__':
    main()
