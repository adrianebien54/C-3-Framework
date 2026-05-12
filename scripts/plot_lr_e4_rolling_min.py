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
    """Apply rolling minimum filter to smooth the series."""
    return series.rolling(window=window, center=True, min_periods=1).min()


def apply_savgol(series, window=51, polyorder=3):
    """Apply Savitzky-Golay filter to further refine the trend."""
    return savgol_filter(series, window_length=window, polyorder=polyorder)


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
    df = df[(df['epoch'] >= 0) & (df['epoch'] <= 1200)].reset_index(drop=True)

    epochs = df['epoch'].values
    train = df['train_loss']
    val = df['val_loss']

    # Apply rolling minimum filter followed by Savitzky-Golay filtering
    train_min = apply_rolling_min(train, window=30)
    val_min = apply_rolling_min(val, window=30)
    train_filtered = apply_savgol(train_min, window=51, polyorder=3)
    val_filtered = apply_savgol(val_min, window=51, polyorder=3)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(epochs, train_filtered, label='train loss (rolling min + Savitzky-Golay)', color='C0', linewidth=2)
    ax.plot(epochs, val_filtered, label='val loss (rolling min + Savitzky-Golay)', color='C1', linewidth=2, linestyle='--')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('LR 1e-4: Epochs 0-1200 — Train vs Val Loss (Rolling Min + Savitzky-Golay)', fontsize=13)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)

    out = os.path.join('exp', 'tenebrio_batch1_1200ep', 'checkpoints',
                       'lr_1e-4_epochs0-1200_rolling_min_w30_savgol.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f'Saved plot to {out}')
    print(f'Data: {len(df)} epochs (range {epochs.min()}-{epochs.max()})')
    print(f'Train loss range: {train.min():.6f} to {train.max():.6f}')
    print(f'Val loss range: {val.min():.6f} to {val.max():.6f}')
    print(f'Rolling minimum filter: window=30')
    print(f'Savitzky-Golay filter: window=51, polyorder=3')

if __name__ == '__main__':
    main()
