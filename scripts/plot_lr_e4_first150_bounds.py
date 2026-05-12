import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_log_file(log_path):
    """Parse training log to extract epoch, train_loss, val_loss."""
    data = []
    pattern = r'lr=[\d.e-]+ epoch=(\d+)/\d+ train_loss=([\d.]+) val_loss=([\d.]+)'
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                val_loss = float(match.group(3))
                data.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
    return pd.DataFrame(data)


def compute_bounds(series, window=21, k=1.0):
    mean = series.rolling(window=window, center=True, min_periods=1).mean()
    std = series.rolling(window=window, center=True, min_periods=1).std().fillna(0.0)
    lower = mean - k * std
    upper = mean + k * std
    return mean, lower, upper


def main():
    log_path = os.path.join('exp', 'tenebrio_batch1_1200ep', 'tenebrio_batch1_200ep', 'train_1e-4.log')
    csv_path = os.path.join('exp', 'tenebrio_batch1_1200ep', 'checkpoints', 'lr_1e-04.csv')
    
    # Load from log file (epochs 1-200)
    if not os.path.exists(log_path):
        print(f'Log file not found: {log_path}')
        return
    
    df_log = parse_log_file(log_path)
    if df_log.empty:
        print('No data parsed from log file')
        return
    
    # Load from CSV (epochs 201+)
    if not os.path.exists(csv_path):
        print(f'CSV file not found: {csv_path}')
        return
    
    df_csv = pd.read_csv(csv_path)
    
    # Combine log and CSV data, filtering for epochs 50-1200
    df = pd.concat([df_log, df_csv], ignore_index=True)
    df = df[(df['epoch'] >= 50) & (df['epoch'] <= 1200)].reset_index(drop=True)
    
    epochs = df['epoch']
    train = df['train_loss']
    val = df['val_loss']
    
    # Compute smoothed mean and bounds
    train_m, _, _ = compute_bounds(train, window=21, k=1.0)
    val_m, _, val_high = compute_bounds(val, window=21, k=1.0)
    avg_curve = (train_m + val_high) / 2.0
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(epochs, train_m, label='train loss (smoothed)', color='C0', linewidth=2)
    ax.plot(epochs, val_high, label='val loss upper bound (smoothed)', color='C1', linewidth=2)
    ax.plot(epochs, avg_curve, label='average curve', color='C2', linewidth=2.5)
    
    ax.fill_between(epochs, train_m, val_high, where=(train_m <= val_high), 
                    color='green', alpha=0.10, label='train < val_upper')
    ax.fill_between(epochs, train_m, val_high, where=(train_m > val_high), 
                    color='red', alpha=0.10, label='train > val_upper')
    ax.fill_between(epochs, train_m, avg_curve, color='C2', alpha=0.08, label='train to average')
    ax.fill_between(epochs, avg_curve, val_high, color='C2', alpha=0.05, label='average to val_upper')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('LR 1e-4: Epochs 50-1200 — Smoothed Train, Val Upper Bound, and Average', fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    out = os.path.join('exp', 'tenebrio_batch1_1200ep', 'checkpoints', 
                       'lr_1e-4_epochs50-1200_train_val_upper_average.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f'Saved plot to {out}')
    print(f'Data: {len(df)} epochs (range {epochs.min()}-{epochs.max()})')
    print(f'Train loss range: {train.min():.6f} to {train.max():.6f}')
    print(f'Val loss range: {val.min():.6f} to {val.max():.6f}')


if __name__ == '__main__':
    main()
