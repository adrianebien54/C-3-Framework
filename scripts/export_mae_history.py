"""
Export MAE (and optionally train_loss, val_loss, mse) history from TFEvents
to a small JSON file alongside each exp dir. Used to transfer training history
without shipping large TFEvents binaries.

Usage:
    python scripts/export_mae_history.py [exp_dir1] [exp_dir2] ...
    # or with no args: auto-finds all exp dirs in ./exp/ and ../MobileCount/exp/
"""
import os
import sys
import json
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def export_dir(exp_dir):
    acc = EventAccumulator(exp_dir, size_guidance={'scalars': 0})
    acc.Reload()

    available = acc.Tags().get('scalars', [])
    data = {}
    for tag in ['mae', 'mse', 'val_loss', 'train_loss']:
        if tag in available:
            events = acc.Scalars(tag)
            data[tag] = {'steps': [e.step for e in events],
                         'values': [e.value for e in events]}

    if not data:
        print(f'  [SKIP] no scalar data in {exp_dir}')
        return

    out_path = os.path.join(exp_dir, 'mae_history.json')
    with open(out_path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))

    mae_count = len(data.get('mae', {}).get('steps', []))
    size_kb = os.path.getsize(out_path) / 1024
    print(f'  Saved {out_path}  ({mae_count} MAE steps, {size_kb:.1f} KB)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', nargs='*', help='exp dirs to export')
    args = parser.parse_args()

    if args.dirs:
        dirs = args.dirs
    else:
        # Auto-discover
        dirs = []
        for base in ['exp', '../MobileCount/exp']:
            if os.path.isdir(base):
                for d in sorted(os.listdir(base)):
                    full = os.path.join(base, d)
                    if os.path.isdir(full) and any('tfevents' in f for f in os.listdir(full)):
                        dirs.append(full)

    print(f'Exporting {len(dirs)} exp directories...')
    for d in dirs:
        print(f'\n{d}')
        export_dir(d)
    print('\nDone.')


if __name__ == '__main__':
    main()
