#!/usr/bin/env python3
"""
Split batches into train/val/test (80/10/10).
"""

import os
import shutil
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, 'metadata.json'), 'r') as f:
        meta = json.load(f)

    n = meta['num_batches']
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)

    splits = {
        'train': list(range(0, train_end)),
        'val': list(range(train_end, val_end)),
        'test': list(range(val_end, n))
    }

    print(f"Total: {n} batches")
    for name, indices in splits.items():
        print(f"  {name}: {len(indices)}")

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(args.data_dir, split), exist_ok=True)

    for split, indices in splits.items():
        for idx in indices:
            for suffix in ['_frames.npy', '_states.npy']:
                src = os.path.join(args.data_dir, f'batch_{idx:05d}{suffix}')
                dst = os.path.join(args.data_dir, split, f'batch_{idx:05d}{suffix}')
                if os.path.exists(src):
                    shutil.move(src, dst)

    meta['splits'] = {k: len(v) for k, v in splits.items()}
    with open(os.path.join(args.data_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
