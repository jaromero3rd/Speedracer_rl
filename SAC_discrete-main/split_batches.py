#!/usr/bin/env python3
"""
Split batch files into train/val/test folders (80/10/10).
Just moves files, no copying.
"""

import os
import shutil
import json
import argparse
from glob import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    # Load metadata
    with open(os.path.join(args.data_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    num_batches = metadata['num_batches']
    print(f"Total batches: {num_batches}")

    # Calculate split sizes
    train_end = int(num_batches * 0.8)
    val_end = train_end + int(num_batches * 0.1)

    train_batches = list(range(0, train_end))
    val_batches = list(range(train_end, val_end))
    test_batches = list(range(val_end, num_batches))

    print(f"Train: {len(train_batches)} batches (0-{train_end-1})")
    print(f"Val: {len(val_batches)} batches ({train_end}-{val_end-1})")
    print(f"Test: {len(test_batches)} batches ({val_end}-{num_batches-1})")

    # Create subdirectories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(args.data_dir, split), exist_ok=True)

    # Move files
    def move_batches(batch_indices, split_name):
        for batch_idx in batch_indices:
            for suffix in ['_frames.npy', '_states.npy']:
                src = os.path.join(args.data_dir, f'batch_{batch_idx:05d}{suffix}')
                dst = os.path.join(args.data_dir, split_name, f'batch_{batch_idx:05d}{suffix}')
                if os.path.exists(src):
                    shutil.move(src, dst)
        print(f"  Moved {len(batch_indices)} batches to {split_name}/")

    move_batches(train_batches, 'train')
    move_batches(val_batches, 'val')
    move_batches(test_batches, 'test')

    # Update metadata with split info
    metadata['splits'] = {
        'train': len(train_batches),
        'val': len(val_batches),
        'test': len(test_batches)
    }
    with open(os.path.join(args.data_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
