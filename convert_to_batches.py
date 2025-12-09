#!/usr/bin/env python3
"""
Convert chunks to individual batch files for training.
"""

import numpy as np
import os
from glob import glob
from tqdm import tqdm
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # find all chunks
    all_chunks = []
    for subdir in sorted(os.listdir(args.data_dir)):
        subdir_path = os.path.join(args.data_dir, subdir)
        if os.path.isdir(subdir_path):
            chunks = sorted(glob(os.path.join(subdir_path, 'chunk_*.npz')))
            all_chunks.extend(chunks)
            print(f"  {subdir}: {len(chunks)} chunks")

    print(f"Total: {len(all_chunks)} chunks")

    # count samples
    total_samples = 0
    frame_shape = None
    state_dim = None

    for chunk_file in tqdm(all_chunks, desc="Scanning"):
        data = np.load(chunk_file)
        total_samples += len(data['states'])
        if frame_shape is None:
            frame_shape = data['frame_stacks'].shape[1:]
            state_dim = data['states'].shape[1]

    num_batches = (total_samples + args.batch_size - 1) // args.batch_size
    print(f"Samples: {total_samples:,}, Batches: {num_batches}")

    # write batches
    batch_idx = 0
    current_frames = None
    current_states = None
    current_size = 0

    pbar = tqdm(total=num_batches, desc="Writing")

    for chunk_file in all_chunks:
        data = np.load(chunk_file)
        chunk_frames = data['frame_stacks']
        chunk_states = data['states'].astype(np.float32)
        chunk_pos = 0

        while chunk_pos < len(chunk_states):
            needed = args.batch_size - current_size
            available = len(chunk_states) - chunk_pos
            take = min(needed, available)

            new_frames = chunk_frames[chunk_pos:chunk_pos + take]
            new_states = chunk_states[chunk_pos:chunk_pos + take]
            chunk_pos += take

            if current_frames is None:
                current_frames = new_frames
                current_states = new_states
            else:
                current_frames = np.concatenate([current_frames, new_frames])
                current_states = np.concatenate([current_states, new_states])
            current_size += take

            if current_size >= args.batch_size:
                np.save(os.path.join(args.output_dir, f'batch_{batch_idx:05d}_frames.npy'), current_frames)
                np.save(os.path.join(args.output_dir, f'batch_{batch_idx:05d}_states.npy'), current_states)
                batch_idx += 1
                pbar.update(1)
                current_frames = None
                current_states = None
                current_size = 0

    # save leftover
    if current_size > 0:
        np.save(os.path.join(args.output_dir, f'batch_{batch_idx:05d}_frames.npy'), current_frames)
        np.save(os.path.join(args.output_dir, f'batch_{batch_idx:05d}_states.npy'), current_states)
        batch_idx += 1
        pbar.update(1)

    pbar.close()

    # metadata
    metadata = {
        'total_samples': total_samples,
        'num_batches': batch_idx,
        'batch_size': args.batch_size,
        'frame_shape': list(frame_shape),
        'state_dim': state_dim
    }
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    sample_size = os.path.getsize(os.path.join(args.output_dir, 'batch_00000_frames.npy'))
    print(f"\nDone! {batch_idx} batches, ~{sample_size * batch_idx / 1e9:.1f} GB total")


if __name__ == "__main__":
    main()
