#!/usr/bin/env python3
"""
Evaluate trained CNN encoder on the held-out test set.
"""

import numpy as np
import torch
import torch.nn as nn
import os
import json
from glob import glob
from tqdm import tqdm

from vision_network import ResNetStateEncoder


def get_batch_files(folder):
    """Get sorted list of batch files in folder."""
    return sorted(glob(os.path.join(folder, 'batch_*_frames.npy')))


def load_batch(frames_file):
    """Load a batch given the frames file path."""
    states_file = frames_file.replace('_frames.npy', '_states.npy')
    frames = np.load(frames_file)
    states = np.load(states_file)
    return frames, states


def process_frames(frames_np):
    """(N, 16, 224, 224, 3) -> (N, 48, 224, 224)"""
    t = torch.from_numpy(frames_np).float() / 255.0
    t = t.permute(0, 1, 4, 2, 3)
    t = t.reshape(t.shape[0], 48, 224, 224)
    return t


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = "vision_batches"
    model_path = "trained_cnn/best_model.pth"
    norm_path = "trained_cnn/norm_stats.pt"

    # Load test files
    test_files = get_batch_files(os.path.join(data_dir, 'test'))
    print(f"Test batches: {len(test_files)}")

    # Load model
    model = ResNetStateEncoder(input_channels=48, output_dim=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    # Load normalization stats
    norm_stats = torch.load(norm_path, map_location=device)
    state_mean = norm_stats['mean'].to(device)
    state_std = norm_stats['std'].to(device)
    print(f"Loaded normalization stats")

    criterion = nn.MSELoss()

    # Evaluate
    test_loss = 0
    test_samples = 0

    with torch.no_grad():
        for frames_file in tqdm(test_files, desc="Evaluating test set"):
            frames_np, states_np = load_batch(frames_file)

            frames = process_frames(frames_np).to(device)
            states = torch.from_numpy(states_np).float().to(device)

            # Normalize states (same as training)
            states_norm = (states - state_mean) / state_std

            pred = model(frames)
            loss = criterion(pred, states_norm)

            test_loss += loss.item()
            test_samples += len(frames)

    avg_test_loss = test_loss / len(test_files)

    print(f"\n{'='*50}")
    print(f"Test Set Evaluation Results")
    print(f"{'='*50}")
    print(f"Test batches: {len(test_files)}")
    print(f"Test samples: {test_samples:,}")
    print(f"Test MSE (normalized): {avg_test_loss:.4f}")
    print(f"Test RMSE: {np.sqrt(avg_test_loss):.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
