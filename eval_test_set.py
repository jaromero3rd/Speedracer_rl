#!/usr/bin/env python3
"""
Evaluate CNN on held-out test set.
"""

import numpy as np
import torch
import torch.nn as nn
import os
from glob import glob
from tqdm import tqdm

from vision_network import ResNetStateEncoder


def get_batch_files(folder):
    return sorted(glob(os.path.join(folder, 'batch_*_frames.npy')))


def load_batch(frames_file):
    states_file = frames_file.replace('_frames.npy', '_states.npy')
    return np.load(frames_file), np.load(states_file)


def process_frames(frames_np):
    t = torch.from_numpy(frames_np).float() / 255.0
    t = t.permute(0, 1, 4, 2, 3)
    return t.reshape(t.shape[0], 48, 224, 224)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = "vision_batches"
    model_path = "trained_cnn/best_model.pth"
    norm_path = "trained_cnn/norm_stats.pt"

    test_files = get_batch_files(os.path.join(data_dir, 'test'))
    print(f"Test batches: {len(test_files)}")

    model = ResNetStateEncoder(input_channels=48, output_dim=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    norm = torch.load(norm_path, map_location=device)
    mean = norm['mean'].to(device)
    std = norm['std'].to(device)

    criterion = nn.MSELoss()
    test_loss = 0
    test_samples = 0

    with torch.no_grad():
        for f in tqdm(test_files, desc="Testing"):
            frames_np, states_np = load_batch(f)
            frames = process_frames(frames_np).to(device)
            states = torch.from_numpy(states_np).float().to(device)
            states_norm = (states - mean) / std

            pred = model(frames)
            test_loss += criterion(pred, states_norm).item()
            test_samples += len(frames)

    avg_loss = test_loss / len(test_files)
    print(f"\nTest MSE: {avg_loss:.4f}, RMSE: {np.sqrt(avg_loss):.4f}")
    print(f"Samples: {test_samples:,}")


if __name__ == "__main__":
    main()
