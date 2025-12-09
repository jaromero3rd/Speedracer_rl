#!/usr/bin/env python3
"""
Create visualization of frame stack for the report.
Shows 6 frames from a 16-frame stack to illustrate temporal input.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # Load a sample batch
    batch_path = "vision_batches/train/batch_00500_frames.npy"
    frames = np.load(batch_path)

    print(f"Loaded batch shape: {frames.shape}")  # (256, 16, 224, 224, 3)

    # Take first sample
    sample = frames[0]  # (16, 224, 224, 3)

    # Select 6 frames evenly spaced
    indices = [0, 3, 6, 9, 12, 15]

    # Create figure
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))

    for i, idx in enumerate(indices):
        axes[i].imshow(sample[idx])
        axes[i].set_title(f"Frame {idx+1}")
        axes[i].axis('off')

    plt.suptitle("Sample 16-Frame Stack (showing frames 1, 4, 7, 10, 13, 16)", fontsize=12)
    plt.tight_layout()

    # Save
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/frame_stack_example.png", dpi=150, bbox_inches='tight')
    plt.savefig("figures/frame_stack_example.pdf", bbox_inches='tight')
    print("Saved to figures/frame_stack_example.png and .pdf")

    plt.close()

if __name__ == "__main__":
    main()
