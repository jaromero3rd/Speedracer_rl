# Vision-Based RL Pipeline Setup Guide

## Overview
This guide walks through the complete pipeline for training a CNN encoder and vision-based RL agent.

## Files to Transfer to High-Storage Computer

### Required Files:
```
SAC_discrete-main/
├── trained_models/                    # Trained agents
│   ├── grid_search_005SAC_discrete0.pth
│   ├── grid_search_047SAC_discrete0.pth
│   └── grid_search_073SAC_discrete0.pth
├── generate_vision_dataset.py         # Dataset generation script
├── train_cnn_encoder.py               # CNN training script
├── agent.py                           # SAC agent (needed for inference)
├── networks.py                        # Network architectures
└── buffer.py                          # (may be needed)
```

### Environment:
- Use the same `environment.yml` from this repository
- Or install: `pytorch`, `gymnasium`, `opencv-python`, `tqdm`, `matplotlib`

---

## Pipeline Steps

### Step 1: Generate Large Dataset (~42 GB for 200 episodes)

```bash
cd SAC_discrete-main

# Activate conda environment
conda activate speedracer_rl

# Generate dataset from all 3 trained agents
python generate_vision_dataset.py \
  --episodes 200 \
  --obs_buffer_len 16 \
  --frame_stack 4 \
  --output_dir vision_dataset \
  --resize 224 224

# Expected output: ~300k frames
# Storage: ~42 GB compressed
# Time: ~2-3 hours (depending on system)
```

**Output Structure:**
```
vision_dataset/
├── grid_search_005SAC_discrete0/
│   ├── dataset.npz              # Compressed data
│   ├── metadata.json            # Dataset info
│   └── state_stats.json         # Normalization stats
├── grid_search_047SAC_discrete0/
│   └── ...
└── grid_search_073SAC_discrete0/
    └── ...
```

---

### Step 2: Train CNN Encoder

```bash
# Train ResNet18 from scratch on combined dataset
python train_cnn_encoder.py \
  --data_dir vision_dataset \
  --output_dir trained_cnn \
  --epochs 50 \
  --batch_size 64 \
  --lr 1e-4 \
  --normalize \
  --seed 42

# With GPU: ~2-4 hours
# Without GPU: ~10-20 hours
```

**Outputs:**
```
trained_cnn/
├── best_model.pth              # Best model (lowest val loss)
├── final_model.pth             # Final epoch model
├── checkpoint_epoch10.pth      # Periodic checkpoints
├── checkpoint_epoch20.pth
└── training_history.png        # Loss curves
```

**Monitor Training:**
- Watch validation loss (should decrease)
- Check MAE per dimension (x, x_dot, theta, theta_dot)
- Typical good values: MAE < 0.1 for each dimension (if normalized)

---

### Step 3: Create Vision-Based RL Agent (Next Step)

After CNN is trained, we'll create a modified SAC agent that:
1. Takes images as input
2. Uses CNN encoder to get embeddings (512D)
3. Feeds embeddings to Actor/Critic networks
4. Trains end-to-end or with frozen CNN

---

## Storage Requirements Summary

| Episodes per Agent | Total Frames | Compressed Size | Free Space Needed |
|-------------------|--------------|-----------------|-------------------|
| 50                | 75,000       | ~10 GB         | ~15 GB           |
| 100               | 150,000      | ~21 GB         | ~32 GB           |
| 200               | 300,000      | ~42 GB         | ~63 GB           |
| 500               | 750,000      | ~105 GB        | ~160 GB          |

---

## Alternative: Smaller Images (If Storage Limited)

If you want to reduce storage by 75%:

```bash
# Use 112x112 images instead of 224x224
python generate_vision_dataset.py \
  --episodes 200 \
  --resize 112 112 \
  --frame_stack 4

# Storage: ~10 GB instead of ~42 GB
```

**Note:** 112x112 may be sufficient for CartPole (simple visuals).

---

## Troubleshooting

### Out of Memory During Training:
```bash
# Reduce batch size
python train_cnn_encoder.py --batch_size 32  # or 16
```

### Slow Dataset Generation:
```bash
# Reduce episodes per agent
python generate_vision_dataset.py --episodes 100  # instead of 200
```

### CNN Not Learning Well:
- Check if states are normalized (`--normalize` flag)
- Increase training epochs (`--epochs 100`)
- Try lower learning rate (`--lr 5e-5`)
- Generate more data

---

## Next Steps After CNN Training

1. Transfer trained CNN model back to this computer (if needed)
2. Create vision-based RL training script
3. Train vision-based SAC agent using CNN embeddings
4. Compare performance: state-based vs vision-based

---

## Quick Reference Commands

```bash
# Full pipeline on high-storage computer:
conda activate speedracer_rl
cd SAC_discrete-main

# 1. Generate data
python generate_vision_dataset.py --episodes 200 --obs_buffer_len 16

# 2. Train CNN
python train_cnn_encoder.py --data_dir vision_dataset --normalize --epochs 50

# 3. Check results
ls -lh vision_dataset/*/dataset.npz
ls -lh trained_cnn/best_model.pth
```
