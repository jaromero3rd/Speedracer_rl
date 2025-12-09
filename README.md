# Speedracer RL: Vision-Based CartPole Control

A reinforcement learning project that trains SAC agents on CartPole using both ground-truth state observations and vision-based control via a CNN encoder.

## Project Overview

This project demonstrates:
1. **SAC Agent Training** - Soft Actor-Critic with discrete actions for CartPole balancing
2. **CNN State Prediction** - ResNet18 encoder that predicts state from RGB frames
3. **Vision-Based Control** - Complete pipeline from pixels to actions
4. **Noise Robustness** - Comparison of standard vs noise-trained agents

## Installation

### Option 1: Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate speedracer_rl
```

### Option 2: pip
```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## Quick Start

### 1. Run Pre-trained Agents
```bash
# List available agents
python run_agent.py --list-agents

# Run with ground-truth state (perfect performance)
python run_agent.py --mode state --episodes 10

# Run with vision (CNN-predicted state from pixels)
python run_agent.py --mode vision --episodes 10

# Use noise-trained agent (more robust to CNN errors)
python run_agent.py --mode vision --agent noise2 --episodes 10

# Show live rendering window
python run_agent.py --mode state --render

# Save video to file
python run_agent.py --mode state --episodes 1 --save_video demo.mp4
```

### 2. Compare Agents (Standard vs Noise-Trained)
```bash
# Compares agent 073 vs noise2 agent using CNN predictions
python compare_agents.py --episodes 20
```

### 3. Train a New SAC Agent
```bash
# Train SAC on CartPole with 16-frame observation buffer
python train.py --env CartPole-v1 --episodes 500 --obs_buffer_max_len 16
```

## Pre-trained Models

All models are included in the repository:

| Model | Location | Description |
|-------|----------|-------------|
| SAC Agent 073 | `trained_models/grid_search_073SAC_discrete0.pth` | Best performing agent (500 steps) |
| SAC Agent noise2 | `trained_models/noise2SAC_discrete0.pth` | Noise-trained agent (robust to CNN errors) |
| CNN Encoder | `trained_cnn/best_model.pth` | ResNet18 state predictor |
| Normalization | `trained_cnn/norm_stats.pt` | State normalization statistics |

## Demo Videos

Pre-recorded videos are in `videos/`:
- `state_agent_*.mp4` - Agents using ground-truth state (500 steps)
- `vision_agent_*.mp4` - Agents using CNN-predicted state

## Full Pipeline

### Step 1: Train SAC Agent on Ground-Truth State
```bash
python train.py --env CartPole-v1 --episodes 500 --obs_buffer_max_len 16 --run_name my_agent
```

### Step 2: Generate Vision Dataset
```bash
# Collect frame-state pairs from trained agent
python generate_vision_dataset.py \
    --model_path trained_models/my_agentSAC_discrete0.pth \
    --episodes 200 \
    --obs_buffer_len 16 \
    --output_dir vision_dataset
```

### Step 3: Convert to Batches
```bash
python convert_to_batches.py --input_dir vision_dataset --output_dir vision_batches
python split_batches.py --input_dir vision_batches
```

### Step 4: Train CNN Encoder
```bash
python train_cnn_batches.py --data_dir vision_batches --epochs 50
```

### Step 5: Run Vision-Based Agent
```bash
python run_vision_agent.py \
    --agent_path trained_models/my_agentSAC_discrete0.pth \
    --cnn_path trained_cnn/best_model.pth \
    --episodes 10
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `run_agent.py` | **Main entry point** - run agents in state or vision mode |
| `train.py` | Train SAC agent on CartPole |
| `compare_agents.py` | Compare standard vs noise-trained agents |
| `generate_vision_dataset.py` | Collect frame-state pairs |
| `train_cnn_batches.py` | Train CNN encoder |
| `eval_test_set.py` | Evaluate CNN on held-out test data |

## Architecture

### SAC Agent
- **State**: 64-dim (16 frames × 4 state variables)
- **Actions**: 2 (left/right)
- **Networks**: Actor + Dual Critics (32 hidden units)

### CNN Encoder
- **Architecture**: ResNet18 (modified)
- **Input**: 48 channels (16 frames × 3 RGB)
- **Output**: 64-dim state prediction
- **Parameters**: 11.35M

## Results

### CNN Training
| Split | MSE | RMSE |
|-------|-----|------|
| Train | 0.0226 | 0.150 |
| Validation | 0.0239 | 0.155 |
| Test | 0.0232 | 0.152 |

### Vision-Based Agent Performance
| Agent | Mean Steps | Best | Worst |
|-------|------------|------|-------|
| Standard (073) | 59.3 | 79 | 15 |
| Noise-trained | 119.9 | 348 | 71 |

The noise-trained agent handles CNN prediction errors significantly better.

## Project Structure

```
Speedracer_rl/
├── train.py                 # SAC training
├── agent.py                 # SAC agent implementation
├── networks.py              # Actor/Critic networks
├── buffer.py                # Replay buffer
├── vision_network.py        # CNN encoder (ResNet18)
├── run_agent.py             # Main entry point (state/vision modes)
├── compare_agents.py        # Agent comparison
├── generate_vision_dataset.py
├── train_cnn_batches.py
├── trained_models/          # SAC agent weights
├── trained_cnn/             # CNN encoder weights
└── videos/                  # Demo recordings
```

## Team

**Speed Racers** - CIS 5200 Fall 2025
- Jaime Romero
- Casey Mogilevsky
- Henry Corkran
