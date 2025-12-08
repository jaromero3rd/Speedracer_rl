# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAC (Soft Actor-Critic) implementation for discrete action spaces, focused on CartPole. Includes a vision-based learning pipeline for training CNN encoders from pixel observations.

## Common Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Training
```bash
cd SAC_discrete-main

# Train SAC agent on CartPole
python train.py --env CartPole-v1 --episodes 500 --run_name my_run

# Key training arguments:
#   --env: Gymnasium environment (default: CartPole-v1)
#   --episodes: Number of training episodes (default: 100)
#   --batch_size: Training batch size (default: 256)
#   --learning_rate: Learning rate (default: 5e-4)
#   --entropy_bonus: Fixed alpha or 'None' for learnable (default: None)
#   --obs_buffer_max_len: Observation history length (default: 4)
#   --epsilon: Epsilon-greedy exploration (default: 0.0)
#   --log_video: Enable video logging to TensorBoard (0 or 1)
#   --save_every: Save model every N episodes (default: 100)
```

### Hyperparameter Grid Search
```bash
cd SAC_discrete-main
python grid_search.py
```

### Vision Pipeline (CNN Encoder Training)
```bash
cd SAC_discrete-main

# Generate dataset from trained agents
python generate_vision_dataset.py --episodes 200 --obs_buffer_len 16 --frame_stack 4 --output_dir vision_dataset

# Train CNN encoder
python train_cnn_encoder.py --data_dir vision_dataset --output_dir trained_cnn --epochs 50 --normalize
```

### TensorBoard
```bash
tensorboard --logdir SAC_discrete-main/logs
```

## Architecture

### Neural Networks (networks.py)
- **Actor**: MLP with softmax output for discrete action probabilities. Uses Categorical distribution for sampling.
- **Critic**: Dual Q-networks mapping states to Q-values for each action. No action input needed (outputs Q for all actions).

Network structure: `state → Linear(hidden) → ReLU → Linear(hidden) → ReLU → Linear(output)`

Default hidden size: 32

### SAC Agent (agent.py)
- Discrete SAC with entropy regularization
- Learnable or fixed temperature parameter (alpha)
- Dual critics (Q1, Q2) with target networks
- Soft target updates (tau)

### Observation Buffering
The agent can stack multiple consecutive observations (`--obs_buffer_max_len`) to capture temporal information. Default stacks 4 frames.

## Key Files

| File | Purpose |
|------|---------|
| `train.py` | Main training loop |
| `agent.py` | SAC agent implementation |
| `networks.py` | Actor/Critic network definitions |
| `buffer.py` | Replay buffer |
| `grid_search.py` | Hyperparameter search |
| `generate_vision_dataset.py` | Generate image dataset from trained agents |
| `train_cnn_encoder.py` | Train CNN to predict states from images |
| `logging_utils.py` | TensorBoard logging utilities |

## Directory Structure

```
SAC_discrete-main/
├── train.py                    # Main training script
├── agent.py                    # SAC agent
├── networks.py                 # Actor/Critic networks
├── buffer.py                   # Replay buffer
├── grid_search.py              # Hyperparameter search
├── generate_vision_dataset.py  # Vision dataset generation
├── train_cnn_encoder.py        # CNN encoder training
├── trained_models/             # Saved model checkpoints
└── logs/                       # TensorBoard logs
```

## Vision-Based RL Pipeline

1. **Train state-based agent** → `train.py`
2. **Generate vision dataset** → `generate_vision_dataset.py` (collects images + corresponding states)
3. **Train CNN encoder** → `train_cnn_encoder.py` (learns to predict states from images)
4. **Use CNN embeddings** for vision-based RL agent

See `SETUP_PIPELINE.md` for detailed storage requirements and step-by-step instructions.
