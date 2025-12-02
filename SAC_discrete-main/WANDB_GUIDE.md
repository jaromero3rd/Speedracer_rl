# Using Wandb with SAC Discrete

The training script has been updated to use Weights & Biases (wandb) instead of TensorBoard for experiment tracking and video logging.

## Installation

First, install wandb:

```bash
pip install wandb
```

## Login

Before your first run, login to wandb:

```bash
wandb login
```

This will prompt you to paste your API key from https://wandb.ai/authorize

## Usage

### Basic Training

```bash
python train.py --env CartPole-v1 --episodes 100
```

This will:
- Create a run in the default project "SAC_discrete"
- Log all metrics (rewards, losses, alpha, etc.) to wandb
- Use the default run name "SAC"

### Custom Project and Run Names

```bash
python train.py \
    --env CartPole-v1 \
    --wandb_project "my-rl-project" \
    --run_name "cartpole-experiment-1" \
    --episodes 200
```

### With Video Logging

```bash
python train.py \
    --env CartPole-v1 \
    --log_video 1 \
    --episodes 100
```

This will record videos every 10 episodes (1, 11, 21, 31, ...) and upload them to wandb.

### Team/Organization Projects

If you want to log to a team workspace:

```bash
python train.py \
    --env CartPole-v1 \
    --wandb_entity "my-team-name" \
    --wandb_project "rl-experiments"
```

## Logged Metrics

The following metrics are logged to wandb at each episode:

- `episode`: Current episode number
- `reward`: Total reward for the episode
- `avg_reward_10`: Rolling average of last 10 episodes
- `total_steps`: Cumulative steps across all episodes
- `policy_loss`: Actor/policy loss
- `alpha_loss`: Entropy temperature loss
- `bellmann_error1`: Critic 1 loss
- `bellmann_error2`: Critic 2 loss  
- `current_alpha`: Current entropy temperature value
- `episode_steps`: Steps in current episode
- `buffer_size`: Current replay buffer size

## Hyperparameters

All hyperparameters are automatically logged to wandb config:

- Environment name
- Number of episodes
- Buffer size
- Batch size
- Learning rate
- Entropy bonus (alpha)
- Epsilon (for epsilon-greedy)
- Observation buffer length
- Random seed

## Viewing Results

After starting training, wandb will print a URL to view your run:

```
wandb: 🚀 View run at https://wandb.ai/your-username/SAC_discrete/runs/xxxxx
```

Click this link to see:
- Real-time metric plots
- Hyperparameter configuration
- System metrics (GPU/CPU usage)
- Videos of agent behavior (if enabled)
- Code versioning

## Offline Mode

To run without internet connection (logs stored locally):

```bash
wandb offline
python train.py --env CartPole-v1
```

Later, you can sync runs:

```bash
wandb sync
```

## Disabling Wandb

If you want to disable wandb logging temporarily:

```bash
wandb disabled
python train.py --env CartPole-v1
```

## Comparison with TensorBoard

| Feature | TensorBoard | Wandb |
|---------|-------------|-------|
| Setup | Local files in `./logs` | Cloud-based with local caching |
| Viewing | `tensorboard --logdir ./logs` | Automatic web UI |
| Video Logging | Add to IMAGES tab | Native video player |
| Hyperparameter Tracking | HPARAMS tab | Config tab + sweeps |
| Collaboration | Share log files | Share URL |
| Experiment Comparison | Multiple `--logdir` | Built-in comparison UI |

## Troubleshooting

### "wandb not found"
```bash
pip install wandb
```

### "Not logged in"
```bash
wandb login
```

### Videos not uploading
- Check that `--log_video 1` is set
- Verify moviepy is installed: `pip install moviepy`
- Videos are logged every 10 episodes

### Too many runs in project
You can delete old runs from the wandb UI or use:
```bash
wandb artifact cache cleanup
```

