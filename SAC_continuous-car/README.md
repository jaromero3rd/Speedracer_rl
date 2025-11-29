# SAC Continuous - Soft Actor-Critic for Continuous Action Spaces

This module implements Soft Actor-Critic (SAC) for continuous action spaces, adapted from the discrete SAC implementation in `SAC_discrete-main`.

## Key Differences from Discrete SAC

| Component | Discrete SAC | Continuous SAC |
|-----------|--------------|----------------|
| **Actor Output** | Action probabilities (softmax) | Mean & log_std (Gaussian) |
| **Action Selection** | Categorical sampling | Gaussian sampling + tanh squashing |
| **Critic Input** | State only → Q for all actions | (State, Action) → Single Q value |
| **Policy Loss** | Expectation over action probabilities | Reparameterization trick |
| **Target Entropy** | -log(1/|A|) * 0.98 | -dim(A) |

## Architecture

### Actor (Gaussian Policy)
```
State → FC(256) → ReLU → FC(256) → ReLU → [Mean, Log_Std]
                                              ↓
                                    Sample from N(mean, std)
                                              ↓
                                        tanh squashing
                                              ↓
                                   Scale to action bounds
```

### Critic (Twin Q-Networks)
```
[State, Action] → FC(256) → ReLU → FC(256) → ReLU → Q-value
```

## Usage

### Training

```bash
# Train on Pendulum (default)
python train.py --env Pendulum-v1 --episodes 200

# Train on other continuous control environments
python train.py --env HalfCheetah-v4 --episodes 1000 --hidden_size 256

# Train with video logging
python train.py --env Pendulum-v1 --log_video 1

# Train with custom hyperparameters
python train.py --env Pendulum-v1 \
    --learning_rate 3e-4 \
    --gamma 0.99 \
    --tau 5e-3 \
    --batch_size 256 \
    --hidden_size 256
```

### As a Module

```python
import gymnasium as gym
import torch
from SAC_continuous_car import SACContinuous

# Create environment
env = gym.make('Pendulum-v1')

# Create agent
agent = SACContinuous(
    state_size=env.observation_space.shape[0],
    action_size=env.action_space.shape[0],
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    action_space=env.action_space,
    learning_rate=3e-4
)

# Training loop
state, _ = env.reset()
action = agent.get_action(state, training=True)
next_state, reward, done, truncated, info = env.step(action)

# Store in buffer and learn
# buffer.add(state, action, reward, next_state, done)
# agent.learn(step, buffer.sample())
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--env` | Pendulum-v1 | Gymnasium environment name |
| `--episodes` | 200 | Number of training episodes |
| `--max_steps` | 1000 | Max steps per episode (0 = no limit) |
| `--buffer_size` | 1,000,000 | Replay buffer capacity |
| `--batch_size` | 256 | Training batch size |
| `--learning_rate` | 3e-4 | Learning rate for all networks |
| `--gamma` | 0.99 | Discount factor |
| `--tau` | 5e-3 | Soft update coefficient |
| `--hidden_size` | 256 | Hidden layer size |
| `--entropy_bonus` | None | Fixed alpha (None = learnable) |
| `--random_samples` | 10000 | Initial random samples |
| `--updates_per_step` | 1 | Gradient updates per env step |
| `--log_video` | 0 | Enable video logging (1 = on) |
| `--save_every` | 50 | Save model every N episodes |

## Shared Components

The following components are imported from `SAC_discrete-main`:
- `ReplayBuffer`: Experience replay buffer
- `logging_utils`: TensorBoard logging utilities
- `hidden_init`: Weight initialization function

## Tested Environments

- `Pendulum-v1` - Classic pendulum swing-up
- `MountainCarContinuous-v0` - Continuous mountain car
- `LunarLanderContinuous-v2` - Continuous lunar lander
- `BipedalWalker-v3` - 2D walking robot
- MuJoCo environments (requires mujoco installation):
  - `HalfCheetah-v4`
  - `Ant-v4`
  - `Humanoid-v4`

## For Racecar Gym

To use with racecar_gym (which has continuous steering/throttle):

```python
import racecar_gym
from SAC_continuous_car import SACContinuous

# Create racecar environment
env = gym.make('racecar_gym:SingleAgentRaceEnv-v0', scenario='austria')

# The action space is typically Box(-1, 1, shape=(2,)) for [steering, throttle]
agent = SACContinuous(
    state_size=env.observation_space.shape[0],  # Or your encoded state size
    action_size=env.action_space.shape[0],
    device=device,
    action_space=env.action_space
)
```

## References

- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL](https://arxiv.org/abs/1801.01290)
- [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)

