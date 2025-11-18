# Speedracer RL

A reinforcement learning project for training agents to race in the `racecar_gym` environment using PyBullet physics simulation.

## Overview

This project contains scripts for:
- Running random policies in the racecar environment (`random_actor.py`)
- Interactive testing and demonstration of the racecar gym (`run_racecar_gym.py`)
- Training DQN agents for racing tasks
- Collecting and managing training data

## Installation

### Prerequisites

- Python 3.8 or higher
- pip
- Git

### Installing racecar_gym

The `racecar_gym` package must be installed from source. Follow these steps:

#### initial racecar_gym

git clone https://github.com/axelbr/racecar_gym.git
cd racecar_gym
pip install -e .

#### Verify Installation

After installation, verify that `racecar_gym` is properly installed:

```bash
python -c "import racecar_gym; print('racecar_gym installed successfully')"
```

Or run the check in `run_racecar_gym.py`:

```bash
python run_racecar_gym.py
# Select option 6 to diagnose environments
```

### Installing Other Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```


## Scripts

### random_actor.py

**Purpose**: Runs random actions in the racecar_gym environment for testing and data collection.

**What it does**:
- Creates a racecar_gym environment
- Executes random actions for a specified number of episodes
- Collects statistics (rewards, episode lengths)
- Supports different rendering modes for visualization

**Usage**:

```bash
# Basic usage (default: Austria track, 5 episodes, human rendering)
python random_actor.py

# Custom environment and episodes
python random_actor.py --env SingleAgentAustria-v0 --episodes 10

# Different rendering modes
python random_actor.py --render-mode human              # Visual window
python random_actor.py --render-mode rgb_array_birds_eye # Bird's eye view
python random_actor.py --render-mode rgb_array_follow    # Third-person view
python random_actor.py --render-mode none                # No rendering (faster)

# Custom max steps per episode
python random_actor.py --max-steps 2000
```

**Command-line Arguments**:
- `--env`: Environment name (default: `SingleAgentAustria-v0`)
- `--episodes`: Number of episodes to run (default: 5)
- `--max-steps`: Maximum steps per episode (default: 1000)
- `--render-mode`: Rendering mode - `human`, `rgb_array_birds_eye`, `rgb_array_follow`, or `none` (default: `human`)

**Output**:
- Prints episode statistics (steps, rewards)
- Displays final statistics (average reward, episode length, min/max values)
- Renders visualization if `render_mode='human'`

### run_racecar_gym.py

**Purpose**: Interactive script for testing, demonstrating, and diagnosing the racecar_gym environment.

**What it does**:
Provides multiple modes for interacting with the racecar environment:
1. Random policy demo
2. Keyboard control
3. Performance benchmarking
4. Observation space inspection
5. Episode recording
6. Environment diagnosis
7. Exit

**Usage**:

```bash
python run_racecar_gym.py
```

Then select a mode from the interactive menu.

#### Mode 1: Random Policy Demo

Runs 3 episodes with a random policy to demonstrate the environment.

- Shows the car driving with random actions
- Displays episode statistics
- Useful for quick testing

#### Mode 2: Keyboard Control Demo

Allows manual control of the racecar using keyboard input.

**Controls**:
- `W` / `↑`: Accelerate forward
- `S` / `↓`: Brake/Reverse
- `A` / `←`: Steer left
- `D` / `→`: Steer right
- `Q` / `ESC`: Quit

**Features**:
- Real-time manual control
- Visual feedback
- Useful for understanding the environment dynamics

#### Mode 3: Performance Benchmark

Runs a performance benchmark to measure environment step times.

- Tests environment performance without rendering
- Measures average step time
- Useful for optimizing training speed

#### Mode 4: Inspect Observation Space

Examines the observation space structure and prints detailed information.

- Shows observation space structure
- Displays sample observations
- Useful for understanding what data the agent receives

#### Mode 5: Record Episode to Video

Records an episode and saves it as a video file.

- Captures episode as video
- Saves to file for later analysis
- Useful for documenting agent behavior

#### Mode 6: Diagnose Environments

Lists all available environments and their configurations.

- Finds all registered racecar environments
- Shows environment IDs
- Useful for discovering available tracks/scenarios

## Available Environments

The racecar_gym supports multiple tracks and scenarios. Environment IDs follow the pattern:

```
SingleAgent{TrackName}-v0
MultiAgent{TrackName}-v0
```

**Available Tracks**:
- `SingleAgentAustria-v0`
- `SingleAgentBerlin-v0`
- `SingleAgentMontreal-v0`
- `SingleAgentTorino-v0`
- `SingleAgentCircleCW-v0` (Circle Clockwise)
- `SingleAgentCircleCCW-v0` (Circle Counter-Clockwise)
- `SingleAgentPlechaty-v0`

## Environment Details

### Observation Space

The observation space is a dictionary containing sensor data:

- `velocity`: Box(6,) - Translational and rotational velocity components
- `acceleration`: Box(6,) - Translational and rotational acceleration components
- `rgb_camera`: Box(240, 320, 3) - RGB image from front camera (default: 240x320)

### Action Space

The action space is a dictionary with:

- `motor`: Box(low=-1, high=1, shape=(1,)) - Throttle command (-1 to 1)
- `steering`: Box(low=-1, high=1, shape=(1,)) - Steering angle (-1 to 1)

**Note**: Actions are normalized between -1 and 1.

### Reset Modes

Two reset modes are available:

- `'grid'`: Places the agent at predefined starting positions
- `'random'`: Places the agent at random positions on the track

Usage:
```python
obs, info = env.reset(options=dict(mode='grid'))
# or
obs, info = env.reset(options=dict(mode='random'))
```

### Render Modes

Three rendering modes are available:

- `'human'`: Renders the scene in a window (interactive visualization)
- `'rgb_array_birds_eye'`: Returns RGB array from bird's eye perspective
- `'rgb_array_follow'`: Returns RGB array from third-person following view
- `None`: No rendering (fastest, for training)

## Example Usage

### Basic Random Actor

```bash
# Run 5 episodes on Austria track with visualization
python random_actor.py --env SingleAgentAustria-v0 --episodes 5
```

### Interactive Testing

```bash
# Start interactive menu
python run_racecar_gym.py

# Select mode 2 for keyboard control
# Use WASD or arrow keys to drive
```

### Training Data Collection

```bash
# Run without rendering for faster data collection
python random_actor.py --env SingleAgentAustria-v0 --episodes 100 --render-mode none
```

## Troubleshooting

### racecar_gym Not Found

If you get `ImportError: No module named 'racecar_gym'`:

1. Make sure you've installed it from source:
   ```bash
   cd racecar_gym
   pip install -e .
   ```

2. Verify installation:
   ```bash
   python -c "import racecar_gym; print(racecar_gym.__file__)"
   ```

### Tracks Not Downloading

If tracks don't download automatically:

1. Manually download tracks (see Installation section)
2. Check your internet connection
3. Verify the `racecar_gym/models/scenes/` directory exists

### Rendering Issues

If rendering doesn't work:

1. Make sure you have a display (for `render_mode='human'`)
2. For headless servers, use `render_mode='rgb_array_birds_eye'` or `None`
3. Check that PyBullet is properly installed

### Environment Not Found

If you get errors about environment IDs:

1. Run `python run_racecar_gym.py` and select mode 6 to list available environments
2. Make sure you're using the correct environment ID format: `SingleAgent{TrackName}-v0`
3. Verify the track name is spelled correctly (case-sensitive)

## Project Structure

```
Speedracer_rl/
├── random_actor.py          # Random policy runner
├── run_racecar_gym.py      # Interactive testing script
├── dqn.py                  # DQN agent implementation
├── replay_buffer.py        # Experience replay buffer
├── run_policy.py           # Policy evaluation script
├── racecar_gym/            # racecar_gym package (installed from source)
│   ├── racecar_gym/        # Main package
│   ├── examples/          # Example scripts
│   ├── scenarios/          # Track scenario files
│   └── models/             # 3D models and tracks
└── data/                   # Training data and checkpoints
```

## Additional Resources

- [racecar_gym GitHub](https://github.com/axelbr/racecar_gym)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyBullet Documentation](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/)

## License

Check individual package licenses. The racecar_gym package has its own license in `racecar_gym/LICENSE`.

