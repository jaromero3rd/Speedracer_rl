#!/usr/bin/env python3
"""
Record videos of trained CartPole agents playing.
This script loads saved models and records their gameplay for generating training data.
"""

import numpy as np
import gymnasium as gym
import torch
import argparse
import os
import cv2
from collections import deque
from agent import SAC
from gymnasium.wrappers import TransformObservation


def record_agent_videos(model_path: str, output_dir: str, num_episodes: int = 10,
                        max_steps: int = 500, obs_buffer_max_len: int = 4,
                        resize_shape: tuple = None):
    """
    Load a trained agent and record videos of it playing CartPole.

    Args:
        model_path: Path to saved actor model (.pth file)
        output_dir: Directory to save videos
        num_episodes: Number of episodes to record
        max_steps: Maximum steps per episode
        obs_buffer_max_len: Length of observation buffer (must match training)
    """
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    video_subdir = os.path.join(output_dir, model_name)
    os.makedirs(video_subdir, exist_ok=True)

    print(f"=" * 60)
    print(f"Recording agent: {model_name}")
    print(f"Model path: {model_path}")
    print(f"Output directory: {video_subdir}")
    print(f"Episodes: {num_episodes}")
    print(f"=" * 60)

    # Create environment with video recording
    env = gym.make('CartPole-v1', render_mode='rgb_array')

    # Add resize wrapper if specified
    if resize_shape is not None:
        print(f"Resizing frames to: {resize_shape[0]}x{resize_shape[1]}")

        # Create a custom wrapper to resize the rendered frames
        class ResizeRender(gym.Wrapper):
            def __init__(self, env, size):
                super().__init__(env)
                self.size = size

            def render(self):
                frame = self.env.render()
                if frame is not None:
                    # Resize using cv2.INTER_AREA (best for downsampling)
                    return cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
                return frame

        env = ResizeRender(env, resize_shape)

    env = gym.wrappers.RecordVideo(
        env,
        video_subdir,
        episode_trigger=lambda x: True,  # Record all episodes
        name_prefix=f"{model_name}_episode"
    )

    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get sample observation to determine state size
    sample_obs, _ = env.reset()
    state_size = sample_obs.shape[0] * obs_buffer_max_len
    action_size = env.action_space.n

    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Observation buffer length: {obs_buffer_max_len}\n")

    # Create SAC agent
    agent = SAC(
        state_size=state_size,
        action_size=action_size,
        device=device,
        learning_rate=5e-4,  # Not used for inference
        entropy_bonus=None,  # Learnable alpha (matches training default)
        epsilon=0.0  # No exploration during evaluation
    )

    # Load the saved actor model
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        agent.actor_local.load_state_dict(torch.load(model_path, map_location=device))
        agent.actor_local.eval()  # Set to evaluation mode
        print(f"✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return None

    episode_rewards = []
    episode_lengths = []

    print("Starting video recording...\n")

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}", end=" ")

        # Reset environment
        obs, info = env.reset()

        # Initialize observation buffer
        obs_buffer = deque(maxlen=obs_buffer_max_len)

        # Fill buffer with initial observation
        for _ in range(obs_buffer_max_len):
            obs_buffer.append(obs)
        state = np.stack(obs_buffer, axis=0).flatten(order="C")

        total_reward = 0.0
        step_count = 0
        done = False

        while not done and step_count < max_steps:
            # Get action from agent (deterministic policy)
            action = agent.get_action(state, training=False)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update observation buffer
            obs_buffer.append(next_obs)
            next_state = np.stack(obs_buffer, axis=0).flatten(order="C")

            obs = next_obs
            state = next_state
            total_reward += reward
            step_count += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)

        print(f"- Steps: {step_count}, Reward: {total_reward:.2f}")

    env.close()

    # Print statistics
    print("\n" + "-" * 60)
    print(f"Agent: {model_name}")
    print(f"  Episodes recorded: {num_episodes}")
    print(f"  Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Min reward: {np.min(episode_rewards):.2f}")
    print(f"  Max reward: {np.max(episode_rewards):.2f}")
    print(f"  Videos saved to: {video_subdir}")
    print("-" * 60 + "\n")

    return {
        'model_name': model_name,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'num_episodes': num_episodes,
        'video_dir': video_subdir
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Record CartPole videos from trained agents')
    parser.add_argument('--models', type=str, nargs='+',
                        default=[
                            'trained_models/grid_search_005SAC_discrete0.pth',
                            'trained_models/grid_search_047SAC_discrete0.pth',
                            'trained_models/grid_search_073SAC_discrete0.pth'
                        ],
                        help='Paths to saved actor model files (default: 3 grid search models)')
    parser.add_argument('--output_dir', type=str, default='cartpole_videos',
                        help='Directory to save videos (default: cartpole_videos)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to record per agent (default: 10)')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Maximum steps per episode (default: 500)')
    parser.add_argument('--obs_buffer_len', type=int, default=4,
                        help='Observation buffer length (must match training, default: 4)')
    parser.add_argument('--resize', type=int, nargs=2, default=[224, 224],
                        help='Resize video frames to WIDTH HEIGHT (default: 224 224)')
    parser.add_argument('--no_resize', action='store_true',
                        help='Do not resize frames, keep original 600x400 resolution')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine resize shape
    resize_shape = None if args.no_resize else tuple(args.resize)

    print("\n" + "=" * 60)
    print("CartPole Video Recording")
    print("=" * 60)
    print(f"Number of agents: {len(args.models)}")
    print(f"Episodes per agent: {args.episodes}")
    print(f"Output directory: {args.output_dir}")
    if resize_shape:
        print(f"Video resolution: {resize_shape[0]}x{resize_shape[1]}")
    else:
        print(f"Video resolution: 600x400 (original)")
    print("=" * 60 + "\n")

    results = []

    try:
        for model_path in args.models:
            result = record_agent_videos(
                model_path=model_path,
                output_dir=args.output_dir,
                num_episodes=args.episodes,
                max_steps=args.max_steps,
                obs_buffer_max_len=args.obs_buffer_len,
                resize_shape=resize_shape
            )
            if result:
                results.append(result)

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for result in results:
            print(f"{result['model_name']}:")
            print(f"  Mean reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
            print(f"  Mean length: {result['mean_length']:.1f}")
            print(f"  Videos: {result['video_dir']}")
            print()
        print(f"Total videos recorded: {sum(r['num_episodes'] for r in results)}")
        print(f"All videos saved to: {args.output_dir}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
