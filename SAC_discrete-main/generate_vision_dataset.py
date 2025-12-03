#!/usr/bin/env python3
"""
Generate labeled vision dataset for CNN pre-training.
Records videos of trained agents and saves corresponding state information.
Each frame is paired with the true CartPole state: [x, x_dot, theta, theta_dot]
"""

import numpy as np
import gymnasium as gym
import torch
import argparse
import os
import cv2
import json
from collections import deque
from agent import SAC
from tqdm import tqdm
import pickle


def generate_dataset(model_path: str, output_dir: str, num_episodes: int = 50,
                     max_steps: int = 500, obs_buffer_max_len: int = 4,
                     resize_shape: tuple = (224, 224), frame_stack: int = 4):
    """
    Generate a dataset of (image, state) pairs from a trained agent.

    Args:
        model_path: Path to saved actor model (.pth file)
        output_dir: Directory to save dataset
        num_episodes: Number of episodes to record
        max_steps: Maximum steps per episode
        obs_buffer_max_len: Length of observation buffer (must match training)
        resize_shape: Size to resize images (width, height)
        frame_stack: Number of consecutive frames to stack (for velocity information)

    Returns:
        Dictionary with dataset statistics
    """
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    dataset_dir = os.path.join(output_dir, model_name)

    # Create directories
    images_dir = os.path.join(dataset_dir, 'images')
    states_dir = os.path.join(dataset_dir, 'states')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(states_dir, exist_ok=True)

    print(f"=" * 80)
    print(f"Generating dataset from: {model_name}")
    print(f"=" * 80)
    print(f"Model path: {model_path}")
    print(f"Output directory: {dataset_dir}")
    print(f"Episodes: {num_episodes}")
    print(f"Image size: {resize_shape[0]}x{resize_shape[1]}")
    print(f"Frame stack: {frame_stack} frames")
    print(f"=" * 80 + "\n")

    # Create environment with RGB rendering
    env = gym.make('CartPole-v1', render_mode='rgb_array')

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
        learning_rate=5e-4,
        entropy_bonus=None,
        epsilon=0.0
    )

    # Load the saved actor model
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        agent.actor_local.load_state_dict(torch.load(model_path, map_location=device))
        agent.actor_local.eval()
        print(f"✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return None

    # Storage for dataset
    all_frame_stacks = []
    all_states = []
    episode_rewards = []
    episode_lengths = []

    total_frames = 0

    print("Generating dataset...\n")

    for episode in tqdm(range(num_episodes), desc="Episodes"):
        # Reset environment
        obs, info = env.reset()

        # Initialize observation buffer (for RL agent)
        obs_buffer = deque(maxlen=obs_buffer_max_len)

        # Fill buffer with initial observation
        for _ in range(obs_buffer_max_len):
            obs_buffer.append(obs)
        state = np.stack(obs_buffer, axis=0).flatten(order="C")

        # Initialize frame buffer (for CNN input)
        frame_buffer = deque(maxlen=frame_stack)

        # Get initial frame and fill frame buffer
        initial_frame = env.render()
        if resize_shape is not None:
            initial_frame = cv2.resize(initial_frame, resize_shape, interpolation=cv2.INTER_AREA)

        for _ in range(frame_stack):
            frame_buffer.append(initial_frame)

        episode_reward = 0.0
        step_count = 0
        done = False

        while not done and step_count < max_steps:
            # Get rendered frame
            frame = env.render()

            # Resize frame
            if resize_shape is not None:
                frame = cv2.resize(frame, resize_shape, interpolation=cv2.INTER_AREA)

            # Add frame to buffer
            frame_buffer.append(frame)

            # Stack frames: (frame_stack, H, W, 3)
            stacked_frames = np.array(frame_buffer)

            # Store stacked frames and current state
            # Note: obs is the actual CartPole state [x, x_dot, theta, theta_dot]
            all_frame_stacks.append(stacked_frames)
            all_states.append(obs.copy())

            # Get action from agent
            action = agent.get_action(state, training=False)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update observation buffer
            obs_buffer.append(next_obs)
            next_state = np.stack(obs_buffer, axis=0).flatten(order="C")

            obs = next_obs
            state = next_state
            episode_reward += reward
            step_count += 1
            total_frames += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)

    env.close()

    # Convert to numpy arrays
    all_frame_stacks = np.array(all_frame_stacks, dtype=np.uint8)  # Shape: (N, frame_stack, H, W, 3)
    all_states = np.array(all_states, dtype=np.float32)  # Shape: (N, 4)

    print(f"\n" + "=" * 80)
    print(f"Dataset Generation Complete!")
    print(f"=" * 80)
    print(f"Total frames collected: {total_frames}")
    print(f"Frame stacks shape: {all_frame_stacks.shape}")
    print(f"States shape: {all_states.shape}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"=" * 80 + "\n")

    # Save dataset
    print("Saving dataset...")

    # Save as compressed numpy arrays
    np.savez_compressed(
        os.path.join(dataset_dir, 'dataset.npz'),
        frame_stacks=all_frame_stacks,
        states=all_states
    )

    # Save metadata
    metadata = {
        'model_name': model_name,
        'num_episodes': num_episodes,
        'total_frames': total_frames,
        'frame_stack_shape': list(all_frame_stacks.shape[1:]),
        'state_dim': all_states.shape[1],
        'resize_shape': resize_shape,
        'frame_stack': frame_stack,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'state_labels': ['cart_position', 'cart_velocity', 'pole_angle', 'pole_angular_velocity']
    }

    with open(os.path.join(dataset_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save statistics about the state distributions (useful for normalization)
    state_stats = {
        'mean': all_states.mean(axis=0).tolist(),
        'std': all_states.std(axis=0).tolist(),
        'min': all_states.min(axis=0).tolist(),
        'max': all_states.max(axis=0).tolist()
    }

    with open(os.path.join(dataset_dir, 'state_stats.json'), 'w') as f:
        json.dump(state_stats, f, indent=2)

    print(f"✓ Dataset saved to: {dataset_dir}")
    print(f"  - dataset.npz (frame stacks and states)")
    print(f"  - metadata.json")
    print(f"  - state_stats.json")
    print(f"\nDataset format:")
    print(f"  frame_stacks: {all_frame_stacks.shape} - (N, {frame_stack}, {resize_shape[1]}, {resize_shape[0]}, 3)")
    print(f"  states: {all_states.shape} - (N, 4) [x, x_dot, theta, theta_dot]")

    # Print state statistics
    print(f"\nState Statistics:")
    print(f"  Mean: {state_stats['mean']}")
    print(f"  Std:  {state_stats['std']}")
    print(f"  Min:  {state_stats['min']}")
    print(f"  Max:  {state_stats['max']}")
    print("=" * 80 + "\n")

    return metadata


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate vision dataset from trained CartPole agents')
    parser.add_argument('--models', type=str, nargs='+',
                        default=[
                            'trained_models/grid_search_005SAC_discrete0.pth',
                            'trained_models/grid_search_047SAC_discrete0.pth',
                            'trained_models/grid_search_073SAC_discrete0.pth'
                        ],
                        help='Paths to saved actor model files')
    parser.add_argument('--output_dir', type=str, default='vision_dataset',
                        help='Directory to save datasets (default: vision_dataset)')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of episodes to record per agent (default: 50)')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Maximum steps per episode (default: 500)')
    parser.add_argument('--obs_buffer_len', type=int, default=16,
                        help='Observation buffer length (must match training, default: 16)')
    parser.add_argument('--resize', type=int, nargs=2, default=[224, 224],
                        help='Resize images to WIDTH HEIGHT (default: 224 224)')
    parser.add_argument('--frame_stack', type=int, default=16,
                        help='Number of consecutive frames to stack (default: 16)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    resize_shape = tuple(args.resize)

    print("\n" + "=" * 80)
    print("CartPole Vision Dataset Generation")
    print("=" * 80)
    print(f"Number of agents: {len(args.models)}")
    print(f"Episodes per agent: {args.episodes}")
    print(f"Image resolution: {resize_shape[0]}x{resize_shape[1]}")
    print(f"Frame stack: {args.frame_stack} consecutive frames")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80 + "\n")

    results = []

    try:
        for model_path in args.models:
            result = generate_dataset(
                model_path=model_path,
                output_dir=args.output_dir,
                num_episodes=args.episodes,
                max_steps=args.max_steps,
                obs_buffer_max_len=args.obs_buffer_len,
                resize_shape=resize_shape,
                frame_stack=args.frame_stack
            )
            if result:
                results.append(result)

        # Print summary
        print("\n" + "=" * 80)
        print("DATASET GENERATION SUMMARY")
        print("=" * 80)
        total_frames = 0
        for result in results:
            print(f"\n{result['model_name']}:")
            print(f"  Episodes: {result['num_episodes']}")
            print(f"  Total frames: {result['total_frames']}")
            print(f"  Mean reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
            print(f"  Mean length: {result['mean_length']:.1f}")
            total_frames += result['total_frames']

        print(f"\n{'=' * 80}")
        print(f"TOTAL FRAMES ACROSS ALL AGENTS: {total_frames}")
        print(f"All datasets saved to: {args.output_dir}")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
