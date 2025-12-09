#!/usr/bin/env python3
"""
Generate labeled vision dataset for supervised CNN training.
Records frames from trained agents and saves corresponding state buffers.

Output format:
    frame_stacks: (N, 16, 224, 224, 3) - uint8 RGB images, channels-last
    states: (N, 64) - float32, flattened observation buffer (16 timesteps x 4 state dims)

The CNN will learn to predict the 64-dim state buffer from the 16 stacked frames.
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
                     max_steps: int = 500, agent_obs_buffer_len: int = 16,
                     resize_shape: tuple = (224, 224), frame_stack: int = 16,
                     output_state_dim: int = 64):
    """
    Generate a dataset of (stacked frames, state buffer) pairs from a trained agent.

    Args:
        model_path: Path to saved actor model (.pth file)
        output_dir: Directory to save dataset
        num_episodes: Number of episodes to record
        max_steps: Maximum steps per episode
        agent_obs_buffer_len: Length of observation buffer for THIS agent (must match its training)
        resize_shape: Size to resize images (width, height)
        frame_stack: Number of consecutive frames to stack for CNN input (always 16)
        output_state_dim: Dimension of output state buffer for CNN labels (always 64)

    Returns:
        Dictionary with dataset statistics

    Note:
        frame_stack and output_state_dim should always be 16 and 64 respectively,
        matching the target agent (073). agent_obs_buffer_len varies per agent.
    """
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    dataset_dir = os.path.join(output_dir, model_name)

    # Create directories
    images_dir = os.path.join(dataset_dir, 'images')
    states_dir = os.path.join(dataset_dir, 'states')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(states_dir, exist_ok=True)

    # CNN output buffer is always 16 timesteps (64 dim)
    cnn_obs_buffer_len = output_state_dim // 4  # 64 // 4 = 16

    print(f"=" * 80)
    print(f"Generating dataset from: {model_name}")
    print(f"=" * 80)
    print(f"Model path: {model_path}")
    print(f"Output directory: {dataset_dir}")
    print(f"Episodes: {num_episodes}")
    print(f"Image size: {resize_shape[0]}x{resize_shape[1]}")
    print(f"Frame stack: {frame_stack} frames (CNN input)")
    print(f"Agent obs buffer: {agent_obs_buffer_len} (for running this agent)")
    print(f"CNN output dim: {output_state_dim} ({cnn_obs_buffer_len} timesteps x 4)")
    print(f"=" * 80 + "\n")

    # Create environment with RGB rendering
    env = gym.make('CartPole-v1', render_mode='rgb_array')

    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get sample observation to determine state size
    sample_obs, _ = env.reset()
    agent_state_size = sample_obs.shape[0] * agent_obs_buffer_len  # For running the agent
    action_size = env.action_space.n

    print(f"Agent state size: {agent_state_size} ({agent_obs_buffer_len} x 4)")
    print(f"Action size: {action_size}")

    # Create SAC agent with its specific state size
    agent = SAC(
        state_size=agent_state_size,
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

    # Storage for current chunk
    chunk_frame_stacks = []
    chunk_states = []
    episode_rewards = []
    episode_lengths = []

    total_frames = 0
    chunk_idx = 0
    episodes_per_chunk = 20  # Save every 20 episodes to limit memory usage

    print("Generating dataset...\n")

    for episode in tqdm(range(num_episodes), desc="Episodes"):
        # Reset environment
        obs, info = env.reset()

        # Initialize observation buffer for AGENT (variable size per agent)
        agent_obs_buffer = deque(maxlen=agent_obs_buffer_len)
        for _ in range(agent_obs_buffer_len):
            agent_obs_buffer.append(obs)
        agent_state = np.stack(agent_obs_buffer, axis=0).flatten(order="C")

        # Initialize observation buffer for CNN OUTPUT (always 16 timesteps = 64 dim)
        cnn_obs_buffer = deque(maxlen=cnn_obs_buffer_len)
        for _ in range(cnn_obs_buffer_len):
            cnn_obs_buffer.append(obs)

        # Initialize frame buffer (for CNN input, always 16 frames)
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
            stacked_frames = np.array(frame_buffer, dtype=np.uint8)

            # Get CNN label: flattened 16-timestep observation buffer (64 dim)
            cnn_state = np.stack(cnn_obs_buffer, axis=0).flatten(order="C").astype(np.float32)

            # Store stacked frames and CNN state label in chunk
            chunk_frame_stacks.append(stacked_frames)
            chunk_states.append(cnn_state)

            # Get action from agent using ITS state representation
            action = agent.get_action(agent_state, training=False)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update BOTH observation buffers
            agent_obs_buffer.append(next_obs)
            cnn_obs_buffer.append(next_obs)

            agent_state = np.stack(agent_obs_buffer, axis=0).flatten(order="C")

            obs = next_obs
            episode_reward += reward
            step_count += 1
            total_frames += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)

        # Save chunk every N episodes to avoid memory issues
        if (episode + 1) % episodes_per_chunk == 0:
            chunk_frames_arr = np.array(chunk_frame_stacks, dtype=np.uint8)
            chunk_states_arr = np.array(chunk_states, dtype=np.float32)

            chunk_file = os.path.join(dataset_dir, f'chunk_{chunk_idx:04d}.npz')
            np.savez_compressed(chunk_file, frame_stacks=chunk_frames_arr, states=chunk_states_arr)

            print(f"  Saved chunk {chunk_idx}: {len(chunk_frame_stacks)} samples")

            # Clear chunk storage
            chunk_frame_stacks = []
            chunk_states = []
            chunk_idx += 1

    # Save any remaining data
    if len(chunk_frame_stacks) > 0:
        chunk_frames_arr = np.array(chunk_frame_stacks, dtype=np.uint8)
        chunk_states_arr = np.array(chunk_states, dtype=np.float32)

        chunk_file = os.path.join(dataset_dir, f'chunk_{chunk_idx:04d}.npz')
        np.savez_compressed(chunk_file, frame_stacks=chunk_frames_arr, states=chunk_states_arr)

        print(f"  Saved chunk {chunk_idx}: {len(chunk_frame_stacks)} samples")
        chunk_idx += 1

    env.close()

    print(f"\n" + "=" * 80)
    print(f"Dataset Generation Complete!")
    print(f"=" * 80)
    print(f"Total frames collected: {total_frames}")
    print(f"Total chunks saved: {chunk_idx}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"=" * 80 + "\n")

    # Compute state stats by loading chunks one at a time
    print("Computing state statistics...")
    all_states_for_stats = []
    for i in range(chunk_idx):
        chunk_file = os.path.join(dataset_dir, f'chunk_{i:04d}.npz')
        chunk_data = np.load(chunk_file)
        all_states_for_stats.append(chunk_data['states'])
    all_states_for_stats = np.concatenate(all_states_for_stats, axis=0)

    # Save metadata
    metadata = {
        'model_name': model_name,
        'num_episodes': num_episodes,
        'total_frames': total_frames,
        'num_chunks': chunk_idx,
        'frame_stack_shape': [frame_stack, resize_shape[1], resize_shape[0], 3],
        'state_dim': output_state_dim,
        'agent_obs_buffer_len': agent_obs_buffer_len,
        'cnn_obs_buffer_len': cnn_obs_buffer_len,
        'resize_shape': resize_shape,
        'frame_stack': frame_stack,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'state_labels': ['cart_position', 'cart_velocity', 'pole_angle', 'pole_angular_velocity'] * cnn_obs_buffer_len,
        'state_format': f'flattened observation buffer: {cnn_obs_buffer_len} timesteps x 4 state dims = {output_state_dim} total'
    }

    with open(os.path.join(dataset_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save statistics about the state distributions (useful for normalization)
    state_stats = {
        'mean': all_states_for_stats.mean(axis=0).tolist(),
        'std': all_states_for_stats.std(axis=0).tolist(),
        'min': all_states_for_stats.min(axis=0).tolist(),
        'max': all_states_for_stats.max(axis=0).tolist()
    }

    with open(os.path.join(dataset_dir, 'state_stats.json'), 'w') as f:
        json.dump(state_stats, f, indent=2)

    # Clean up stats array
    del all_states_for_stats

    print(f"✓ Dataset saved to: {dataset_dir}")
    print(f"  - {chunk_idx} chunk files (chunk_0000.npz, chunk_0001.npz, ...)")
    print(f"  - metadata.json")
    print(f"  - state_stats.json")
    print(f"\nDataset format (per chunk):")
    print(f"  frame_stacks: (N, {frame_stack}, {resize_shape[1]}, {resize_shape[0]}, 3)")
    print(f"  states: (N, {output_state_dim}) [{cnn_obs_buffer_len} timesteps x 4 state dims, flattened]")

    # Print state statistics (first 4 dims only for brevity)
    print(f"\nState Statistics (first 4 dims):")
    print(f"  Mean: {state_stats['mean'][:4]}")
    print(f"  Std:  {state_stats['std'][:4]}")
    print("=" * 80 + "\n")

    return metadata


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate vision dataset from trained CartPole agents')
    parser.add_argument('--output_dir', type=str, default='vision_dataset',
                        help='Directory to save datasets (default: vision_dataset)')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of episodes to record per agent (default: 50)')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Maximum steps per episode (default: 500)')
    parser.add_argument('--resize', type=int, nargs=2, default=[224, 224],
                        help='Resize images to WIDTH HEIGHT (default: 224 224)')

    args = parser.parse_args()

    # Define agents with their specific obs_buffer_len from grid search
    # Model 005: obs_buffer=2, Model 047: obs_buffer=8, Model 073: obs_buffer=16
    agents = [
        ('trained_models/grid_search_005SAC_discrete0.pth', 2),
        ('trained_models/grid_search_047SAC_discrete0.pth', 8),
        ('trained_models/grid_search_073SAC_discrete0.pth', 16),
    ]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    resize_shape = tuple(args.resize)

    # Fixed CNN parameters (always 16 frames -> 64 dim output)
    frame_stack = 16
    output_state_dim = 64

    print("\n" + "=" * 80)
    print("CartPole Vision Dataset Generation")
    print("=" * 80)
    print(f"Number of agents: {len(agents)}")
    print(f"Episodes per agent: {args.episodes}")
    print(f"Image resolution: {resize_shape[0]}x{resize_shape[1]}")
    print(f"Frame stack: {frame_stack} consecutive frames (CNN input)")
    print(f"Output state dim: {output_state_dim} (CNN output)")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80 + "\n")

    results = []

    try:
        for model_path, agent_obs_buffer_len in agents:
            result = generate_dataset(
                model_path=model_path,
                output_dir=args.output_dir,
                num_episodes=args.episodes,
                max_steps=args.max_steps,
                agent_obs_buffer_len=agent_obs_buffer_len,
                resize_shape=resize_shape,
                frame_stack=frame_stack,
                output_state_dim=output_state_dim
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
