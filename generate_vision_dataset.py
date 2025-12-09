#!/usr/bin/env python3
"""
Generate labeled dataset for CNN training.
Records frames from trained agents with corresponding state buffers.
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


def generate_dataset(model_path, output_dir, num_episodes=50, max_steps=500,
                     agent_obs_buffer_len=16, resize_shape=(224, 224)):
    """
    Generate (stacked frames, state buffer) pairs from a trained agent.
    CNN input is always 16 frames, output is always 64-dim state.
    """
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    dataset_dir = os.path.join(output_dir, model_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # always use 16 timesteps for CNN output
    cnn_obs_buffer_len = 16
    output_state_dim = 64

    print(f"\nGenerating dataset: {model_name}")
    print(f"  Episodes: {num_episodes}, Agent buffer: {agent_obs_buffer_len}")

    env = gym.make('CartPole-v1', render_mode='rgb_array')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sample_obs, _ = env.reset()
    agent_state_size = sample_obs.shape[0] * agent_obs_buffer_len
    action_size = env.action_space.n

    # load agent
    agent = SAC(
        state_size=agent_state_size,
        action_size=action_size,
        device=device,
        learning_rate=5e-4,
        entropy_bonus=None,
        epsilon=0.0
    )

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        env.close()
        return None

    agent.actor_local.load_state_dict(torch.load(model_path, map_location=device))
    agent.actor_local.eval()
    print(f"  Loaded model")

    chunk_frame_stacks = []
    chunk_states = []
    episode_rewards = []
    episode_lengths = []

    total_frames = 0
    chunk_idx = 0
    episodes_per_chunk = 20

    for episode in tqdm(range(num_episodes), desc="Episodes"):
        obs, info = env.reset()

        # agent's observation buffer (varies per agent)
        agent_obs_buffer = deque(maxlen=agent_obs_buffer_len)
        for _ in range(agent_obs_buffer_len):
            agent_obs_buffer.append(obs)
        agent_state = np.stack(agent_obs_buffer, axis=0).flatten(order="C")

        # CNN output buffer (always 16)
        cnn_obs_buffer = deque(maxlen=cnn_obs_buffer_len)
        for _ in range(cnn_obs_buffer_len):
            cnn_obs_buffer.append(obs)

        # frame buffer for CNN input
        frame_buffer = deque(maxlen=16)
        initial_frame = env.render()
        if resize_shape:
            initial_frame = cv2.resize(initial_frame, resize_shape, interpolation=cv2.INTER_AREA)
        for _ in range(16):
            frame_buffer.append(initial_frame)

        episode_reward = 0.0
        step_count = 0
        done = False

        while not done and step_count < max_steps:
            frame = env.render()
            if resize_shape:
                frame = cv2.resize(frame, resize_shape, interpolation=cv2.INTER_AREA)

            frame_buffer.append(frame)
            stacked_frames = np.array(frame_buffer, dtype=np.uint8)
            cnn_state = np.stack(cnn_obs_buffer, axis=0).flatten(order="C").astype(np.float32)

            chunk_frame_stacks.append(stacked_frames)
            chunk_states.append(cnn_state)

            action = agent.get_action(agent_state, training=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent_obs_buffer.append(next_obs)
            cnn_obs_buffer.append(next_obs)
            agent_state = np.stack(agent_obs_buffer, axis=0).flatten(order="C")

            obs = next_obs
            episode_reward += reward
            step_count += 1
            total_frames += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)

        # save chunk periodically to avoid OOM
        if (episode + 1) % episodes_per_chunk == 0:
            chunk_frames_arr = np.array(chunk_frame_stacks, dtype=np.uint8)
            chunk_states_arr = np.array(chunk_states, dtype=np.float32)
            chunk_file = os.path.join(dataset_dir, f'chunk_{chunk_idx:04d}.npz')
            np.savez_compressed(chunk_file, frame_stacks=chunk_frames_arr, states=chunk_states_arr)
            print(f"  Saved chunk {chunk_idx}: {len(chunk_frame_stacks)} samples")
            chunk_frame_stacks = []
            chunk_states = []
            chunk_idx += 1

    # save remaining
    if len(chunk_frame_stacks) > 0:
        chunk_frames_arr = np.array(chunk_frame_stacks, dtype=np.uint8)
        chunk_states_arr = np.array(chunk_states, dtype=np.float32)
        chunk_file = os.path.join(dataset_dir, f'chunk_{chunk_idx:04d}.npz')
        np.savez_compressed(chunk_file, frame_stacks=chunk_frames_arr, states=chunk_states_arr)
        chunk_idx += 1

    env.close()

    print(f"  Total frames: {total_frames}, chunks: {chunk_idx}")
    print(f"  Avg reward: {np.mean(episode_rewards):.1f}, avg length: {np.mean(episode_lengths):.1f}")

    # compute state stats for normalization
    all_states = []
    for i in range(chunk_idx):
        chunk_file = os.path.join(dataset_dir, f'chunk_{i:04d}.npz')
        data = np.load(chunk_file)
        all_states.append(data['states'])
    all_states = np.concatenate(all_states, axis=0)

    # save metadata
    metadata = {
        'model_name': model_name,
        'num_episodes': num_episodes,
        'total_frames': total_frames,
        'num_chunks': chunk_idx,
        'agent_obs_buffer_len': agent_obs_buffer_len,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
    }
    with open(os.path.join(dataset_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    state_stats = {
        'mean': all_states.mean(axis=0).tolist(),
        'std': all_states.std(axis=0).tolist(),
        'min': all_states.min(axis=0).tolist(),
        'max': all_states.max(axis=0).tolist()
    }
    with open(os.path.join(dataset_dir, 'state_stats.json'), 'w') as f:
        json.dump(state_stats, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='vision_dataset')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--resize', type=int, nargs=2, default=[224, 224])
    args = parser.parse_args()

    # agents with their obs_buffer_len from grid search
    agents = [
        ('trained_models/grid_search_005SAC_discrete0.pth', 2),
        ('trained_models/grid_search_047SAC_discrete0.pth', 8),
        ('trained_models/grid_search_073SAC_discrete0.pth', 16),
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    resize_shape = tuple(args.resize)

    print(f"Generating vision dataset")
    print(f"  Agents: {len(agents)}, Episodes each: {args.episodes}")

    results = []
    for model_path, obs_buffer_len in agents:
        result = generate_dataset(
            model_path=model_path,
            output_dir=args.output_dir,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            agent_obs_buffer_len=obs_buffer_len,
            resize_shape=resize_shape
        )
        if result:
            results.append(result)

    total = sum(r['total_frames'] for r in results)
    print(f"\nDone! Total frames: {total}")


if __name__ == "__main__":
    main()
