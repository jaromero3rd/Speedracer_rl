#!/usr/bin/env python3
"""
Record videos of the state-based SAC agents playing CartPole.
Uses ground truth coordinates, not vision.
"""

import numpy as np
import torch
import gymnasium as gym
import cv2
import argparse
import os
from collections import deque

from agent import SAC


def record_episode(env, agent, device, output_path, obs_buffer_len=16, fps=30, max_steps=500):
    """Record a single episode to video."""
    obs, info = env.reset()

    # Initialize observation buffer
    obs_buffer = deque(maxlen=obs_buffer_len)
    for _ in range(obs_buffer_len):
        obs_buffer.append(obs)
    state = np.stack(obs_buffer, axis=0).flatten()

    # Get full resolution frame for video
    full_frame = env.render()
    height, width = full_frame.shape[:2]

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write initial frame
    out.write(cv2.cvtColor(full_frame, cv2.COLOR_RGB2BGR))

    total_reward = 0
    steps = 0
    done = False

    while not done and steps < max_steps:
        action = agent.get_action(state, training=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        # Get frame for video
        full_frame = env.render()
        out.write(cv2.cvtColor(full_frame, cv2.COLOR_RGB2BGR))

        if not done:
            obs_buffer.append(next_obs)
            state = np.stack(obs_buffer, axis=0).flatten()

    out.release()
    return total_reward, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='videos')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--max_steps', type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # All three agents with their obs_buffer_len
    agents = [
        ('grid_search_005SAC_discrete0', 'trained_models/grid_search_005SAC_discrete0.pth', 2),
        ('grid_search_047SAC_discrete0', 'trained_models/grid_search_047SAC_discrete0.pth', 8),
        ('grid_search_073SAC_discrete0', 'trained_models/grid_search_073SAC_discrete0.pth', 16),
    ]

    env = gym.make("CartPole-v1", render_mode="rgb_array")

    for agent_name, agent_path, obs_buffer_len in agents:
        print(f"\nRecording {agent_name} (obs_buffer_len={obs_buffer_len})...")

        state_size = obs_buffer_len * 4
        agent = SAC(state_size=state_size, action_size=2, device=device)
        agent.actor_local.load_state_dict(torch.load(agent_path, map_location=device))
        agent.actor_local.eval()

        for ep in range(args.episodes):
            output_path = os.path.join(args.output_dir, f'state_agent_{agent_name}_ep{ep+1}.mp4')
            reward, steps = record_episode(
                env, agent, device, output_path, obs_buffer_len=obs_buffer_len,
                fps=args.fps, max_steps=args.max_steps
            )
            print(f"  Episode {ep+1}: reward={reward:.0f}, steps={steps}, saved to {output_path}")

    env.close()
    print(f"\nDone! Videos saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
