#!/usr/bin/env python3
"""
Run SAC agent using CNN encoder for vision-based control.
The agent sees only images, not the actual state coordinates.

Pipeline:
  CartPole RGB frames -> CNN encoder -> predicted state -> SAC agent -> action
"""

import numpy as np
import torch
import gymnasium as gym
import cv2
import argparse
from collections import deque

from agent import SAC
from vision_network import ResNetStateEncoder


def load_models(agent_path, cnn_path, norm_path, device):
    """Load SAC agent and CNN encoder."""
    # Load SAC agent
    agent = SAC(
        state_size=64,  # 16 frames * 4 state dims
        action_size=2,  # CartPole has 2 actions
        device=device
    )
    agent.actor_local.load_state_dict(torch.load(agent_path, map_location=device))
    agent.actor_local.eval()
    print(f"Loaded SAC agent from {agent_path}")

    # Load CNN encoder
    cnn = ResNetStateEncoder(input_channels=48, output_dim=64).to(device)
    cnn.load_state_dict(torch.load(cnn_path, map_location=device))
    cnn.eval()
    print(f"Loaded CNN encoder from {cnn_path}")

    # Load normalization stats
    norm_stats = torch.load(norm_path, map_location=device)
    state_mean = norm_stats['mean'].to(device)
    state_std = norm_stats['std'].to(device)
    print(f"Loaded normalization stats from {norm_path}")

    return agent, cnn, state_mean, state_std


def frames_to_tensor(frame_buffer, device):
    """Convert frame buffer to CNN input tensor.

    Args:
        frame_buffer: deque of 16 frames, each (224, 224, 3) uint8
        device: torch device

    Returns:
        tensor of shape (1, 48, 224, 224)
    """
    frames = np.array(frame_buffer, dtype=np.uint8)  # (16, 224, 224, 3)
    t = torch.from_numpy(frames).float() / 255.0
    t = t.permute(0, 3, 1, 2)  # (16, 3, 224, 224)
    t = t.reshape(48, 224, 224)  # (48, 224, 224)
    t = t.unsqueeze(0).to(device)  # (1, 48, 224, 224)
    return t


def run_episode(env, agent, cnn, state_mean, state_std, device, render=False, max_steps=500):
    """Run a single episode using vision-based control."""
    obs, info = env.reset()

    # Initialize frame buffer
    frame_buffer = deque(maxlen=16)
    frame = env.render()
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    for _ in range(16):
        frame_buffer.append(frame)

    # Get initial state from CNN
    with torch.no_grad():
        frame_tensor = frames_to_tensor(frame_buffer, device)
        predicted_state = cnn(frame_tensor)
        # Denormalize (CNN was trained on normalized states)
        state = (predicted_state * state_std + state_mean).squeeze(0).cpu().numpy()

    total_reward = 0
    steps = 0
    done = False

    while not done and steps < max_steps:
        # Get action from agent
        action = agent.get_action(state, training=False)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        if not done:
            # Get next frame
            frame = env.render()
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            frame_buffer.append(frame)

            # Predict next state from CNN
            with torch.no_grad():
                frame_tensor = frames_to_tensor(frame_buffer, device)
                predicted_state = cnn(frame_tensor)
                state = (predicted_state * state_std + state_mean).squeeze(0).cpu().numpy()

    return total_reward, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_path', type=str,
                        default='trained_models/grid_search_073SAC_discrete0.pth',
                        help='Path to SAC agent weights')
    parser.add_argument('--cnn_path', type=str,
                        default='trained_cnn/best_model.pth',
                        help='Path to CNN encoder weights')
    parser.add_argument('--norm_path', type=str,
                        default='trained_cnn/norm_stats.pt',
                        help='Path to normalization stats')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Max steps per episode')
    parser.add_argument('--render', action='store_true',
                        help='Render to screen (slower)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    agent, cnn, state_mean, state_std = load_models(
        args.agent_path, args.cnn_path, args.norm_path, device
    )

    # Create environment
    render_mode = "human" if args.render else "rgb_array"
    env = gym.make("CartPole-v1", render_mode=render_mode)

    # Run episodes
    print(f"\nRunning {args.episodes} episodes with vision-based control...")
    rewards = []
    steps_list = []

    for ep in range(args.episodes):
        reward, steps = run_episode(
            env, agent, cnn, state_mean, state_std, device,
            render=args.render, max_steps=args.max_steps
        )
        rewards.append(reward)
        steps_list.append(steps)
        print(f"  Episode {ep+1}: reward={reward:.0f}, steps={steps}")

    env.close()

    # Summary
    print(f"\nResults over {args.episodes} episodes:")
    print(f"  Mean reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    print(f"  Mean steps:  {np.mean(steps_list):.1f} +/- {np.std(steps_list):.1f}")
    print(f"  Max reward:  {np.max(rewards):.0f}")
    print(f"  Min reward:  {np.min(rewards):.0f}")


if __name__ == "__main__":
    main()
