#!/usr/bin/env python3
"""
Unified script to run SAC agents on CartPole.

Supports two modes:
  1. State-based: Agent receives ground-truth state from environment
  2. Vision-based: Agent receives CNN-predicted state from RGB frames

Examples:
  # Run state-based agent (ground truth)
  python run_agent.py --mode state

  # Run vision-based agent (CNN predictions)
  python run_agent.py --mode vision

  # Compare both modes with same agent
  python run_agent.py --mode state --episodes 10
  python run_agent.py --mode vision --episodes 10

  # Use noise-trained agent (more robust to CNN errors)
  python run_agent.py --mode vision --agent noise2
"""

import numpy as np
import torch
import gymnasium as gym
import cv2
import argparse
import os
from collections import deque

from agent import SAC
from vision_network import ResNetStateEncoder


# Pre-configured agents with their observation buffer lengths
AGENTS = {
    '073': {
        'path': 'trained_models/grid_search_073SAC_discrete0.pth',
        'obs_buffer_len': 16,
        'description': 'Best performing agent (500 steps)'
    },
    'noise2': {
        'path': 'trained_models/noise2SAC_discrete0.pth',
        'obs_buffer_len': 16,
        'description': 'Noise-trained agent (robust to CNN errors)'
    },
    '047': {
        'path': 'trained_models/grid_search_047SAC_discrete0.pth',
        'obs_buffer_len': 8,
        'description': 'Agent with 8-frame buffer'
    },
    '005': {
        'path': 'trained_models/grid_search_005SAC_discrete0.pth',
        'obs_buffer_len': 2,
        'description': 'Agent with 2-frame buffer'
    },
}


def load_sac_agent(agent_path, state_size, device):
    """Load SAC agent."""
    agent = SAC(state_size=state_size, action_size=2, device=device)
    agent.actor_local.load_state_dict(torch.load(agent_path, map_location=device))
    agent.actor_local.eval()
    return agent


def load_cnn_encoder(cnn_path, norm_path, device):
    """Load CNN encoder and normalization stats."""
    cnn = ResNetStateEncoder(input_channels=48, output_dim=64).to(device)
    cnn.load_state_dict(torch.load(cnn_path, map_location=device))
    cnn.eval()

    norm_stats = torch.load(norm_path, map_location=device)
    state_mean = norm_stats['mean'].to(device)
    state_std = norm_stats['std'].to(device)

    return cnn, state_mean, state_std


def frames_to_tensor(frame_buffer, device):
    """Convert frame buffer to CNN input tensor."""
    frames = np.array(frame_buffer, dtype=np.uint8)  # (16, 224, 224, 3)
    t = torch.from_numpy(frames).float() / 255.0
    t = t.permute(0, 3, 1, 2)  # (16, 3, 224, 224)
    t = t.reshape(48, 224, 224)  # (48, 224, 224)
    t = t.unsqueeze(0).to(device)  # (1, 48, 224, 224)
    return t


def run_state_episode(env, agent, obs_buffer_len, max_steps=500, video_writer=None):
    """Run episode using ground-truth state."""
    obs, info = env.reset()

    # Initialize observation buffer
    obs_buffer = deque(maxlen=obs_buffer_len)
    for _ in range(obs_buffer_len):
        obs_buffer.append(obs)
    state = np.stack(obs_buffer, axis=0).flatten()

    # Record initial frame if saving video
    if video_writer is not None:
        frame = env.render()
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    total_reward = 0
    steps = 0
    done = False

    while not done and steps < max_steps:
        action = agent.get_action(state, training=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        # Record frame if saving video
        if video_writer is not None:
            frame = env.render()
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if not done:
            obs_buffer.append(next_obs)
            state = np.stack(obs_buffer, axis=0).flatten()

    return total_reward, steps


def run_vision_episode(env, agent, cnn, state_mean, state_std, device, max_steps=500, video_writer=None):
    """Run episode using CNN-predicted state from frames."""
    obs, info = env.reset()

    # Initialize frame buffer (vision mode always uses 16 frames)
    frame_buffer = deque(maxlen=16)
    full_frame = env.render()
    frame = cv2.resize(full_frame, (224, 224), interpolation=cv2.INTER_AREA)
    for _ in range(16):
        frame_buffer.append(frame)

    # Record initial frame if saving video
    if video_writer is not None:
        video_writer.write(cv2.cvtColor(full_frame, cv2.COLOR_RGB2BGR))

    # Get initial state from CNN
    with torch.no_grad():
        frame_tensor = frames_to_tensor(frame_buffer, device)
        predicted_state = cnn(frame_tensor)
        state = (predicted_state * state_std + state_mean).squeeze(0).cpu().numpy()

    total_reward = 0
    steps = 0
    done = False

    while not done and steps < max_steps:
        action = agent.get_action(state, training=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        if not done:
            full_frame = env.render()
            frame = cv2.resize(full_frame, (224, 224), interpolation=cv2.INTER_AREA)
            frame_buffer.append(frame)

            # Record frame if saving video
            if video_writer is not None:
                video_writer.write(cv2.cvtColor(full_frame, cv2.COLOR_RGB2BGR))

            with torch.no_grad():
                frame_tensor = frames_to_tensor(frame_buffer, device)
                predicted_state = cnn(frame_tensor)
                state = (predicted_state * state_std + state_mean).squeeze(0).cpu().numpy()

    return total_reward, steps


def main():
    parser = argparse.ArgumentParser(
        description='Run SAC agent on CartPole',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_agent.py --mode state              # Ground truth state
  python run_agent.py --mode vision             # CNN predictions
  python run_agent.py --mode vision --agent noise2  # Noise-robust agent
  python run_agent.py --list-agents             # Show available agents
        """
    )
    parser.add_argument('--mode', type=str, choices=['state', 'vision'], default='state',
                        help='Control mode: state (ground truth) or vision (CNN)')
    parser.add_argument('--agent', type=str, default='073',
                        help='Agent name (073, noise2, 047, 005) or path to .pth file')
    parser.add_argument('--obs_buffer_len', type=int, default=None,
                        help='Observation buffer length (auto-detected for known agents)')
    parser.add_argument('--cnn_path', type=str, default='trained_cnn/best_model.pth',
                        help='Path to CNN encoder (vision mode only)')
    parser.add_argument('--norm_path', type=str, default='trained_cnn/norm_stats.pt',
                        help='Path to normalization stats (vision mode only)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Max steps per episode')
    parser.add_argument('--render', action='store_true',
                        help='Render to screen (slower)')
    parser.add_argument('--save_video', type=str, default=None,
                        help='Save video to this path (e.g., output.mp4)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video framerate (default: 30)')
    parser.add_argument('--list-agents', action='store_true',
                        help='List available pre-configured agents')
    args = parser.parse_args()

    # List agents and exit
    if args.list_agents:
        print("Available agents:")
        for name, info in AGENTS.items():
            print(f"  {name:8s} - {info['description']} (obs_buffer_len={info['obs_buffer_len']})")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")

    # Resolve agent path and obs_buffer_len
    if args.agent in AGENTS:
        agent_info = AGENTS[args.agent]
        agent_path = agent_info['path']
        obs_buffer_len = args.obs_buffer_len or agent_info['obs_buffer_len']
        print(f"Agent: {args.agent} - {agent_info['description']}")
    else:
        agent_path = args.agent
        obs_buffer_len = args.obs_buffer_len or 16
        print(f"Agent: {agent_path} (custom)")

    # Vision mode requires 16-frame buffer agents
    if args.mode == 'vision' and obs_buffer_len != 16:
        print(f"Warning: Vision mode requires obs_buffer_len=16, but agent uses {obs_buffer_len}")
        print("         CNN was trained on 16-frame stacks. Results may be poor.")

    state_size = obs_buffer_len * 4

    # Load agent
    agent = load_sac_agent(agent_path, state_size, device)
    print(f"Loaded agent from {agent_path}")

    # Load CNN if vision mode
    cnn, state_mean, state_std = None, None, None
    if args.mode == 'vision':
        cnn, state_mean, state_std = load_cnn_encoder(args.cnn_path, args.norm_path, device)
        print(f"Loaded CNN from {args.cnn_path}")

    # Create environment
    render_mode = "human" if args.render else "rgb_array"
    env = gym.make("CartPole-v1", render_mode=render_mode)

    # Setup video writer if saving
    video_writer = None
    if args.save_video:
        # Get frame size from first render (need to reset first)
        env.reset()
        test_frame = env.render()
        height, width = test_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.save_video, fourcc, args.fps, (width, height))
        print(f"Recording video to {args.save_video}")

    # Run episodes
    print(f"\nRunning {args.episodes} episodes...")
    rewards = []
    steps_list = []

    for ep in range(args.episodes):
        if args.mode == 'state':
            reward, steps = run_state_episode(env, agent, obs_buffer_len, args.max_steps, video_writer)
        else:
            reward, steps = run_vision_episode(
                env, agent, cnn, state_mean, state_std, device, args.max_steps, video_writer
            )
        rewards.append(reward)
        steps_list.append(steps)
        print(f"  Episode {ep+1}: steps={steps}")

    # Cleanup
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {args.save_video}")
    env.close()

    # Summary
    print(f"\nResults ({args.mode} mode, {args.episodes} episodes):")
    print(f"  Mean steps: {np.mean(steps_list):.1f} +/- {np.std(steps_list):.1f}")
    print(f"  Best: {np.max(steps_list)}, Worst: {np.min(steps_list)}")


if __name__ == "__main__":
    main()
