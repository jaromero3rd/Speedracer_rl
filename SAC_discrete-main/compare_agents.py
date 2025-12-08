#!/usr/bin/env python3
"""
Compare two SAC agents (noise2 vs 073) using the same CNN encoder.
Both agents receive predicted states from the CNN, not ground truth.
"""

import numpy as np
import torch
import gymnasium as gym
import cv2
import argparse
from collections import deque
from tqdm import tqdm

from agent import SAC
from vision_network import ResNetStateEncoder


def load_cnn_and_norm(cnn_path, norm_path, device):
    """Load CNN encoder and normalization stats."""
    cnn = ResNetStateEncoder(input_channels=48, output_dim=64).to(device)
    cnn.load_state_dict(torch.load(cnn_path, map_location=device))
    cnn.eval()

    norm_stats = torch.load(norm_path, map_location=device)
    state_mean = norm_stats['mean'].to(device)
    state_std = norm_stats['std'].to(device)

    return cnn, state_mean, state_std


def load_agent(agent_path, device):
    """Load SAC agent."""
    agent = SAC(
        state_size=64,  # 16 frames * 4 state dims
        action_size=2,  # CartPole has 2 actions
        device=device
    )
    agent.actor_local.load_state_dict(torch.load(agent_path, map_location=device))
    agent.actor_local.eval()
    return agent


def frames_to_tensor(frame_buffer, device):
    """Convert frame buffer to CNN input tensor."""
    frames = np.array(frame_buffer, dtype=np.uint8)
    t = torch.from_numpy(frames).float() / 255.0
    t = t.permute(0, 3, 1, 2)
    t = t.reshape(48, 224, 224)
    t = t.unsqueeze(0).to(device)
    return t


def run_episode(env, agent, cnn, state_mean, state_std, device, max_steps=500):
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
            frame = env.render()
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            frame_buffer.append(frame)

            with torch.no_grad():
                frame_tensor = frames_to_tensor(frame_buffer, device)
                predicted_state = cnn(frame_tensor)
                state = (predicted_state * state_std + state_mean).squeeze(0).cpu().numpy()

    return total_reward, steps


def evaluate_agent(env, agent, cnn, state_mean, state_std, device, episodes, agent_name):
    """Evaluate an agent over multiple episodes."""
    rewards = []
    steps_list = []

    for ep in tqdm(range(episodes), desc=f"Evaluating {agent_name}"):
        reward, steps = run_episode(env, agent, cnn, state_mean, state_std, device)
        rewards.append(reward)
        steps_list.append(steps)

    return {
        'name': agent_name,
        'rewards': rewards,
        'steps': steps_list,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_steps': np.mean(steps_list),
        'std_steps': np.std(steps_list),
        'max_reward': np.max(rewards),
        'min_reward': np.min(rewards),
    }


def main():
    parser = argparse.ArgumentParser(description='Compare noise2 vs 073 agents with CNN')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Number of episodes per agent')
    parser.add_argument('--cnn_path', type=str, default='trained_cnn/best_model.pth')
    parser.add_argument('--norm_path', type=str, default='trained_cnn/norm_stats.pt')
    parser.add_argument('--agent_073', type=str,
                        default='trained_models/grid_search_073SAC_discrete0.pth')
    parser.add_argument('--agent_noise', type=str,
                        default='trained_models/noise2SAC_discrete0.pth')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load shared CNN
    print("\nLoading CNN encoder...")
    cnn, state_mean, state_std = load_cnn_and_norm(args.cnn_path, args.norm_path, device)

    # Load agents
    print("Loading agents...")
    agent_073 = load_agent(args.agent_073, device)
    agent_noise = load_agent(args.agent_noise, device)

    # Create environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Evaluate both agents
    print(f"\nEvaluating {args.episodes} episodes per agent...")

    results_073 = evaluate_agent(
        env, agent_073, cnn, state_mean, state_std, device,
        args.episodes, "073 (standard)"
    )

    results_noise = evaluate_agent(
        env, agent_noise, cnn, state_mean, state_std, device,
        args.episodes, "noise2 (noise-trained)"
    )

    env.close()

    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS (Vision-based control via CNN)")
    print("="*60)

    print(f"\n{'Metric':<20} {'073 (standard)':<20} {'noise2 (noise-trained)':<20}")
    print("-"*60)
    print(f"{'Mean Reward':<20} {results_073['mean_reward']:<20.1f} {results_noise['mean_reward']:<20.1f}")
    print(f"{'Std Reward':<20} {results_073['std_reward']:<20.1f} {results_noise['std_reward']:<20.1f}")
    print(f"{'Mean Steps':<20} {results_073['mean_steps']:<20.1f} {results_noise['mean_steps']:<20.1f}")
    print(f"{'Std Steps':<20} {results_073['std_steps']:<20.1f} {results_noise['std_steps']:<20.1f}")
    print(f"{'Max Reward':<20} {results_073['max_reward']:<20.0f} {results_noise['max_reward']:<20.0f}")
    print(f"{'Min Reward':<20} {results_073['min_reward']:<20.0f} {results_noise['min_reward']:<20.0f}")
    print("="*60)

    # Determine winner
    if results_noise['mean_reward'] > results_073['mean_reward']:
        diff = results_noise['mean_reward'] - results_073['mean_reward']
        print(f"\nNoise-trained agent performs better by {diff:.1f} mean reward")
    else:
        diff = results_073['mean_reward'] - results_noise['mean_reward']
        print(f"\nStandard agent performs better by {diff:.1f} mean reward")


if __name__ == "__main__":
    main()
