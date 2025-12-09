#!/usr/bin/env python3
"""
Record videos of the CNN-based agent playing CartPole.
CNN sees 224x224, video is recorded at full resolution (400x600).
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


def load_models(agent_path, cnn_path, norm_path, device):
    """Load SAC agent and CNN encoder."""
    agent = SAC(state_size=64, action_size=2, device=device)
    agent.actor_local.load_state_dict(torch.load(agent_path, map_location=device))
    agent.actor_local.eval()

    cnn = ResNetStateEncoder(input_channels=48, output_dim=64).to(device)
    cnn.load_state_dict(torch.load(cnn_path, map_location=device))
    cnn.eval()

    norm_stats = torch.load(norm_path, map_location=device)
    state_mean = norm_stats['mean'].to(device)
    state_std = norm_stats['std'].to(device)

    return agent, cnn, state_mean, state_std


def frames_to_tensor(frame_buffer, device):
    """Convert frame buffer to CNN input tensor."""
    frames = np.array(frame_buffer, dtype=np.uint8)
    t = torch.from_numpy(frames).float() / 255.0
    t = t.permute(0, 3, 1, 2)
    t = t.reshape(48, 224, 224)
    t = t.unsqueeze(0).to(device)
    return t


def record_episode(env, agent, cnn, state_mean, state_std, device, output_path, fps=30, max_steps=500):
    """Record a single episode to video."""
    obs, info = env.reset()

    # Get full resolution frame for video
    full_frame = env.render()
    height, width = full_frame.shape[:2]

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize frame buffer for CNN (224x224)
    frame_buffer = deque(maxlen=16)
    small_frame = cv2.resize(full_frame, (224, 224), interpolation=cv2.INTER_AREA)
    for _ in range(16):
        frame_buffer.append(small_frame)

    # Write initial frame
    out.write(cv2.cvtColor(full_frame, cv2.COLOR_RGB2BGR))

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

        # Get frame at full resolution for video
        full_frame = env.render()
        out.write(cv2.cvtColor(full_frame, cv2.COLOR_RGB2BGR))

        if not done:
            # Resize for CNN
            small_frame = cv2.resize(full_frame, (224, 224), interpolation=cv2.INTER_AREA)
            frame_buffer.append(small_frame)

            with torch.no_grad():
                frame_tensor = frames_to_tensor(frame_buffer, device)
                predicted_state = cnn(frame_tensor)
                state = (predicted_state * state_std + state_mean).squeeze(0).cpu().numpy()

    out.release()
    return total_reward, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_path', type=str,
                        default='trained_models/grid_search_073SAC_discrete0.pth')
    parser.add_argument('--cnn_path', type=str,
                        default='trained_cnn/best_model.pth')
    parser.add_argument('--norm_path', type=str,
                        default='trained_cnn/norm_stats.pt')
    parser.add_argument('--output_dir', type=str, default='videos')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--max_steps', type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    agent, cnn, state_mean, state_std = load_models(
        args.agent_path, args.cnn_path, args.norm_path, device
    )
    print("Models loaded")

    env = gym.make("CartPole-v1", render_mode="rgb_array")

    print(f"\nRecording {args.episodes} episodes...")
    for ep in range(args.episodes):
        output_path = os.path.join(args.output_dir, f'vision_agent_ep{ep+1}.mp4')
        reward, steps = record_episode(
            env, agent, cnn, state_mean, state_std, device,
            output_path, fps=args.fps, max_steps=args.max_steps
        )
        print(f"  Episode {ep+1}: reward={reward:.0f}, steps={steps}, saved to {output_path}")

    env.close()
    print(f"\nDone! Videos saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
