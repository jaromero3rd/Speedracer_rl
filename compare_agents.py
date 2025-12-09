#!/usr/bin/env python3
"""
Compare 073 vs noise2 agents using CNN predictions.
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


def load_cnn(cnn_path, norm_path, device):
    cnn = ResNetStateEncoder(input_channels=48, output_dim=64).to(device)
    cnn.load_state_dict(torch.load(cnn_path, map_location=device))
    cnn.eval()

    norm = torch.load(norm_path, map_location=device)
    return cnn, norm['mean'].to(device), norm['std'].to(device)


def load_agent(path, device):
    agent = SAC(state_size=64, action_size=2, device=device)
    agent.actor_local.load_state_dict(torch.load(path, map_location=device))
    agent.actor_local.eval()
    return agent


def frames_to_tensor(frame_buffer, device):
    frames = np.array(frame_buffer, dtype=np.uint8)
    t = torch.from_numpy(frames).float() / 255.0
    t = t.permute(0, 3, 1, 2).reshape(48, 224, 224)
    return t.unsqueeze(0).to(device)


def run_episode(env, agent, cnn, mean, std, device, max_steps=500):
    obs, _ = env.reset()

    frame_buffer = deque(maxlen=16)
    frame = cv2.resize(env.render(), (224, 224), interpolation=cv2.INTER_AREA)
    for _ in range(16):
        frame_buffer.append(frame)

    with torch.no_grad():
        pred = cnn(frames_to_tensor(frame_buffer, device))
        state = (pred * std + mean).squeeze(0).cpu().numpy()

    steps = 0
    done = False

    while not done and steps < max_steps:
        action = agent.get_action(state, training=False)
        _, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        steps += 1

        if not done:
            frame = cv2.resize(env.render(), (224, 224), interpolation=cv2.INTER_AREA)
            frame_buffer.append(frame)
            with torch.no_grad():
                pred = cnn(frames_to_tensor(frame_buffer, device))
                state = (pred * std + mean).squeeze(0).cpu().numpy()

    return steps


def evaluate(env, agent, cnn, mean, std, device, episodes, name):
    steps = []
    for _ in tqdm(range(episodes), desc=name):
        steps.append(run_episode(env, agent, cnn, mean, std, device))
    return {
        'name': name,
        'mean': np.mean(steps),
        'std': np.std(steps),
        'max': np.max(steps),
        'min': np.min(steps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--cnn_path', default='trained_cnn/best_model.pth')
    parser.add_argument('--norm_path', default='trained_cnn/norm_stats.pt')
    parser.add_argument('--agent_073', default='trained_models/grid_search_073SAC_discrete0.pth')
    parser.add_argument('--agent_noise', default='trained_models/noise2SAC_discrete0.pth')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cnn, mean, std = load_cnn(args.cnn_path, args.norm_path, device)
    agent_073 = load_agent(args.agent_073, device)
    agent_noise = load_agent(args.agent_noise, device)

    env = gym.make("CartPole-v1", render_mode="rgb_array")

    r073 = evaluate(env, agent_073, cnn, mean, std, device, args.episodes, "073")
    rnoise = evaluate(env, agent_noise, cnn, mean, std, device, args.episodes, "noise2")

    env.close()

    print(f"\nResults ({args.episodes} episodes each):")
    print(f"  073:    {r073['mean']:.1f} +/- {r073['std']:.1f} (max {r073['max']}, min {r073['min']})")
    print(f"  noise2: {rnoise['mean']:.1f} +/- {rnoise['std']:.1f} (max {rnoise['max']}, min {rnoise['min']})")

    if rnoise['mean'] > r073['mean']:
        print(f"\nNoise-trained wins by {rnoise['mean'] - r073['mean']:.1f} steps")
    else:
        print(f"\n073 wins by {r073['mean'] - rnoise['mean']:.1f} steps")


if __name__ == "__main__":
    main()
