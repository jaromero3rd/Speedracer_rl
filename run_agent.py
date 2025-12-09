#!/usr/bin/env python3
"""
Run SAC agents on CartPole (state or vision mode).
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


AGENTS = {
    '073': ('trained_models/grid_search_073SAC_discrete0.pth', 16),
    'noise2': ('trained_models/noise2SAC_discrete0.pth', 16),
    '047': ('trained_models/grid_search_047SAC_discrete0.pth', 8),
    '005': ('trained_models/grid_search_005SAC_discrete0.pth', 2),
}


def load_agent(path, state_size, device):
    agent = SAC(state_size=state_size, action_size=2, device=device)
    agent.actor_local.load_state_dict(torch.load(path, map_location=device))
    agent.actor_local.eval()
    return agent


def load_cnn(cnn_path, norm_path, device):
    cnn = ResNetStateEncoder(input_channels=48, output_dim=64).to(device)
    cnn.load_state_dict(torch.load(cnn_path, map_location=device))
    cnn.eval()
    norm = torch.load(norm_path, map_location=device)
    return cnn, norm['mean'].to(device), norm['std'].to(device)


def frames_to_tensor(buf, device):
    frames = np.array(buf, dtype=np.uint8)
    t = torch.from_numpy(frames).float() / 255.0
    t = t.permute(0, 3, 1, 2).reshape(48, 224, 224)
    return t.unsqueeze(0).to(device)


def run_state_episode(env, agent, obs_buf_len, max_steps=500, writer=None):
    obs, _ = env.reset()
    obs_buf = deque(maxlen=obs_buf_len)
    for _ in range(obs_buf_len):
        obs_buf.append(obs)
    state = np.stack(obs_buf, axis=0).flatten()

    if writer:
        writer.write(cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR))

    steps = 0
    done = False
    while not done and steps < max_steps:
        action = agent.get_action(state, training=False)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
        steps += 1

        if writer:
            writer.write(cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR))
        if not done:
            obs_buf.append(obs)
            state = np.stack(obs_buf, axis=0).flatten()

    return steps


def run_vision_episode(env, agent, cnn, mean, std, device, max_steps=500, writer=None):
    obs, _ = env.reset()
    frame_buf = deque(maxlen=16)
    frame = cv2.resize(env.render(), (224, 224), interpolation=cv2.INTER_AREA)
    for _ in range(16):
        frame_buf.append(frame)

    if writer:
        writer.write(cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR))

    with torch.no_grad():
        pred = cnn(frames_to_tensor(frame_buf, device))
        state = (pred * std + mean).squeeze(0).cpu().numpy()

    steps = 0
    done = False
    while not done and steps < max_steps:
        action = agent.get_action(state, training=False)
        _, _, term, trunc, _ = env.step(action)
        done = term or trunc
        steps += 1

        if not done:
            full = env.render()
            if writer:
                writer.write(cv2.cvtColor(full, cv2.COLOR_RGB2BGR))
            frame = cv2.resize(full, (224, 224), interpolation=cv2.INTER_AREA)
            frame_buf.append(frame)
            with torch.no_grad():
                pred = cnn(frames_to_tensor(frame_buf, device))
                state = (pred * std + mean).squeeze(0).cpu().numpy()

    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['state', 'vision'], default='state')
    parser.add_argument('--agent', default='073')
    parser.add_argument('--obs_buffer_len', type=int, default=None)
    parser.add_argument('--cnn_path', default='trained_cnn/best_model.pth')
    parser.add_argument('--norm_path', default='trained_cnn/norm_stats.pt')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save_video', type=str, default=None)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--list-agents', action='store_true')
    args = parser.parse_args()

    if args.list_agents:
        print("Agents:")
        for name, (path, buf) in AGENTS.items():
            print(f"  {name}: obs_buffer={buf}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Mode: {args.mode}")

    # resolve agent
    if args.agent in AGENTS:
        agent_path, obs_buf_len = AGENTS[args.agent]
    else:
        agent_path = args.agent
        obs_buf_len = args.obs_buffer_len or 16

    if args.obs_buffer_len:
        obs_buf_len = args.obs_buffer_len

    if args.mode == 'vision' and obs_buf_len != 16:
        print(f"Warning: vision mode needs obs_buffer=16, got {obs_buf_len}")

    state_size = obs_buf_len * 4
    agent = load_agent(agent_path, state_size, device)
    print(f"Loaded: {agent_path}")

    cnn, mean, std = None, None, None
    if args.mode == 'vision':
        cnn, mean, std = load_cnn(args.cnn_path, args.norm_path, device)

    render_mode = "human" if args.render else "rgb_array"
    env = gym.make("CartPole-v1", render_mode=render_mode)

    # video writer
    writer = None
    if args.save_video:
        env.reset()
        frame = env.render()
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_video, fourcc, args.fps, (w, h))

    steps_list = []
    for ep in range(args.episodes):
        if args.mode == 'state':
            steps = run_state_episode(env, agent, obs_buf_len, args.max_steps, writer)
        else:
            steps = run_vision_episode(env, agent, cnn, mean, std, device, args.max_steps, writer)
        steps_list.append(steps)
        print(f"  Ep {ep+1}: {steps} steps")

    if writer:
        writer.release()
        print(f"Saved: {args.save_video}")

    env.close()

    print(f"\nMean: {np.mean(steps_list):.1f}, Best: {max(steps_list)}, Worst: {min(steps_list)}")


if __name__ == "__main__":
    main()
