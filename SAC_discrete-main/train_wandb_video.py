"""
Alternative training script with wandb's built-in gym video recording.
This version uses wandb.gym.monitor() for more reliable video logging.

Usage:
    python train_wandb_video.py --env CartPole-v1 --episodes 100
"""

import gymnasium as gym
import numpy as np
from collections import deque
import torch
import argparse
import os
import random
from buffer import ReplayBuffer
from utils import save, collect_random
from agent import SAC
import wandb


def get_config():
    parser = argparse.ArgumentParser(description="RL with Wandb Video")
    parser.add_argument(
        "--run_name", type=str, default="SAC", help="Run name, default: SAC"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v0",
        help="Gym environment name, default: CartPole-v1",
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes, default: 100"
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=100_000,
        help="Maximal training dataset size, default: 100_000",
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument(
        "--video_freq",
        type=int,
        default=10,
        help="Record video every N episodes, default: 10",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="cartpole_vision",
        help="Wandb project name, default: cartpole_vision",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Wandb entity (username or team), default: None",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="Saves the network every x epochs, default: 25",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size, default: 256"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-4, help="Learning rate, default: 5e-4"
    )
    parser.add_argument(
        "--entropy_bonus",
        type=str,
        default="None",
        help="Fixed entropy bonus (alpha). 'None' = learnable, default: None",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Epsilon for epsilon-greedy exploration, default: 0.0",
    )
    parser.add_argument(
        "--obs_buffer_max_len",
        type=int,
        default=4,
        help="Observation buffer length, default: 4",
    )

    args = parser.parse_args()
    return args


def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Create environment with render_mode for video
    env = gym.make(config.env, render_mode="rgb_array")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    steps = 0
    average10 = deque(maxlen=10)

    obs_buffer_max_len = getattr(config, "obs_buffer_max_len", 4)
    obs_buffer = deque(maxlen=obs_buffer_max_len)
    total_steps = 0

    # Initialize Weights & Biases with gym monitoring
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.run_name,
        config={
            "env": config.env,
            "episodes": config.episodes,
            "buffer_size": config.buffer_size,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "learning_rate": getattr(config, "learning_rate", 5e-4),
            "entropy_bonus": getattr(config, "entropy_bonus", "None"),
            "epsilon": getattr(config, "epsilon", 0.0),
            "obs_buffer_max_len": obs_buffer_max_len,
            "video_freq": config.video_freq,
        },
        monitor_gym=True,  # Enable gym monitoring
    )

    # Wrap environment with wandb video monitoring
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=f"videos/{wandb.run.id}",
        episode_trigger=lambda episode_id: episode_id % config.video_freq == 0,
        name_prefix="rl-video",
    )

    state_size_flat = env.observation_space.shape[0] * obs_buffer_max_len

    # Get hyperparameters from config (with defaults)
    learning_rate = getattr(config, "learning_rate", 5e-4)
    entropy_bonus_str = getattr(config, "entropy_bonus", "None")
    entropy_bonus = (
        None
        if entropy_bonus_str == "None" or entropy_bonus_str is None
        else float(entropy_bonus_str)
    )
    epsilon = getattr(config, "epsilon", 0.0)

    agent = SAC(
        state_size=state_size_flat,
        action_size=env.action_space.n,
        device=device,
        learning_rate=learning_rate,
        entropy_bonus=entropy_bonus,
        epsilon=epsilon,
    )

    buffer = ReplayBuffer(
        buffer_size=config.buffer_size, batch_size=config.batch_size, device=device
    )

    collect_random(
        env=env,
        dataset=buffer,
        num_samples=10000,
        obs_buffer_max_len=obs_buffer_max_len,
    )

    for i in range(1, config.episodes + 1):
        obs, info = env.reset()

        # Reinitialize observation buffer for new episode
        obs_buffer.clear()
        for _ in range(obs_buffer_max_len):
            obs_buffer.append(obs)
        state = np.stack(obs_buffer, axis=0).flatten(order="C")

        episode_steps = 0
        rewards = 0
        while True:
            action = agent.get_action(state)
            steps += 1
            next_obs, reward, terminated, truncated, info = env.step(action)

            obs_buffer.append(next_obs)
            next_state = np.stack(obs_buffer, axis=0).flatten(order="C")

            done = terminated or truncated
            
            # Custom reward shaping for CartPole
            mean = 1
            std = 1
            reward = reward + np.exp(-((state[0] - mean) ** 2) / (2 * std**2))

            buffer.add(state, action, reward, next_state, done)
            policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = (
                agent.learn(steps, buffer.sample(), gamma=0.99)
            )
            obs = next_obs
            state = next_state
            rewards += reward
            episode_steps += 1
            if done:
                break

        average10.append(rewards)
        total_steps += episode_steps
        print(
            f"Episode: {i} | Reward: {rewards:.2f} | Avg(10): {sum(average10)/len(average10):.2f} | "
            f"Policy Loss: {policy_loss:.4f} | Steps: {steps}"
        )

        # Log metrics to Weights & Biases
        log_dict = {
            "episode": i,
            "reward": rewards,
            "avg_reward_10": sum(average10) / len(average10),
            "total_steps": total_steps,
            "policy_loss": policy_loss,
            "alpha_loss": alpha_loss,
            "bellmann_error1": bellmann_error1,
            "bellmann_error2": bellmann_error2,
            "current_alpha": current_alpha,
            "episode_steps": episode_steps,
            "buffer_size": len(buffer),
        }
        
        # Wandb will automatically log videos from RecordVideo wrapper
        wandb.log(log_dict)

        if i % config.save_every == 0:
            save(config, save_name="SAC_discrete", model=agent.actor_local, ep=0)

    # Close video recorder if it exists
    if hasattr(env, "close_video_recorder"):
        env.close_video_recorder()

    # Finish wandb run
    wandb.finish()
    env.close()


if __name__ == "__main__":
    config = get_config()
    train(config)

