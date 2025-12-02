import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo

import numpy as np
from collections import deque
import torch
import argparse
import os
import glob
import time
from buffer import ReplayBuffer
from utils import save, collect_random
import random
from agent import SAC

import wandb


def get_config():
    parser = argparse.ArgumentParser(description="RL")
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
        "--log_video",
        type=int,
        default=0,
        help="Log agent behaviour to wandb when set to 1, default: 0",
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

    if config.log_video:
        env = gym.make(config.env, render_mode="rgb_array")
    else:
        env = gym.make(config.env)

    # Note: env.seed() and env.action_space.seed() are deprecated in new gymnasium
    # Seed is now handled via gym.make() or through the environment's reset() method

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    steps = 0
    average10 = deque(maxlen=10)

    obs_buffer_max_len = getattr(config, "obs_buffer_max_len", 4)
    obs_buffer = deque(maxlen=obs_buffer_max_len)
    total_steps = 0

    # Initialize Weights & Biases
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
        },
    )
    state_size_flat = env.observation_space.shape[0] * obs_buffer_max_len

    # Get hyperparameters from config (with defaults)
    learning_rate = getattr(config, "learning_rate", 5e-4)
    entropy_bonus_str = getattr(config, "entropy_bonus", "None")
    # Convert string "None" to actual None
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

    # Setup video recording for wandb
    video_dir = "./logs"
    if config.log_video:
        try:
            os.makedirs(video_dir, exist_ok=True)
            # Use RecordVideo wrapper
            # Record every 10th episode
            env = RecordVideo(
                env,
                video_dir,
                episode_trigger=lambda x: (x - 1) % 10 == 0,  # Episodes 1, 11, 21, etc.
                name_prefix="episode",
            )
            print(f"Video recording enabled. Videos will be saved to: {video_dir}")
            print("Videos will be recorded for episodes: 1, 11, 21, 31, ...")
        except Exception as e:
            print(f"Warning: Failed to enable video recording: {e}")
            print("Install moviepy with: pip install moviepy")
            print("Continuing without video recording...")
            config.log_video = 0

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
            ## WE GET TO ADD IN THIS LINE THE REWARDS FUNCTIONS DEPENDENT ON STATES AND EVEN ADD IN OUR STATE -> z LATENT SPACE EMBEDDING FUNCTION HERE
            # Found that the reward function was best when it followed a normal distribution
            # It helps with giving signal far away without highjacking the reward function (found + was better than - for some reason)
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
            "Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(
                i,
                rewards,
                policy_loss,
                steps,
            )
        )

        # Log metrics to Weights & Biases
        wandb.log(
            {
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
        )

        if (i % 10 == 0) and config.log_video:
            # Wait for video file to be written (check file size stability)
            video_written = False
            video_path = None
            for attempt in range(20):  # Wait up to 2 seconds
                # Look for the video file that should have been created for this episode
                pattern = os.path.join(video_dir, f"**/episode-episode-{i}.mp4")
                video_files = glob.glob(pattern, recursive=True)
                if len(video_files) > 0:
                    # Check if file size is stable (not being written)
                    try:
                        video_path = video_files[0]
                        size1 = os.path.getsize(video_path)
                        time.sleep(0.1)
                        size2 = os.path.getsize(video_path)
                        if size1 == size2 and size1 > 0:
                            video_written = True
                            break
                    except OSError:
                        pass
                time.sleep(0.1)

            if video_written and video_path:
                # Additional wait to ensure video is fully flushed
                time.sleep(1.0)
                # Log video to wandb
                try:
                    wandb.log(
                        {
                            "video": wandb.Video(video_path, fps=4, format="mp4"),
                            "episode": i,
                        }
                    )
                    print(f"Video logged to wandb for episode {i}")
                except Exception as e:
                    print(f"Warning: Failed to log video to wandb: {e}")

        if i % config.save_every == 0:
            save(config, save_name="SAC_discrete", model=agent.actor_local, ep=0)

    # Close video recorder if it exists
    if config.log_video and hasattr(env, "close_video_recorder"):
        env.close_video_recorder()

    # Finish wandb run
    wandb.finish()
    env.close()


if __name__ == "__main__":
    config = get_config()
    train(config)
