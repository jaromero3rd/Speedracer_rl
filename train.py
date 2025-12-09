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
from utils import save, collect_random, collect_random_RND, collect_policy
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
        default="CartPole-v1",
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
        default=25,
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
        type=float,
        default=0.2,
        help="Fixed entropy bonus (alpha) default: 0.2",
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
        default=16,
        help="Observation buffer length, default: 16",
    )
    parser.add_argument(
        "--buffer_init_method",
        type=str,
        default="random",
        choices=["random", "RND", "policy"],
        help="Method to initialize replay buffer: 'random' for random actor, 'RND' for Random Network Distillation, 'policy' for initial actor policy, default: random",
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
    buffer_init_method = getattr(config, "buffer_init_method", "random")
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
            "buffer_init_method": buffer_init_method,
        },
    )
    state_size_flat = env.observation_space.shape[0] * obs_buffer_max_len

    # Get hyperparameters from config (with defaults)
    learning_rate = getattr(config, "learning_rate", 5e-4)
    entropy_bonus = getattr(config, "entropy_bonus", "None")
    # Convert string "None" to actual None

    epsilon = getattr(config, "epsilon", 0.0)

    buffer = ReplayBuffer(
        buffer_size=config.buffer_size, batch_size=config.batch_size, device=device
    )

    # Initialize replay buffer using specified method
    buffer_init_method = getattr(config, "buffer_init_method", "random")
    num_samples = config.buffer_size
    use_rnd = (buffer_init_method == "RND")
    use_policy = (buffer_init_method == "policy")
    
    print(f"Initializing replay buffer with {num_samples} samples using method: {buffer_init_method}")
    
    # If using policy method, we need to create agent first (with epsilon=0)
    if use_policy:
        # Create agent with epsilon=0 for deterministic policy
        agent = SAC(
            state_size=state_size_flat,
            action_size=env.action_space.n,
            device=device,
            learning_rate=learning_rate,
            entropy_bonus=entropy_bonus,
            epsilon=0.0,  # Set epsilon to 0 for deterministic policy
            use_rnd=False,  # No RND during buffer initialization with policy
        )
        collect_policy(
            env=env,
            dataset=buffer,
            agent=agent,
            num_samples=num_samples,
            obs_buffer_max_len=obs_buffer_max_len,
        )
    elif use_rnd:
        collect_random_RND(
            env=env,
            dataset=buffer,
            num_samples=num_samples,
            obs_buffer_max_len=obs_buffer_max_len,
            device=device,
        )
    else:  # random
        collect_random(
            env=env,
            dataset=buffer,
            num_samples=num_samples,
            obs_buffer_max_len=obs_buffer_max_len,
        )
    
    print(f"Buffer initialized with {len(buffer)} samples")
    
    # Initialize agent (or reuse if already created for policy method)
    if not use_policy:
        agent = SAC(
            state_size=state_size_flat,
            action_size=env.action_space.n,
            device=device,
            learning_rate=learning_rate,
            entropy_bonus=entropy_bonus,
            epsilon=epsilon,
            use_rnd=use_rnd,
        )
    else:
        # Update agent's epsilon to the configured value for training
        agent.epsilon = epsilon

    # Setup video recording for wandb
    video_dir = "./logs"
    if config.log_video:
        try:
            os.makedirs(video_dir, exist_ok=True)
            env = RecordVideo(env,video_dir,episode_trigger=lambda episode_id: (episode_id % 10 == 0),   name_prefix="rl-video",)
            print(f"Video recording enabled. Videos will be saved to: {video_dir}")
            print("Videos will be recorded for episodes: 10, 20, 30, ...")
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
        rnd_loss = 0.0  
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

            # Reward shaping functions
            # 1. Gaussian reward shaping
            # mean = 1
            # std = 1
            # reward = reward + np.exp(-((state[0] - mean) ** 2) / (2 * std**2))
            # 2. linear reward shaping
            # reward = reward + np.maximum(0.0, 1.0 - np.abs(state[0] - 1.0))
            # 3. inverse reward:
            reward = reward + 1 / (np.abs(state[0]-1)+1)

            
            # Add RND intrinsic reward if RND is enabled
            if use_rnd:
                rnd_intrinsic_reward = agent.compute_rnd_intrinsic_reward(next_state)
                reward = reward + rnd_intrinsic_reward

            buffer.add(state, action, reward, next_state, done)
            policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha, rnd_loss = (
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
        if use_rnd:
            log_dict["rnd_loss"] = rnd_loss
        wandb.log(log_dict, commit=True)

        if (i % 10 == 0) and config.log_video:
            time.sleep(2.0)
            video_files = glob.glob(
                os.path.join(video_dir, "**", "*.mp4"), recursive=True
            )

            if len(video_files) > 0:
                video_path = max(video_files, key=os.path.getmtime)
                print(f"Found video file: {video_path}")
                try:
                    file_size = os.path.getsize(video_path)
                    if file_size > 0:
                        # Log video to wandb
                        try:
                            wandb.log(
                                {"video": wandb.Video(video_path, format="mp4")},
                                commit=True,
                            )
                            print(f"✓ Video logged to wandb for episode {i}")
                        except Exception as e:
                            print(f"Warning: Failed to log video to wandb: {e}")
                            print(f"  Video path: {video_path}")
                            print(f"  File size: {file_size} bytes")
                    else:
                        print(f"Warning: Video file is empty: {video_path}")
                except Exception as e:
                    print(f"Warning: Error accessing video file: {e}")
            else:
                print(f"Warning: No video files found in {video_dir} for episode {i}")
                # Debug: show what's in the directory
                if os.path.exists(video_dir):
                    all_files = glob.glob(
                        os.path.join(video_dir, "**", "*"), recursive=True
                    )
                    print(f"  Files in video_dir: {all_files[:5]}")

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
