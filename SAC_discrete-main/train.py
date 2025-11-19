import gymnasium as gym
# Optional import for PyBullet environments (only needed if using PyBullet envs)
try:
    import pybullet_envs
except (ImportError, ModuleNotFoundError):
    # pybullet_envs not available or requires old gym package
    # This is fine if not using PyBullet environments
    pass
import numpy as np
from collections import deque
import torch
import argparse
import os
from buffer import ReplayBuffer
from utils import save, collect_random
import random
from agent import SAC
from logging_utils import (
    create_writer,
    log_hyperparameters,
    log_episode_metrics,
    log_video
)

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="SAC", help="Run name, default: SAC")
    parser.add_argument("--env", type=str, default="CartPole-v0", help="Gym environment name, default: CartPole-v0")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes, default: 100")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to tensorboard when set to 1, default: 0")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for TensorBoard logs, default: ./logs")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    
    args = parser.parse_args()
    return args

def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Create environment with render_mode for video recording
    if config.log_video:
        env = gym.make(config.env, render_mode='rgb_array')
    else:
        env = gym.make(config.env)
    
    # Note: env.seed() and env.action_space.seed() are deprecated in new gymnasium
    # Seed is now handled via gym.make() or through the environment's reset() method

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0
    
    # Create TensorBoard writer
    writer, log_path = create_writer(config.run_name, config.log_dir)
    
    # Log hyperparameters
    log_hyperparameters(writer, config)
    
    agent = SAC(state_size=env.observation_space.shape[0],
                     action_size=env.action_space.n,
                     device=device)

    buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)
    
    collect_random(env=env, dataset=buffer, num_samples=10000)
    
    # Save videos to a subdirectory within the TensorBoard log directory
    video_dir = os.path.join(log_path, 'videos')
    if config.log_video:
        try:
            # Use RecordVideo wrapper instead of deprecated Monitor
            # Note: episode_trigger uses 0-based indexing, so episode 0, 10, 20, etc. will be recorded
            env = gym.wrappers.RecordVideo(
                env, 
                video_dir, 
                episode_trigger=lambda x: (x - 1) % 10 == 0,  # Adjust for 1-based episode indexing
                name_prefix=f"episode"
            )
            print(f"Video recording enabled. Videos will be saved to: {video_dir}")
            print(f"Videos will be recorded for episodes: 1, 11, 21, 31, ...")
        except Exception as e:
            print(f"Warning: Failed to enable video recording: {e}")
            print("Install moviepy with: pip install moviepy")
            print("Continuing without video recording...")
            config.log_video = 0

    for i in range(1, config.episodes+1):
        state, info = env.reset()
        episode_steps = 0
        rewards = 0
        while True:
            # print("here")
            action = agent.get_action(state)
            steps += 1
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated  # Combine terminated and truncated
            buffer.add(state, action, reward, next_state, done)
            policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = agent.learn(steps, buffer.sample(), gamma=0.99)
            state = next_state
            rewards += reward
            episode_steps += 1
            if done:
                break

        average10.append(rewards)
        total_steps += episode_steps
        print("Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, rewards, policy_loss, steps,))
        
        # Log metrics to TensorBoard
        log_episode_metrics(
            writer=writer,
            episode=i,
            reward=rewards,
            avg_reward_10=np.mean(average10),
            total_steps=total_steps,
            policy_loss=policy_loss,
            alpha_loss=alpha_loss,
            bellmann_error1=bellmann_error1,
            bellmann_error2=bellmann_error2,
            current_alpha=current_alpha,
            steps=steps,
            buffer_size=buffer.__len__()
        )

        if (i % 10 == 0) and config.log_video:
            # Close the video recorder to ensure the video is saved
            if hasattr(env, 'close_video_recorder'):
                env.close_video_recorder()
            # Small delay to ensure video file is written
            import time
            time.sleep(1.0)
            log_video(writer, i, video_dir=video_dir)

        if i % config.save_every == 0:
            save(config, save_name="SAC_discrete", model=agent.actor_local, ep=0)
    
    # Close video recorder if it exists
    if config.log_video and hasattr(env, 'close_video_recorder'):
        env.close_video_recorder()
    
    writer.close()
    env.close()

if __name__ == "__main__":
    config = get_config()
    train(config)
