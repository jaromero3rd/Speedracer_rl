"""
Training script for SAC with continuous action spaces.

Adapted from SAC_discrete-main/train.py for continuous control tasks.
Designed for use with gymnasium environments with Box action spaces.
"""

import gymnasium as gym
import numpy as np
from collections import deque
import torch
import argparse
import os
import sys
import glob
import time
import random

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SAC_discrete-main'))

# Import from discrete SAC (shared components)
from buffer import ReplayBuffer
from logging_utils import (
    create_writer,
    log_hyperparameters,
    log_episode_metrics,
    log_video
)

# Import from this module
from agent import SACContinuous
from utils import save, collect_random


def get_config():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SAC Continuous')
    parser.add_argument("--run_name", type=str, default="SAC_continuous", 
                        help="Run name, default: SAC_continuous")
    parser.add_argument("--env", type=str, default="Pendulum-v1", 
                        help="Gym environment name, default: Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=200, 
                        help="Number of episodes, default: 200")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Max steps per episode (0 = no limit), default: 1000")
    parser.add_argument("--buffer_size", type=int, default=1_000_000, 
                        help="Maximal replay buffer size, default: 1_000_000")
    parser.add_argument("--seed", type=int, default=1, 
                        help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, 
                        help="Log agent behaviour to tensorboard when set to 1, default: 0")
    parser.add_argument("--log_dir", type=str, default="./logs", 
                        help="Directory for TensorBoard logs, default: ./logs")
    parser.add_argument("--save_every", type=int, default=50, 
                        help="Saves the network every x episodes, default: 50")
    parser.add_argument("--batch_size", type=int, default=256, 
                        help="Batch size, default: 256")
    parser.add_argument("--learning_rate", type=float, default=3e-4, 
                        help="Learning rate, default: 3e-4")
    parser.add_argument("--entropy_bonus", type=str, default="None", 
                        help="Fixed entropy bonus (alpha). 'None' = learnable, default: None")
    parser.add_argument("--gamma", type=float, default=0.99, 
                        help="Discount factor, default: 0.99")
    parser.add_argument("--tau", type=float, default=5e-3, 
                        help="Soft update coefficient, default: 5e-3")
    parser.add_argument("--hidden_size", type=int, default=256, 
                        help="Hidden layer size, default: 256")
    parser.add_argument("--obs_buffer_max_len", type=int, default=1, 
                        help="Observation buffer length (frame stacking), default: 1")
    parser.add_argument("--random_samples", type=int, default=10000, 
                        help="Number of random samples to collect before training, default: 10000")
    parser.add_argument("--updates_per_step", type=int, default=1,
                        help="Number of gradient updates per environment step, default: 1")
    parser.add_argument("--start_training_after", type=int, default=1000,
                        help="Start training after N steps in buffer, default: 1000")
    
    args = parser.parse_args()
    return args


def train(config):
    """Main training loop for continuous SAC."""
    # Set seeds for reproducibility
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Create environment
    if config.log_video:
        env = gym.make(config.env, render_mode='rgb_array')
    else:
        env = gym.make(config.env)
    
    # Verify continuous action space
    if not hasattr(env.action_space, 'low'):
        raise ValueError(f"Environment {config.env} does not have a continuous (Box) action space. "
                        f"Got {type(env.action_space)} instead.")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get environment info
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    action_size = action_shape[0]
    
    print(f"Environment: {config.env}")
    print(f"Observation shape: {obs_shape}")
    print(f"Action shape: {action_shape}")
    print(f"Action range: [{env.action_space.low}, {env.action_space.high}]")
    
    # Calculate state size with observation buffer
    obs_buffer_max_len = getattr(config, 'obs_buffer_max_len', 1)
    if obs_buffer_max_len > 1:
        state_size = np.prod(obs_shape) * obs_buffer_max_len
    else:
        state_size = np.prod(obs_shape)
    
    print(f"State size (flattened): {state_size}")
    
    # Initialize tracking variables
    total_steps = 0
    average10 = deque(maxlen=10)
    obs_buffer = deque(maxlen=obs_buffer_max_len)
    
    # Create TensorBoard writer
    writer, log_path = create_writer(config.run_name, config.log_dir)
    print(f"TensorBoard logs: {log_path}")
    
    # Log hyperparameters
    log_hyperparameters(writer, config)
    
    # Parse entropy bonus
    entropy_bonus_str = getattr(config, 'entropy_bonus', 'None')
    entropy_bonus = None if entropy_bonus_str == "None" or entropy_bonus_str is None else float(entropy_bonus_str)
    
    # Create SAC agent
    agent = SACContinuous(
        state_size=state_size,
        action_size=action_size,
        device=device,
        action_space=env.action_space,
        learning_rate=config.learning_rate,
        entropy_bonus=entropy_bonus,
        gamma=config.gamma,
        tau=config.tau,
        hidden_size=config.hidden_size
    )
    
    # Create replay buffer
    buffer = ReplayBuffer(
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        device=device
    )
    
    # Collect random samples for buffer initialization
    print(f"Collecting {config.random_samples} random samples...")
    collect_random(
        env=env, 
        dataset=buffer, 
        num_samples=config.random_samples,
        obs_buffer_max_len=obs_buffer_max_len
    )
    print(f"Buffer size after random collection: {len(buffer)}")
    
    # Setup video recording
    video_dir = os.path.join(log_path, 'videos')
    if config.log_video:
        try:
            env = gym.wrappers.RecordVideo(
                env,
                video_dir,
                episode_trigger=lambda x: (x - 1) % 10 == 0,
                name_prefix="episode"
            )
            print(f"Video recording enabled. Videos saved to: {video_dir}")
        except Exception as e:
            print(f"Warning: Failed to enable video recording: {e}")
            config.log_video = 0
    
    # Training loop
    for episode in range(1, config.episodes + 1):
        obs, info = env.reset()
        
        # Initialize observation buffer
        obs_buffer.clear()
        for _ in range(obs_buffer_max_len):
            obs_buffer.append(obs)
        
        if obs_buffer_max_len > 1:
            state = np.stack(obs_buffer, axis=0).flatten(order="C")
        else:
            state = obs.flatten() if hasattr(obs, 'flatten') else np.array(obs).flatten()
        
        episode_reward = 0
        episode_steps = 0
        policy_loss = 0
        alpha_loss = 0
        critic1_loss = 0
        critic2_loss = 0
        current_alpha = 0
        
        while True:
            # Get action from agent
            action = agent.get_action(state, training=True)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update observation buffer
            obs_buffer.append(next_obs)
            if obs_buffer_max_len > 1:
                next_state = np.stack(obs_buffer, axis=0).flatten(order="C")
            else:
                next_state = next_obs.flatten() if hasattr(next_obs, 'flatten') else np.array(next_obs).flatten()
            
            # Store transition
            buffer.add(state, action, reward, next_state, done)
            
            # Update networks
            if len(buffer) >= config.start_training_after:
                for _ in range(config.updates_per_step):
                    experiences = buffer.sample()
                    policy_loss, alpha_loss, critic1_loss, critic2_loss, current_alpha = agent.learn(
                        total_steps, experiences, gamma=config.gamma
                    )
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            # Check termination
            if done:
                break
            
            # Check max steps
            if config.max_steps > 0 and episode_steps >= config.max_steps:
                break
        
        # Track progress
        average10.append(episode_reward)
        
        print(f"Episode: {episode} | Reward: {episode_reward:.2f} | "
              f"Avg10: {np.mean(average10):.2f} | Steps: {episode_steps} | "
              f"Total Steps: {total_steps}")
        
        # Log to TensorBoard
        log_episode_metrics(
            writer=writer,
            episode=episode,
            reward=episode_reward,
            avg_reward_10=np.mean(average10),
            total_steps=total_steps,
            policy_loss=policy_loss,
            alpha_loss=alpha_loss,
            bellmann_error1=critic1_loss,
            bellmann_error2=critic2_loss,
            current_alpha=current_alpha,
            steps=episode_steps,
            buffer_size=len(buffer)
        )
        
        # Log final metric for hparams
        if episode == config.episodes:
            try:
                writer.add_scalar("hparam/final_avg_reward", np.mean(average10), episode)
            except:
                pass
        
        # Handle video logging
        if (episode % 10 == 0) and config.log_video:
            video_written = False
            for attempt in range(20):
                pattern = os.path.join(video_dir, f"**/episode-episode-{episode}.mp4")
                video_files = glob.glob(pattern, recursive=True)
                if len(video_files) > 0:
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
            
            if video_written:
                time.sleep(1.0)
                log_video(writer, episode, video_dir=video_dir)
        
        # Save model periodically
        if episode % config.save_every == 0:
            save(config, save_name="_continuous", model=agent, ep=episode)
            print(f"Model saved at episode {episode}")
    
    # Final save
    save(config, save_name="_continuous_final", model=agent)
    
    # Cleanup
    if config.log_video and hasattr(env, 'close_video_recorder'):
        env.close_video_recorder()
    
    writer.close()
    env.close()
    
    print("Training complete!")
    print(f"Final average reward (last 10): {np.mean(average10):.2f}")


if __name__ == "__main__":
    config = get_config()
    train(config)

