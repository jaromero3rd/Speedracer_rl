#!/usr/bin/env python3
"""
Run a trained SAC policy
Loads a saved actor model and runs it in the environment.
"""

import numpy as np
import gymnasium as gym
import torch
import argparse
import os
import sys
from collections import deque

# Add racecar_gym to path
racecar_gym_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'racecar_gym')
sys.path.append(racecar_gym_path)
import racecar_gym.envs.gym_api

from race_agent import SAC
from utils import flatten_racecar_obs, map_racecar_action


def run_policy(model_path: str, env_name: str = 'SingleAgentAustria-v0', 
               num_episodes: int = 5, render: bool = True, max_steps: int = 1000,
               obs_buffer_max_len: int = 1):
    """
    Run a trained SAC policy.
    
    Args:
        model_path: Path to saved actor model (.pth file)
        env_name: Environment name
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        max_steps: Maximum steps per episode
        obs_buffer_max_len: Length of observation buffer (must match training)
    """
    print(f"Loading policy from: {model_path}")
    print(f"Environment: {env_name}")
    print(f"Episodes: {num_episodes}")
    print(f"Observation buffer length: {obs_buffer_max_len}\n")
    
    # Create environment
    env = gym.make(env_name, render_mode='human' if render else None)
    
    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Get sample observation to determine network architecture
    sample_obs, _ = env.reset()
    flattened_sample = flatten_racecar_obs(sample_obs)
    state_size_flat = len(flattened_sample) * obs_buffer_max_len
    
    # Extract observation dimensions for network
    image_shape = tuple(sample_obs['rgb_camera'].shape)  # (H, W, C)
    vel_acc_dim = sample_obs['velocity'].shape[0] + sample_obs['acceleration'].shape[0]
    
    print(f"Network architecture:")
    print(f"  State size: {state_size_flat}")
    print(f"  Image shape: {image_shape}")
    print(f"  Velocity/Acceleration dim: {vel_acc_dim}")
    print(f"  Action size: 6\n")
    
    # Create SAC agent with same configuration as training
    # Note: We only need the actor network for inference
    agent = SAC(
        state_size=state_size_flat,
        action_size=6,
        device=device,
        learning_rate=5e-4,  # Not used for inference, but required
        entropy_bonus=None,  # Learnable alpha (matches training default)
        epsilon=0.0,  # No exploration during evaluation
        image_shape=image_shape,
        vel_acc_dim=vel_acc_dim,
        cnn_latent_dim=256
    )
    
    # Load the saved actor model
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        agent.actor_local.load_state_dict(torch.load(model_path, map_location=device))
        agent.actor_local.eval()  # Set to evaluation mode
        print(f"✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return
    
    episode_rewards = []
    episode_lengths = []
    
    print("Starting policy evaluation...\n")
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        print("-" * 40)
        
        # Reset environment
        obs, info = env.reset()
        
        # Initialize observation buffer
        obs_buffer = deque(maxlen=obs_buffer_max_len)
        flat_obs = flatten_racecar_obs(obs)
        
        # Fill buffer with initial observation
        for _ in range(obs_buffer_max_len):
            obs_buffer.append(flat_obs)
        state = np.stack(obs_buffer, axis=0).flatten(order="C")
        
        total_reward = 0.0
        step_count = 0
        done = False
        
        while not done and step_count < max_steps:
            # Get action from agent (training=False for deterministic policy)
            action_idx = agent.get_action(state, training=False)
            
            # Map discrete action to racecar dictionary action
            action = map_racecar_action(action_idx)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update observation buffer
            flat_next_obs = flatten_racecar_obs(next_obs)
            obs_buffer.append(flat_next_obs)
            next_state = np.stack(obs_buffer, axis=0).flatten(order="C")
            
            obs = next_obs
            state = next_state
            total_reward += reward
            step_count += 1
            
            if render:
                env.render()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        print(f"  Steps: {step_count}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Done: {done}\n")
    
    env.close()
    
    # Print statistics
    print("=" * 40)
    print("Evaluation Results:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Min reward: {np.min(episode_rewards):.2f}")
    print(f"  Max reward: {np.max(episode_rewards):.2f}")
    print("=" * 40)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run a trained SAC policy')
    parser.add_argument('--model', type=str, default='trained_models/SACSAC_discrete0.pth',
                        help='Path to saved actor model file (default: trained_models/SACSAC_discrete0.pth)')
    parser.add_argument('--env', type=str, default='SingleAgentAustria-v0',
                        help='Environment name (default: SingleAgentAustria-v0)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run (default: 5)')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Maximum steps per episode (default: 1000)')
    parser.add_argument('--obs_buffer_len', type=int, default=1,
                        help='Observation buffer length (must match training, default: 1)')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering')
    
    args = parser.parse_args()
    
    try:
        run_policy(
            model_path=args.model,
            env_name=args.env,
            num_episodes=args.episodes,
            render=not args.no_render,
            max_steps=args.max_steps,
            obs_buffer_max_len=args.obs_buffer_len
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
