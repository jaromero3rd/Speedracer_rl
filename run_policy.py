#!/usr/bin/env python3
"""
Run a trained DQN policy
Loads a saved model and runs it in the environment.
"""

import numpy as np
import gymnasium
import racecar_gym.envs.gym_api
import argparse
from collections import OrderedDict
from dqn import DQNAgent


def run_policy(model_path: str, env_name: str = 'SingleAgentAustria-v0', 
               num_episodes: int = 5, render: bool = True, max_steps: int = 1000):
    """
    Run a trained policy.
    
    Args:
        model_path: Path to saved model (.pth file)
        env_name: Environment name
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        max_steps: Maximum steps per episode
    """
    print(f"Loading policy from: {model_path}")
    print(f"Environment: {env_name}")
    print(f"Episodes: {num_episodes}\n")
    
    # Create environment
    env = gymnasium.make(env_name, render_mode='human' if render else None)
    
    # Create agent with same configuration as training
    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=1e-4,  # Not used for inference, but required
        gamma=0.99,
        epsilon_start=0.0,  # No exploration during evaluation
        epsilon_end=0.0,
        epsilon_decay=1.0,
        buffer_size=1000,  # Not used for inference
        batch_size=64,
        target_update_freq=100
    )
    
    # Load the saved model
    try:
        agent.load(model_path)
        print(f"✓ Model loaded successfully")
        print(f"  Epsilon: {agent.epsilon:.4f}")
        print(f"  Update counter: {agent.update_counter}\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        env.close()
        return
    
    episode_rewards = []
    episode_lengths = []
    
    print("Starting policy evaluation...\n")
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        print("-" * 40)
        
        # Reset environment
        obs, info = env.reset(options=dict(mode='grid'))
        
        total_reward = 0.0
        step_count = 0
        done = False
        
        while not done and step_count < max_steps:
            # Select action (training=False for greedy policy, no exploration)
            action, action_idx = agent.select_action(obs, training=False)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            obs = next_obs
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
    parser = argparse.ArgumentParser(description='Run a trained DQN policy')
    parser.add_argument('--model', type=str, default='dqn_info_rewards_model.pth',
                        help='Path to saved model file')
    parser.add_argument('--env', type=str, default='SingleAgentAustria-v0',
                        help='Environment name')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Maximum steps per episode')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering')
    
    args = parser.parse_args()
    
    try:
        run_policy(
            model_path=args.model,
            env_name=args.env,
            num_episodes=args.episodes,
            render=not args.no_render,
            max_steps=args.max_steps
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

