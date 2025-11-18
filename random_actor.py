#!/usr/bin/env python3
"""
Random Actor Script
Runs random actions in the racecar_gym environment for testing and data collection.
"""

import numpy as np
import gymnasium
import racecar_gym.envs.gym_api
from time import sleep
from collections import OrderedDict


def random_policy(observation):
    """
    Random policy - returns random actions.
    Actions are in the range [-1, 1] for steering and acceleration.
    """
    steering = np.random.uniform(-1, 1)
    acceleration = np.array([1])
    ordered_dict = OrderedDict()
    ordered_dict["motor"] = acceleration
    ordered_dict["steering"] = steering
    return ordered_dict



def run_random_actor(env_name='SingleAgentAustria-v0', num_episodes=5, max_steps=1000, render_mode='human'):
    """
    Run random actions in the racecar_gym environment.
    
    Args:
        env_name: Name of the environment to use
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        render_mode: Rendering mode ('human', 'rgb_array_birds_eye', 'rgb_array_follow', or None)
    """
    print(f"Creating environment: {env_name}")
    print(f"Render mode: {render_mode}")
    print(f"Number of episodes: {num_episodes}\n")
    
    # Create environment
    env = gymnasium.make(env_name, render_mode=render_mode)
    
    print("Environment information:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}\n")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        print("-" * 40)
        
        # Reset environment
        obs, info = env.reset(options=dict(mode='grid'))

        
        total_reward = 0
        step_count = 0
        done = False
        
        while not done and step_count < max_steps:
            # Sample random action from action space
            # action = env.action_space.sample()
            action = random_policy(obs)
            print(action)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            done = terminated or truncated
            
            # Render periodically
            if render_mode == 'human' and step_count % 30 == 0:
                env.render()
            
            # Small delay for visualization
            if render_mode == 'human':
                sleep(0.01)
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        print(f"  Steps: {step_count}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Done: {done}\n")
    
    env.close()
    
    # Print statistics
    print("=" * 40)
    print("Statistics:")
    print(f"  Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Min reward: {np.min(episode_rewards):.2f}")
    print(f"  Max reward: {np.max(episode_rewards):.2f}")
    print("=" * 40)


def main():
    """Main function to run random actor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run random actions in racecar_gym environment')
    parser.add_argument('--env', type=str, default='SingleAgentAustria-v0',
                        help='Environment name (default: SingleAgentAustria-v0)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run (default: 5)')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Maximum steps per episode (default: 1000)')
    parser.add_argument('--render-mode', type=str, default='human',
                        choices=['human', 'rgb_array_birds_eye', 'rgb_array_follow', 'none'],
                        help='Rendering mode (default: human)')
    
    args = parser.parse_args()
    
    render_mode = None if args.render_mode == 'none' else args.render_mode
    
    try:
        run_random_actor(
            env_name=args.env,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            render_mode=render_mode
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

