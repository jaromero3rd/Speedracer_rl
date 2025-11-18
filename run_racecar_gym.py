#!/usr/bin/env python3
"""
Racecar Gym Runner
Script to run and interact with racecar_gym environment
"""

import numpy as np
import time
import sys
import select
import tty
import termios
from collections import OrderedDict


def check_racecar_gym():
    """Check if racecar_gym is installed."""
    try:
        import racecar_gym
        print("✓ racecar_gym is installed")
        
        # Try to get version
        if hasattr(racecar_gym, '__version__'):
            print(f"  Version: {racecar_gym.__version__}")
        
        # List available environments
        try:
            import gymnasium as gym
            from gymnasium.envs.registration import registry
            
            racecar_envs = [env_id for env_id in registry if 'race' in env_id.lower()]
            
            if racecar_envs:
                print(f"\n  Registered environments: {len(racecar_envs)}")
                for env in racecar_envs[:5]:
                    print(f"    - {env}")
                if len(racecar_envs) > 5:
                    print(f"    ... and {len(racecar_envs) - 5} more")
        except:
            pass
        
        print()
        return True
    except ImportError:
        print("✗ racecar_gym is not installed\n")
        print("Install with:")
        print("  pip install racecar_gym\n")
        print("Or from source:")
        print("  git clone https://github.com/axelbr/racecar_gym.git")
        print("  cd racecar_gym")
        print("  pip install -e .\n")
        return False


def list_available_scenarios():
    """List all available scenarios in racecar_gym."""
    import racecar_gym.envs.gym_api
    
    print("Available scenarios:")
    
    # Known scenarios based on working code
    scenarios = [
        'Austria',
        'CircleCW',
        'CircleCCW',
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"  {i}. {scenario} (SingleAgent{scenario}-v0)")
    print()
    
    return scenarios


def create_environment(scenario='Austria', render_mode='human'):
    """
    Create a racecar_gym environment.
    
    Args:
        scenario: Name of the scenario/track (e.g., 'Austria', 'CircleCW')
        render_mode: 'human', 'rgb_array_birds_eye', or 'rgb_array_follow'
    
    Returns:
        env: The gym environment
    """
    import gymnasium
    import racecar_gym.envs.gym_api
    
    print(f"Creating environment: {scenario}")
    print(f"Render mode: {render_mode}\n")
    
    # Correct format: SingleAgent{Scenario}-v0
    env_id = f'SingleAgent{scenario}-v0'
    
    try:
        env = gymnasium.make(env_id, render_mode=render_mode)
        print(f"✓ Environment created: {env_id}\n")
        return env
    except Exception as e:
        print(f"✗ Failed to create environment: {e}\n")
        print("Available scenarios might be:")
        print("  - Austria")
        print("  - CircleCW")
        print("  - CircleCCW")
        print("  - (check racecar_gym documentation for more)\n")
        return None


def random_policy(observation):
    """
    Random policy - returns random actions.
    Actions are in the range [-1, 1] for steering and acceleration.
    
    Args:
        observation: Current observation from environment
    
    Returns:
        action: Random action as OrderedDict with 'motor' and 'steering' keys
    """
    steering = np.random.uniform(-1, 1)
    acceleration = np.array([1])  # Full acceleration
    ordered_dict = OrderedDict()
    ordered_dict["motor"] = acceleration
    ordered_dict["steering"] = steering
    return ordered_dict


def simple_policy(observation):
    """
    Simple forward-only policy.
    Go straight and accelerate.
    
    Args:
        observation: Current observation from environment
    
    Returns:
        action: Action as OrderedDict with 'motor' and 'steering' keys
    """
    steering = 0.0
    acceleration = np.array([1.0])  # Full acceleration
    ordered_dict = OrderedDict()
    ordered_dict["motor"] = acceleration
    ordered_dict["steering"] = steering
    return ordered_dict


def run_episode(env, policy_fn=random_policy, max_steps=1000, render=True, reset_mode='grid'):
    """
    Run one episode with the given policy.
    
    Args:
        env: The gym environment
        policy_fn: Function that takes observation and returns action
        max_steps: Maximum steps per episode
        render: Whether to render the environment
        reset_mode: 'grid' or 'random' - how to place agent at start
    
    Returns:
        total_reward: Total reward accumulated
        steps: Number of steps taken
    """
    print(f"Starting episode (reset mode: {reset_mode})...")
    
    # Reset with options as shown in working code
    observation = env.reset(options=dict(mode=reset_mode))
    
    # Handle tuple return from reset
    if isinstance(observation, tuple):
        observation = observation[0]
    
    total_reward = 0
    step = 0
    
    for step in range(max_steps):
        # Get action from policy
        action = policy_fn(observation)
        
        # Step environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        # Render periodically (every 30 steps as in working code)
        if render and step % 30 == 0:
            env.render()
        
        # Check if episode is done
        done = terminated or truncated
        
        if done:
            print(f"Episode finished after {step + 1} steps")
            break
        
        # Small delay for visualization
        if render:
            time.sleep(0.01)
    
    print(f"Total reward: {total_reward:.2f}\n")
    
    return total_reward, step + 1


class KeyboardController:
    """
    Keyboard controller for manual racecar control.
    Controls:
    - Arrow keys or WASD for steering and acceleration
    - 'q' to quit
    """
    
    def __init__(self):
        self.steering = 0.0
        self.motor = 0.0
        self.quit_requested = False
        self.old_settings = None
        # Track active keys (keys currently being pressed)
        self.active_keys = set()
        # Track last update time for each key (for timeout-based release detection)
        self.key_times = {}
        # Buffer for incomplete escape sequences
        self.escape_buffer = ''
        
        try:
            # Save terminal settings
            self.old_settings = termios.tcgetattr(sys.stdin)
            # Set terminal to cbreak mode (character-by-character input)
            tty.setcbreak(sys.stdin.fileno())
            self.terminal_configured = True
        except (termios.error, AttributeError, OSError) as e:
            print(f"Warning: Could not configure terminal for keyboard input: {e}")
            print("Keyboard control may not work properly.")
            self.terminal_configured = False
        
        print("\nKeyboard Controls:")
        print("  Arrow Keys or WASD: Control steering and acceleration")
        print("  W/Up Arrow: Forward (motor=1, steering=0)")
        print("  S/Down Arrow: Backward (motor=-1, steering=0)")
        print("  A/Left Arrow: Left (steering=-1)")
        print("  D/Right Arrow: Right (steering=1)")
        print("  Keys can be combined (e.g., W+D = forward+right)")
        print("  Q: Quit episode")
        print("  (Press keys to control, no need to press Enter)\n")
    
    def cleanup(self):
        """Restore terminal settings."""
        if self.terminal_configured and self.old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except:
                pass
    
    def __del__(self):
        """Restore terminal settings on destruction."""
        self.cleanup()
    
    def get_all_keys(self):
        """Get all available keypresses without requiring Enter."""
        if not self.terminal_configured:
            return []
        
        keys = []
        try:
            # Read all available keys in the buffer
            while select.select([sys.stdin], [], [], 0.0)[0]:
                key = sys.stdin.read(1)
                if key:
                    keys.append(key)
        except (OSError, ValueError):
            pass
        
        return keys
    
    def update(self):
        """Update control state based on keyboard input. Returns True if quit requested."""
        if not self.terminal_configured:
            return self.quit_requested
        
        current_time = time.time()
        keys = self.get_all_keys()
        
        # Handle incomplete escape sequence from previous update
        if self.escape_buffer:
            if keys:
                # Combine buffer with new keys
                combined = list(self.escape_buffer) + keys
                keys = combined
                self.escape_buffer = ''
            else:
                # No new keys, keep waiting (don't process yet)
                return self.quit_requested
        
        # Process all keys in the buffer
        i = 0
        while i < len(keys):
            key = keys[i]
            
            # Handle arrow key escape sequences (3 characters: ESC [ A/B/C/D)
            if key == '\x1b':
                # Check if we have enough characters for the full sequence
                if i + 2 < len(keys):
                    seq = keys[i+1] + keys[i+2]
                    if seq == '[A':  # Up arrow -> forward
                        self.active_keys.add('forward')
                        self.key_times['forward'] = current_time
                        i += 3
                        continue
                    elif seq == '[B':  # Down arrow -> back
                        self.active_keys.add('back')
                        self.key_times['back'] = current_time
                        i += 3
                        continue
                    elif seq == '[C':  # Right arrow -> right steering
                        self.active_keys.add('right')
                        self.key_times['right'] = current_time
                        i += 3
                        continue
                    elif seq == '[D':  # Left arrow -> left steering
                        self.active_keys.add('left')
                        self.key_times['left'] = current_time
                        i += 3
                        continue
                    else:
                        # Not an arrow key sequence, skip ESC and continue
                        i += 1
                        continue
                else:
                    # Incomplete sequence, save remaining keys for next update
                    self.escape_buffer = ''.join(keys[i:])
                    break
            
            key_lower = key.lower()
            
            # WASD controls
            if key_lower == 'w':  # Forward
                self.active_keys.add('forward')
                self.key_times['forward'] = current_time
            elif key_lower == 's':  # Back
                self.active_keys.add('back')
                self.key_times['back'] = current_time
            elif key_lower == 'a':  # Left steering
                self.active_keys.add('left')
                self.key_times['left'] = current_time
            elif key_lower == 'd':  # Right steering
                self.active_keys.add('right')
                self.key_times['right'] = current_time
            elif key_lower == 'q':  # Quit
                self.quit_requested = True
                print("  [Key: Q] Quitting...")
                return True
            
            i += 1
        
        # Remove keys that haven't been seen recently (key release detection)
        # Keys timeout after 0.15 seconds of not being seen (increased for better combo support)
        timeout = 0.15
        keys_to_remove = []
        for key in self.active_keys:
            if key in self.key_times:
                if current_time - self.key_times[key] > timeout:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.active_keys.discard(key)
            if key in self.key_times:
                del self.key_times[key]
        
        # Compute motor and steering from active keys
        motor = 0.0
        steering = 0.0
        
        # Forward (W or Up Arrow)
        if 'forward' in self.active_keys:
            motor += 1.0
        # Back (S or Down Arrow)
        if 'back' in self.active_keys:
            motor -= 1.0
        # Right steering (D or Right Arrow)
        if 'right' in self.active_keys:
            steering += 1.0
        # Left steering (A or Left Arrow)
        if 'left' in self.active_keys:
            steering -= 1.0
        
        # Clamp values to [-1, 1]
        self.motor = max(-1.0, min(1.0, motor))
        self.steering = max(-1.0, min(1.0, steering))
        
        # Print debug info when keys change
        if keys:
            active_str = ', '.join(sorted(self.active_keys)) if self.active_keys else 'none'
            print(f"  [Keys: {active_str}] Steering: {self.steering:.2f}, Motor: {self.motor:.2f}")
        
        return self.quit_requested
    
    def get_action(self):
        """Get current action based on keyboard state."""
        ordered_dict = OrderedDict()
        ordered_dict["motor"] = np.array([self.motor])
        ordered_dict["steering"] = self.steering
        return ordered_dict
    
    def reset(self):
        """Reset controls to default."""
        self.steering = 0.0
        self.motor = 0.0
        self.quit_requested = False
        self.active_keys.clear()
        self.key_times.clear()
        self.escape_buffer = ''


def demo_keyboard_control():
    """Run demo with keyboard control."""
    print("\n" + "="*80)
    print("KEYBOARD CONTROL DEMO")
    print("="*80 + "\n")
    
    if not check_racecar_gym():
        return
    
    # List scenarios
    scenarios = list_available_scenarios()
    
    # Select scenario
    choice = input("Enter scenario number (or press Enter for default): ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(scenarios):
        scenario = scenarios[int(choice) - 1]
    else:
        scenario = 'Austria'  # Default to Austria
    
    # Create environment
    env = create_environment(scenario, render_mode='human')
    
    if env is None:
        return
    
    # Print environment info
    print("Environment information:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print()
    
    controller = None
    try:
        # Create keyboard controller
        controller = KeyboardController()
        
        if not controller.terminal_configured:
            print("ERROR: Terminal could not be configured for keyboard input.")
            print("Keyboard control requires a proper terminal. Please run this script")
            print("from a terminal (not from an IDE's integrated terminal).")
            env.close()
            return
        
        print("Starting keyboard-controlled episode...")
        print("Use arrow keys or WASD to control the car. Press 'q' to quit.")
        print("\nIMPORTANT: After the rendering window opens, click back on this terminal")
        print("window to give it focus so it can receive keyboard input!")
        print("(The rendering window may steal focus when it opens)\n")
        
        # Reset environment
        obs, info = env.reset(options=dict(mode='grid'))
        
        # Give user time to see the message and click on terminal
        print("Environment reset. Starting in 2 seconds...")
        print("(Click on this terminal window now if the render window opened)\n")
        time.sleep(2)
        
        controller.reset()
        
        total_reward = 0
        step_count = 0
        done = False
        
        while not done and not controller.quit_requested and step_count < 5000:
            # Update controller and check for quit
            controller.update()
            if controller.quit_requested:
                break
            
            # Get action from keyboard
            action = controller.get_action()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            step_count += 1
            
            # Render
            if step_count % 1 == 0:  # Render every step for responsive control
                env.render()
            
            # Small delay for visualization
            time.sleep(0.01)
        
        print(f"\nEpisode finished!")
        print(f"  Steps: {step_count}")
        print(f"  Total reward: {total_reward:.2f}")
        if controller.quit_requested:
            print("  (Quit by user)")
        print()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up controller
        if controller is not None:
            controller.cleanup()
        env.close()
        print("Demo complete!")


def demo_random_policy():
    """Run demo with random policy."""
    print("\n" + "="*80)
    print("RANDOM POLICY DEMO")
    print("="*80 + "\n")
    
    if not check_racecar_gym():
        return
    
    # List scenarios
    scenarios = list_available_scenarios()
    
    # Select scenario
    choice = input("Enter scenario number (or press Enter for default): ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(scenarios):
        scenario = scenarios[int(choice) - 1]
    else:
        scenario = 'Austria'  # Default to Austria
    
    # Create environment
    env = create_environment(scenario, render_mode='human')
    
    if env is None:
        return
    
    # Print environment info
    print("Environment information:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print()
    
    # Run episodes
    num_episodes = 3
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        print("-" * 40)
        
        reward, steps = run_episode(env, random_policy, max_steps=500)
        
        print(f"Reward: {reward:.2f}, Steps: {steps}\n")
    
    env.close()
    print("Demo complete!")


def benchmark_performance():
    """Benchmark the environment performance."""
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK")
    print("="*80 + "\n")
    
    if not check_racecar_gym():
        return
    
    # Create environment without rendering
    env = create_environment('austria', render_mode=None)
    
    if env is None:
        return
    
    print("Running 100 steps to benchmark FPS...\n")
    
    observation, info = env.reset()
    
    start_time = time.time()
    num_steps = 100
    
    for step in range(num_steps):
        action = random_policy(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            observation, info = env.reset()
    
    elapsed = time.time() - start_time
    fps = num_steps / elapsed
    
    print(f"Completed {num_steps} steps in {elapsed:.2f} seconds")
    print(f"Performance: {fps:.2f} FPS\n")
    
    env.close()


def record_episode():
    """Record an episode to video."""
    print("\n" + "="*80)
    print("RECORD EPISODE")
    print("="*80 + "\n")
    
    if not check_racecar_gym():
        return
    
    try:
        import gymnasium as gym
        from gymnasium.wrappers import RecordVideo
    except ImportError:
        print("✗ Recording requires gymnasium")
        print("  pip install gymnasium\n")
        return
    
    # Create environment with video recording
    env = create_environment('austria', render_mode='rgb_array')
    
    if env is None:
        return
    
    # Wrap with RecordVideo
    env = RecordVideo(env, './videos', episode_trigger=lambda x: True)
    
    print("Recording episode to ./videos/\n")
    
    observation, info = env.reset()
    
    for step in range(500):
        action = random_policy(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    env.close()
    
    print("\n✓ Video saved to ./videos/\n")


def diagnose_environments():
    """Diagnose and list all available racecar_gym environments."""
    print("\n" + "="*80)
    print("ENVIRONMENT DIAGNOSTICS")
    print("="*80 + "\n")
    
    if not check_racecar_gym():
        return
    
    import gymnasium as gym
    import racecar_gym
    
    print("Method 1: Check gymnasium registry")
    print("-" * 40)
    
    try:
        from gymnasium.envs.registration import registry
        
        all_envs = list(registry.keys())
        print(f"Total registered environments: {len(all_envs)}\n")
        
        # Filter for racecar envs
        racecar_envs = [env for env in all_envs if 'race' in env.lower() or 'car' in env.lower()]
        
        if racecar_envs:
            print("Racecar-related environments:")
            for env in racecar_envs:
                print(f"  ✓ {env}")
        else:
            print("✗ No racecar environments found in registry")
        
        print()
    except Exception as e:
        print(f"Error checking registry: {e}\n")
    
    print("Method 2: Check racecar_gym module")
    print("-" * 40)
    
    try:
        # Check what's available in racecar_gym
        import inspect
        
        print("Functions/Classes in racecar_gym:")
        for name, obj in inspect.getmembers(racecar_gym):
            if not name.startswith('_'):
                print(f"  - {name}: {type(obj).__name__}")
        
        print()
        
        # Try to find scenarios
        if hasattr(racecar_gym, 'scenarios'):
            print("Available scenarios:")
            scenarios = racecar_gym.scenarios
            for name in dir(scenarios):
                if not name.startswith('_'):
                    print(f"  - {name}")
        
        print()
    except Exception as e:
        print(f"Error checking module: {e}\n")
    
    print("Method 3: Try creating environment with different methods")
    print("-" * 40)
    
    test_scenarios = ['austria', 'circle_cw', 'vienna']
    
    for scenario in test_scenarios:
        print(f"\nTesting scenario: {scenario}")
        
        # Try different creation methods
        methods = [
            (f"gym.make('SingleAgentRaceEnv-{scenario}-v0')", 
             lambda: gym.make(f'SingleAgentRaceEnv-{scenario}-v0')),
            (f"gym.make('{scenario}')", 
             lambda: gym.make(scenario)),
            (f"racecar_gym.make('{scenario}')", 
             lambda: racecar_gym.make(scenario)),
        ]
        
        for method_name, method_fn in methods:
            try:
                env = method_fn()
                print(f"  ✓ {method_name} - SUCCESS")
                env.close()
                break
            except Exception as e:
                print(f"  ✗ {method_name} - Failed: {str(e)[:50]}")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80 + "\n")
    print("Based on the diagnostics above, use the method that succeeded.")
    print("If none worked, racecar_gym may not be properly installed.\n")


def inspect_observation():
    """Inspect the observation space structure."""
    print("\n" + "="*80)
    print("OBSERVATION SPACE INSPECTION")
    print("="*80 + "\n")
    
    if not check_racecar_gym():
        return
    
    env = create_environment('austria', render_mode=None)
    
    if env is None:
        return
    
    observation, info = env.reset()
    
    print("Observation structure:")
    
    if isinstance(observation, dict):
        print("  Type: Dictionary\n")
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}:")
                print(f"    Shape: {value.shape}")
                print(f"    Dtype: {value.dtype}")
                print(f"    Range: [{value.min():.3f}, {value.max():.3f}]")
            else:
                print(f"  {key}: {value}")
            print()
    elif isinstance(observation, np.ndarray):
        print(f"  Type: Array")
        print(f"  Shape: {observation.shape}")
        print(f"  Dtype: {observation.dtype}")
        print(f"  Range: [{observation.min():.3f}, {observation.max():.3f}]\n")
    else:
        print(f"  Type: {type(observation)}")
        print(f"  Value: {observation}\n")
    
    print("Action space:")
    print(f"  {env.action_space}\n")
    
    env.close()


def main():
    """Main function."""
    print("\n" + "█"*80)
    print("  RACECAR GYM RUNNER")
    print("█"*80 + "\n")
    
    if not check_racecar_gym():
        return
    
    print("Select mode:")
    print("  1. Random policy demo (3 episodes)")
    print("  2. Keyboard control demo (manual control)")
    print("  3. Performance benchmark")
    print("  4. Inspect observation space")
    print("  5. Record episode to video")
    print("  6. Diagnose environments (find correct env names)")
    print("  7. Exit\n")
    
    choice = input("Enter choice (1-7): ").strip()
    
    if choice == '1':
        demo_random_policy()
    elif choice == '2':
        demo_keyboard_control()
    elif choice == '3':
        benchmark_performance()
    elif choice == '4':
        inspect_observation()
    elif choice == '5':
        record_episode()
    elif choice == '6':
        diagnose_environments()
    elif choice == '7':
        print("\nExiting...\n")
    else:
        print("\nInvalid choice\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
