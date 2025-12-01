import os
import sys

import gymnasium as gym
import numpy as np
from collections import deque
import torch
import argparse
import glob
import time
from buffer import ReplayBuffer
from utils import save, save_agent, collect_random, collect_random_RND, flatten_racecar_obs, map_racecar_action, load_model, save_model_number
import random
from race_agent import SAC
from logging_utils import (
    create_writer,
    log_hyperparameters,
    log_episode_metrics,
    log_video
)

# moving the racecar_gym path to the top of the file to avoid import errors
racecar_gym_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'racecar_gym')
sys.path.append(racecar_gym_path)


# Compatibility fix for deprecated np.float and np.int in racecar_gym
# This allows racecar_gym to work with newer NumPy versions
import numpy as np
if not hasattr(np, 'float'):
    np.float = np.float64  # Add deprecated alias for compatibility
if not hasattr(np, 'int'):
    np.int = np.int64  # Add deprecated alias for compatibility

import racecar_gym.envs.gym_api

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="SAC", help="Run name, default: SAC")
    parser.add_argument("--env", type=str, default="SingleAgentAustria-v0", help="Racecar gym environment name, default: SingleAgentAustria-v0")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes, default: 100")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to tensorboard when set to 1, default: 0, allow for viewing of training")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for TensorBoard logs, default: ./logs")
    parser.add_argument("--save_every", type=int, default=25, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate, default: 0.001")
    parser.add_argument("--entropy_bonus", type=str, default=0.02, help="Fixed entropy bonus (alpha). 'None' = learnable, default: None")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Epsilon for epsilon-greedy exploration, default: 0.0")
    parser.add_argument("--obs_buffer_max_len", type=int, default=1, help="Observation buffer length, default: 1")
    parser.add_argument("--num_threads", type=int, default=None, help="Number of PyTorch threads (default: auto-detect, uses all cores)")
    parser.add_argument("--load_model", type=str, default=None, help="Path to pretrained model to load as starting point (default: None)")
 
    args = parser.parse_args()
    return args

def train(config):
    # Optimization 1: PyTorch Threading - Use all available CPU cores
    import os
    if config.num_threads is None:
        # Auto-detect: use all available CPU cores
        num_threads = os.cpu_count() or 14
    else:
        num_threads = config.num_threads
    
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    print(f"✓ PyTorch configured with {num_threads} threads for parallel computation")
    
    # Optimization 2: Device selection (CUDA or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Optimization 3: Enable optimizations for faster execution
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # Faster, but non-deterministic
    
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Create environment(s) - single env for training
    # Optimization 4: Environment speed - disable rendering for faster training
    if config.log_video:
        env = gym.make(config.env, render_mode='human')
    else:
        # Use None for headless operation (no rendering, faster training)
        env = gym.make(config.env, render_mode=None)
    
    print(f"✓ Using device: {device}")
    
    steps = 0
    average10 = deque(maxlen=10)
    
    # Performance timing
    episode_times = []
    training_times = []
    env_step_times = []

    obs_buffer_max_len = getattr(config, 'obs_buffer_max_len', 4)
    obs_buffer = deque(maxlen=obs_buffer_max_len)
    total_steps = 0
    
    # Create TensorBoard writer
    writer, log_path = create_writer(config.run_name, config.log_dir)
    
    # Log hyperparameters
    log_hyperparameters(writer, config)
    
    # Calculate state size from racecar dict observation space
    sample_obs, _ = env.reset()
    flattened_sample = flatten_racecar_obs(sample_obs)
    state_size_flat = len(flattened_sample) * obs_buffer_max_len
    action_size = 6  
    
    # Extract observation dimensions for network
    image_shape = tuple(sample_obs['rgb_camera'].shape)  
    vel_acc_dim = sample_obs['velocity'].shape[0] + sample_obs['acceleration'].shape[0] 
    
    # Get hyperparameters from config (with defaults)
    learning_rate = getattr(config, 'learning_rate', 5e-4)
    entropy_bonus_str = getattr(config, 'entropy_bonus', 'None')

    # Convert string "None" to actual None
    entropy_bonus = None if entropy_bonus_str == "None" or entropy_bonus_str is None else float(entropy_bonus_str)
    epsilon = getattr(config, 'epsilon', 0.0)
    
    agent = SAC(state_size=state_size_flat,
                     action_size=action_size,
                     device=device,
                     learning_rate=learning_rate,
                     entropy_bonus=entropy_bonus,
                     epsilon=epsilon,
                     image_shape=image_shape,
                     vel_acc_dim=vel_acc_dim)

    # Optimization 5: torch.compile() for faster execution (PyTorch 2.0+) - Always enabled if available
    compilation_enabled = False
    try:
        # Check if torch.compile is available (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            # Disable donated buffers to avoid issues with retain_graph=True in backward pass
            import torch._functorch.config as functorch_config
            functorch_config.donated_buffer = False
            
            # Try to clean up any stale lock files first
            lock_dir = '/tmp/torchinductor_jaimeromero/precompiled_headers/locks'
            if os.path.exists(lock_dir):
                try:
                    for lock_file in os.listdir(lock_dir):
                        lock_path = os.path.join(lock_dir, lock_file)
                        if os.path.isfile(lock_path):
                            # Try to remove stale locks (older than 1 minute)
                            if time.time() - os.path.getmtime(lock_path) > 60:
                                os.remove(lock_path)
                except:
                    pass  # Ignore errors when cleaning locks
            
            # Use 'default' mode instead of 'reduce-overhead' to avoid C++ compilation issues
            # 'default' mode is more stable and doesn't require C++ compilation
            agent.actor_local = torch.compile(agent.actor_local, mode='default')
            agent.critic1 = torch.compile(agent.critic1, mode='default')
            agent.critic2 = torch.compile(agent.critic2, mode='default')
            compilation_enabled = True
            print("✓ Networks compiled with torch.compile() for faster execution")
        else:
            print("⚠ torch.compile() not available (requires PyTorch 2.0+), continuing without compilation")
    except Exception as e:
        print(f"⚠ torch.compile() failed: {e}")
        print("  Continuing without compilation...")
        # Ensure networks are not compiled if compilation failed
        compilation_enabled = False
    
    # Update target networks after optimization to ensure they match the optimized structure
    # This is critical because torch.compile wraps models with _orig_mod prefix
    try:
        def unwrap_state_dict(model_state_dict):
            """Remove _orig_mod prefix from state dict keys if present (from torch.compile)."""
            unwrapped = {}
            for key, value in model_state_dict.items():
                if key.startswith('_orig_mod.'):
                    unwrapped[key.replace('_orig_mod.', '')] = value
                else:
                    unwrapped[key] = value
            return unwrapped
        
        # Get state dict from compiled models and unwrap if needed
        critic1_state = unwrap_state_dict(agent.critic1.state_dict())
        critic2_state = unwrap_state_dict(agent.critic2.state_dict())
        
        # Load with strict=False to handle any minor differences
        agent.critic1_target.load_state_dict(critic1_state, strict=False)
        agent.critic2_target.load_state_dict(critic2_state, strict=False)
        print("✓ Target networks synchronized with optimized networks")
    except Exception as e:
        print(f"⚠ Warning: Could not synchronize target networks: {e}")
        print("  Target networks will be updated during first training step")

    # Load pretrained model if specified
    if config.load_model is not None:
        load_model(agent, config.load_model, device)

    buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)
    
    # Collect random samples for buffer initialization
    # collect_random_RND(env=env, dataset=buffer, num_samples=10000, 
    #                  obs_buffer_max_len=obs_buffer_max_len, device=device)
    collect_random(env=env, dataset=buffer, num_samples=10000, 
                     obs_buffer_max_len=obs_buffer_max_len)
    
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
        episode_start_time = time.time()
        obs, info = env.reset()
        
        # Reinitialize observation buffer for new episode
        obs_buffer.clear()
        # Flatten racecar observations if needed
        flat_obs = flatten_racecar_obs(obs)
        for _ in range(obs_buffer_max_len):
            obs_buffer.append(flat_obs)
        state = np.stack(obs_buffer, axis=0).flatten(order="C")
  

        episode_steps = 0
        rewards = 0
        env_step_time_total = 0.0
        training_time_total = 0.0
        
        while True:
            action_idx = agent.get_action(state)
            steps += 1
            # Map discrete action to racecar dictionary action
            action = map_racecar_action(action_idx)
            
            # Time environment step
            env_step_start = time.time()
            next_obs, reward, terminated, truncated, info = env.step(action)
            env_step_time_total += time.time() - env_step_start

            # Flatten racecar observations if needed
            flat_next_obs = flatten_racecar_obs(next_obs)
            obs_buffer.append(flat_next_obs)
            next_state = np.stack(obs_buffer, axis=0).flatten(order="C")

            done = terminated or truncated 
            # TODO: Add reward functions dependent on states and state -> z latent space embedding function here
            # print(f"enviroment reward: {reward}")
            # print(f"state: {state}")
            living_reward = 1
            death_reward = 0
            finish_reward = 0
            velocity_reward = np.sum(np.abs(next_obs['velocity']))*10
            if terminated:
                death_reward = -10
            if truncated:
                finish_reward = 1000
            reward = reward + living_reward + death_reward + finish_reward + velocity_reward

            buffer.add(state, action_idx, reward, next_state, done)
            
            # Time training step
            if len(buffer) >= config.batch_size:
                training_start = time.time()
                policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = agent.learn(steps, buffer.sample(), gamma=0.99)
                training_time_total += time.time() - training_start
            else:
                # Initialize with dummy values when buffer is too small
                policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = 0.0, 0.0, 0.0, 0.0, agent.alpha.item() if hasattr(agent.alpha, 'item') else agent.alpha
            obs = next_obs
            state = next_state
            rewards += reward
            episode_steps += 1
            if done:
                break

        average10.append(rewards)
        total_steps += episode_steps
        
        # Calculate episode timing
        episode_time = time.time() - episode_start_time
        episode_times.append(episode_time)
        training_times.append(training_time_total)
        env_step_times.append(env_step_time_total)
        
        # Calculate averages
        avg_episode_time = np.mean(episode_times[-10:]) if len(episode_times) > 0 else episode_time
        avg_training_time = np.mean(training_times[-10:]) if len(training_times) > 0 else training_time_total
        avg_env_time = np.mean(env_step_times[-10:]) if len(env_step_times) > 0 else env_step_time_total
        
        print("Episode: {} | Reward: {:.2f} | Policy Loss: {:.4f} | Steps: {} | Time: {:.2f}s (env: {:.2f}s, train: {:.2f}s)".format(
            i, rewards, policy_loss, steps, episode_time, env_step_time_total, training_time_total))
        
        if i % 10 == 0:
            print("  [Last 10 episodes avg] Episode time: {:.2f}s | Env step time: {:.2f}s | Training time: {:.2f}s".format(
                avg_episode_time, avg_env_time, avg_training_time))
        
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
        
        # Update hparams metric with final average reward (for HPARAMS tab)
        if i == config.episodes:
            try:
                # Update the hparam metric with final average reward
                writer.add_scalar("hparam/final_avg_reward", np.mean(average10), i)
            except:
                pass

        if (i % 10 == 0) and config.log_video:
            # RecordVideo automatically closes and saves video when episode ends
            # But we need to ensure the file is fully written before reading it
            
            # Wait for video file to be written (check file size stability)
            video_written = False
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
            
            if video_written:
                # Additional wait to ensure video is fully flushed
                time.sleep(1.0)
                log_video(writer, i, video_dir=video_dir)


        
        if i % config.save_every == 0:
            number_of_models = save_model_number(config, save_name=f"SAC_discrete_{i}")
            print(f"Saving all networks (actor, critic1, critic2) for episode {i}...")
            base_path = save_agent(config, save_name=f"SAC_discrete_{number_of_models}", agent=agent, ep=i)
            print(f"✓ Saved: {os.path.basename(base_path)}.pth, {os.path.basename(base_path)}_critic1.pth, {os.path.basename(base_path)}_critic2.pth")
    
    # Save final model at the end of training
    final_number = save_model_number(config, save_name="SAC_discrete_final")
    print(f"\nSaving final model (all networks) at episode {config.episodes}...")
    final_base_path = save_agent(config, save_name=f"SAC_discrete_{final_number}_", agent=agent, ep="final")
    print(f"✓ Final model saved: {os.path.basename(final_base_path)}.pth, {os.path.basename(final_base_path)}_critic1.pth, {os.path.basename(final_base_path)}_critic2.pth\n")
    
    # Close video recorder if it exists
    if config.log_video and hasattr(env, 'close_video_recorder'):
        env.close_video_recorder()
    
    # Print final performance statistics
    if len(episode_times) > 0:
        print("\n" + "=" * 60)
        print("Performance Statistics:")
        print(f"  Total episodes: {len(episode_times)}")
        print(f"  Average episode time: {np.mean(episode_times):.2f}s ± {np.std(episode_times):.2f}s")
        print(f"  Average environment step time: {np.mean(env_step_times):.2f}s ± {np.std(env_step_times):.2f}s")
        print(f"  Average training step time: {np.mean(training_times):.2f}s ± {np.std(training_times):.2f}s")
        print(f"  Total training time: {sum(training_times):.2f}s ({sum(training_times)/60:.2f} minutes)")
        print(f"  Total environment time: {sum(env_step_times):.2f}s ({sum(env_step_times)/60:.2f} minutes)")
        print("=" * 60 + "\n")
    
    writer.close()
    env.close()

if __name__ == "__main__":
    config = get_config()
    train(config)
