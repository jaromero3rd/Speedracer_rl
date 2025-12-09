import torch
import numpy as np
from collections import deque

def save(args, save_name, model, ep=None):
    import os
    save_dir = './trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")

def collect_random(env, dataset, num_samples=200, obs_buffer_max_len=4):
    """Collect random samples for buffer initialization.
    
    Args:
        env: Gym environment
        dataset: ReplayBuffer to add samples to
        num_samples: Number of samples to collect
        obs_buffer_max_len: Length of observation buffer (default: 4)
    """
    
    obs, info = env.reset()
    obs_buffer = deque(maxlen=obs_buffer_max_len)
    
    # Initialize buffer with first observation
    for _ in range(obs_buffer_max_len):
        obs_buffer.append(obs)
    state = np.stack(obs_buffer, axis=0).flatten(order="C")
    
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update observation buffer and create next state
        obs_buffer.append(next_obs)
        next_state = np.stack(obs_buffer, axis=0).flatten(order="C")
        
        dataset.add(state, action, reward, next_state, done)
        
        obs = next_obs
        state = next_state
        
        if done:
            obs, info = env.reset()
            # Reinitialize buffer
            obs_buffer.clear()
            for _ in range(obs_buffer_max_len):
                obs_buffer.append(obs)
            state = np.stack(obs_buffer, axis=0).flatten(order="C")
