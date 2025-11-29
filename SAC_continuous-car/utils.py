"""
Utility functions for SAC Continuous.

Imports shared utilities from SAC_discrete-main and adds
continuous action space specific helpers.
"""

import torch
import numpy as np
from collections import deque
import os
import sys

# Import shared utilities from discrete SAC
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SAC_discrete-main'))
from utils import save as save_discrete


def save(args, save_name, model, ep=None, save_dir='./trained_models/'):
    """
    Save model checkpoint.
    
    Args:
        args: Config/args object with run_name attribute
        save_name: Base name for the saved model
        model: Model to save (can be actor or full agent)
        ep: Episode number (optional)
        save_dir: Directory to save models
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filename = args.run_name + save_name
    if ep is not None:
        filename += str(ep)
    filename += ".pth"
    
    filepath = os.path.join(save_dir, filename)
    
    # Handle both nn.Module (actor) and full agent with save method
    if hasattr(model, 'save'):
        model.save(filepath)
    else:
        torch.save(model.state_dict(), filepath)


def collect_random(env, dataset, num_samples=10000, obs_buffer_max_len=1):
    """
    Collect random samples for buffer initialization with continuous actions.
    
    Args:
        env: Gym environment with continuous action space
        dataset: ReplayBuffer to add samples to
        num_samples: Number of samples to collect
        obs_buffer_max_len: Length of observation buffer (default: 1 for continuous)
    """
    obs, info = env.reset()
    obs_buffer = deque(maxlen=obs_buffer_max_len)
    
    # Initialize buffer with first observation
    for _ in range(obs_buffer_max_len):
        obs_buffer.append(obs)
    
    if obs_buffer_max_len > 1:
        state = np.stack(obs_buffer, axis=0).flatten(order="C")
    else:
        state = obs.flatten() if hasattr(obs, 'flatten') else np.array(obs).flatten()
    
    for _ in range(num_samples):
        # Sample continuous action from action space
        action = env.action_space.sample()
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update observation buffer and create next state
        obs_buffer.append(next_obs)
        if obs_buffer_max_len > 1:
            next_state = np.stack(obs_buffer, axis=0).flatten(order="C")
        else:
            next_state = next_obs.flatten() if hasattr(next_obs, 'flatten') else np.array(next_obs).flatten()
        
        # Add experience to buffer
        dataset.add(state, action, reward, next_state, done)
        
        obs = next_obs
        state = next_state
        
        if done:
            obs, info = env.reset()
            obs_buffer.clear()
            for _ in range(obs_buffer_max_len):
                obs_buffer.append(obs)
            if obs_buffer_max_len > 1:
                state = np.stack(obs_buffer, axis=0).flatten(order="C")
            else:
                state = obs.flatten() if hasattr(obs, 'flatten') else np.array(obs).flatten()


def normalize_action(action, action_space):
    """
    Normalize action to [-1, 1] range.
    
    Args:
        action: Action in original space
        action_space: Gym Box action space
        
    Returns:
        Normalized action in [-1, 1]
    """
    low = action_space.low
    high = action_space.high
    return 2.0 * (action - low) / (high - low) - 1.0


def denormalize_action(action, action_space):
    """
    Denormalize action from [-1, 1] to original space.
    
    Args:
        action: Action in [-1, 1] range
        action_space: Gym Box action space
        
    Returns:
        Action in original space
    """
    low = action_space.low
    high = action_space.high
    return low + (action + 1.0) * 0.5 * (high - low)


def get_action_space_info(action_space):
    """
    Extract useful information from a Gym Box action space.
    
    Args:
        action_space: Gym Box action space
        
    Returns:
        dict with action_size, low, high, scale, bias
    """
    return {
        'action_size': action_space.shape[0],
        'low': action_space.low,
        'high': action_space.high,
        'scale': (action_space.high - action_space.low) / 2.0,
        'bias': (action_space.high + action_space.low) / 2.0
    }

