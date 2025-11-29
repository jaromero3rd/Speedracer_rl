"""
SAC Continuous - Soft Actor-Critic for Continuous Action Spaces

This module provides a SAC implementation designed for gymnasium environments
with continuous (Box) action spaces, adapted from the discrete SAC implementation.

Key Components:
    - SACContinuous: Main agent class for continuous control
    - ContinuousActor: Gaussian policy network with tanh squashing
    - ContinuousCritic: Q-function that takes (state, action) as input
    - TwinContinuousCritic: Twin Q-networks for reducing overestimation bias

Usage:
    from SAC_continuous_car import SACContinuous
    
    agent = SACContinuous(
        state_size=obs_dim,
        action_size=action_dim,
        device=torch.device('cuda'),
        action_space=env.action_space
    )
    
    # Get action
    action = agent.get_action(state, training=True)
    
    # Update
    losses = agent.learn(step, experiences)
"""

from .agent import SACContinuous
from .networks import ContinuousActor, ContinuousCritic, TwinContinuousCritic
from .utils import save, collect_random, normalize_action, denormalize_action, get_action_space_info

__all__ = [
    'SACContinuous',
    'ContinuousActor', 
    'ContinuousCritic',
    'TwinContinuousCritic',
    'save',
    'collect_random',
    'normalize_action',
    'denormalize_action',
    'get_action_space_info'
]

__version__ = '1.0.0'

