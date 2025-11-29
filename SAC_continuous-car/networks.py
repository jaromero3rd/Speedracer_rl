"""
Continuous Action Space Networks for SAC.

This module provides Actor and Critic networks designed for continuous action spaces
using Gaussian policies with tanh squashing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

# Import hidden_init from discrete version for weight initialization
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SAC_discrete-main'))
from networks import hidden_init


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


class ContinuousActor(nn.Module):
    """
    Gaussian Policy Actor for continuous action spaces.
    
    Outputs mean and log_std for a Gaussian distribution.
    Actions are sampled and squashed through tanh to bound them to [-1, 1],
    then scaled to the action space bounds.
    """

    def __init__(self, state_size, action_size, hidden_size=256, action_scale=1.0, action_bias=0.0):
        """
        Initialize the Gaussian Actor.
        
        Args:
            state_size (int): Dimension of state space
            action_size (int): Dimension of action space
            hidden_size (int): Number of hidden units
            action_scale (float): Scale for action output (default: 1.0)
            action_bias (float): Bias for action output (default: 0.0)
        """
        super(ContinuousActor, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.action_scale = action_scale
        self.action_bias = action_bias
        
        # Shared feature extraction layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Separate heads for mean and log_std
        self.mean_linear = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using uniform distribution."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.mean_linear.weight.data.uniform_(-3e-3, 3e-3)
        self.mean_linear.bias.data.uniform_(-3e-3, 3e-3)
        self.log_std_linear.weight.data.uniform_(-3e-3, 3e-3)
        self.log_std_linear.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Forward pass returning mean and log_std.
        
        Args:
            state: State tensor
            
        Returns:
            mean: Mean of the Gaussian distribution
            log_std: Log standard deviation of the Gaussian distribution
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        
        return mean, log_std
    
    def sample(self, state):
        """
        Sample action from the policy using reparameterization trick.
        
        Args:
            state: State tensor
            
        Returns:
            action: Sampled action (squashed through tanh and scaled)
            log_prob: Log probability of the action
            mean: Mean action (deterministic)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Reparameterization trick: sample from N(0,1) and transform
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterized sample
        
        # Squash through tanh
        y_t = torch.tanh(x_t)
        
        # Scale to action space
        action = y_t * self.action_scale + self.action_bias
        
        # Compute log probability with correction for tanh squashing
        # log_prob = log_prob_gaussian - log(1 - tanh^2(x))
        log_prob = normal.log_prob(x_t)
        
        # Enforcing Action Bound - correction for tanh squashing
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Deterministic action (for evaluation)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean_action
    
    def get_action(self, state):
        """
        Get stochastic action for training.
        
        Args:
            state: State tensor
            
        Returns:
            action: numpy array of sampled action
        """
        action, _, _ = self.sample(state)
        return action.detach().cpu().numpy()
    
    def get_det_action(self, state):
        """
        Get deterministic action for evaluation.
        
        Args:
            state: State tensor
            
        Returns:
            action: numpy array of deterministic action (mean)
        """
        mean, _ = self.forward(state)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return mean_action.detach().cpu().numpy()


class ContinuousCritic(nn.Module):
    """
    Critic (Q-function) for continuous action spaces.
    
    Takes state AND action as input and outputs a single Q-value.
    This is different from discrete SAC where the critic outputs Q-values
    for all discrete actions.
    """

    def __init__(self, state_size, action_size, hidden_size=256, seed=1):
        """
        Initialize the Critic network.
        
        Args:
            state_size (int): Dimension of state space
            action_size (int): Dimension of action space
            hidden_size (int): Number of hidden units
            seed (int): Random seed
        """
        super(ContinuousCritic, self).__init__()
        torch.manual_seed(seed)
        
        # Q1 architecture
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using uniform distribution."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Forward pass to compute Q-value.
        
        Args:
            state: State tensor [batch_size, state_size]
            action: Action tensor [batch_size, action_size]
            
        Returns:
            Q-value tensor [batch_size, 1]
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        
        return q_value


class TwinContinuousCritic(nn.Module):
    """
    Twin Critic networks for SAC (to reduce overestimation bias).
    
    Contains two independent Q-networks that take state and action as input.
    The minimum of the two Q-values is used for the target computation.
    """

    def __init__(self, state_size, action_size, hidden_size=256, seed=1):
        """
        Initialize Twin Critic networks.
        
        Args:
            state_size (int): Dimension of state space
            action_size (int): Dimension of action space
            hidden_size (int): Number of hidden units
            seed (int): Random seed
        """
        super(TwinContinuousCritic, self).__init__()
        
        # Q1 architecture
        self.q1_fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.q1_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q1_fc3 = nn.Linear(hidden_size, 1)
        
        # Q2 architecture (independent weights)
        self.q2_fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.q2_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q2_fc3 = nn.Linear(hidden_size, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using uniform distribution."""
        for layer in [self.q1_fc1, self.q1_fc2, self.q2_fc1, self.q2_fc2]:
            layer.weight.data.uniform_(*hidden_init(layer))
        for layer in [self.q1_fc3, self.q2_fc3]:
            layer.weight.data.uniform_(-3e-3, 3e-3)
            layer.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Forward pass for both Q-networks.
        
        Args:
            state: State tensor [batch_size, state_size]
            action: Action tensor [batch_size, action_size]
            
        Returns:
            q1: Q-value from first network [batch_size, 1]
            q2: Q-value from second network [batch_size, 1]
        """
        x = torch.cat([state, action], dim=-1)
        
        # Q1 forward
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)
        
        # Q2 forward
        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)
        
        return q1, q2
    
    def q1_forward(self, state, action):
        """
        Forward pass for Q1 network only.
        Used for policy gradient computation.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            q1: Q-value from first network
        """
        x = torch.cat([state, action], dim=-1)
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)
        return q1

