"""
Soft Actor-Critic (SAC) Agent for Continuous Action Spaces.

This module provides a SAC agent adapted for continuous action spaces,
using Gaussian policies with tanh squashing. It inherits concepts from
the discrete SAC implementation but with key modifications for continuous control.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import copy
import numpy as np

from networks import ContinuousActor, ContinuousCritic


class SACContinuous(nn.Module):
    """
    SAC Agent for continuous action spaces.
    
    Key differences from discrete SAC:
    - Actor outputs Gaussian distribution parameters (mean, log_std)
    - Critic takes (state, action) as input instead of just state
    - Policy loss uses reparameterization trick
    - Target entropy is -dim(action_space)
    """
    
    def __init__(self,
                 state_size,
                 action_size,
                 device,
                 action_space=None,
                 learning_rate=3e-4,
                 entropy_bonus=None,
                 gamma=0.99,
                 tau=5e-3,
                 hidden_size=256):
        """
        Initialize the SAC agent for continuous action spaces.
        
        Args:
            state_size (int): Dimension of state space
            action_size (int): Dimension of action space
            device: torch device
            action_space: Gym action space (Box) for scaling actions
            learning_rate (float): Learning rate for all optimizers
            entropy_bonus (float): Fixed entropy bonus (alpha). If None, uses learnable alpha
            gamma (float): Discount factor
            tau (float): Soft update coefficient for target networks
            hidden_size (int): Number of hidden units in networks
        """
        super(SACContinuous, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.clip_grad_param = 1.0
        
        # Determine action scaling from action space
        if action_space is not None:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            ).to(device)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            ).to(device)
            action_scale_np = float(np.mean((action_space.high - action_space.low) / 2.0))
            action_bias_np = float(np.mean((action_space.high + action_space.low) / 2.0))
        else:
            self.action_scale = torch.ones(action_size).to(device)
            self.action_bias = torch.zeros(action_size).to(device)
            action_scale_np = 1.0
            action_bias_np = 0.0
        
        # Target entropy: -dim(action_space)
        self.target_entropy = -action_size
        
        # Handle entropy bonus (alpha)
        if entropy_bonus is not None:
            # Use fixed entropy bonus
            self.log_alpha = None
            self.alpha = torch.tensor([entropy_bonus], device=device, requires_grad=False)
            self.alpha_optimizer = None
        else:
            # Use learnable entropy bonus
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().detach()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        
        # Actor Network (Gaussian Policy)
        self.actor = ContinuousActor(
            state_size, 
            action_size, 
            hidden_size,
            action_scale=action_scale_np,
            action_bias=action_bias_np
        ).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        
        # Twin Critic Networks
        self.critic1 = ContinuousCritic(state_size, action_size, hidden_size, seed=1).to(device)
        self.critic2 = ContinuousCritic(state_size, action_size, hidden_size, seed=2).to(device)
        
        # Target Critic Networks
        self.critic1_target = ContinuousCritic(state_size, action_size, hidden_size, seed=1).to(device)
        self.critic2_target = ContinuousCritic(state_size, action_size, hidden_size, seed=2).to(device)
        
        # Initialize target networks with same weights
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Critic optimizers
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
    
    def get_action(self, state, training=True):
        """
        Get action for given state.
        
        Args:
            state: State array
            training: If True, sample from policy. If False, use deterministic action.
            
        Returns:
            action: numpy array of action
        """
        # Ensure state is 1D and convert to tensor
        if isinstance(state, np.ndarray):
            state = state.flatten()
        state = torch.FloatTensor(state).to(self.device)
        
        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            if training:
                action, _, _ = self.actor.sample(state)
                action = action.cpu().numpy()
            else:
                action = self.actor.get_det_action(state)
        
        # Remove batch dimension
        if len(action.shape) > 1 and action.shape[0] == 1:
            action = action[0]
        
        return action
    
    def learn(self, step, experiences, gamma=None):
        """
        Update actor, critics and entropy alpha using given batch of experience tuples.
        
        For continuous SAC:
        Q_targets = r + γ * (min_critic_target(s', a') - α * log_pi(a'|s'))
        where a' ~ π(·|s')
        
        Args:
            step: Current training step
            experiences: Tuple of (states, actions, rewards, next_states, dones)
            gamma: Discount factor (uses self.gamma if None)
            
        Returns:
            actor_loss, alpha_loss, critic1_loss, critic2_loss, current_alpha
        """
        if gamma is None:
            gamma = self.gamma
            
        states, actions, rewards, next_states, dones = experiences
        
        # Ensure proper shapes
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)
        
        # ---------------------------- update critics ---------------------------- #
        with torch.no_grad():
            # Sample actions from current policy for next states
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            
            # Compute target Q-values
            q1_target = self.critic1_target(next_states, next_actions)
            q2_target = self.critic2_target(next_states, next_actions)
            min_q_target = torch.min(q1_target, q2_target)
            
            # Soft Bellman backup
            q_target = rewards + gamma * (1 - dones) * (min_q_target - self.alpha * next_log_probs)
        
        # Compute current Q-values
        q1_current = self.critic1(states, actions)
        q2_current = self.critic2(states, actions)
        
        # Critic losses
        critic1_loss = F.mse_loss(q1_current, q_target)
        critic2_loss = F.mse_loss(q2_current, q_target)
        
        # Update critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        
        # Update critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        # Sample actions from current policy
        actions_new, log_probs, _ = self.actor.sample(states)
        
        # Compute Q-values for new actions
        q1_new = self.critic1(states, actions_new)
        q2_new = self.critic2(states, actions_new)
        min_q_new = torch.min(q1_new, q2_new)
        
        # Actor loss: maximize Q - α * log_pi
        actor_loss = (self.alpha * log_probs - min_q_new).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), self.clip_grad_param)
        self.actor_optimizer.step()
        
        # ---------------------------- update alpha ---------------------------- #
        if self.log_alpha is not None:
            # Learnable alpha
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().detach()
            alpha_loss_value = alpha_loss.item()
        else:
            # Fixed alpha
            alpha_loss_value = 0.0
        
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        
        current_alpha = self.alpha.item() if self.alpha.numel() == 1 else self.alpha.mean().item()
        
        return actor_loss.item(), alpha_loss_value, critic1_loss.item(), critic2_loss.item(), current_alpha
    
    def soft_update(self, local_model, target_model):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Args:
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, filepath):
        """
        Save model checkpoints.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
        }
        
        if self.log_alpha is not None:
            checkpoint['log_alpha'] = self.log_alpha
            checkpoint['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        else:
            checkpoint['alpha'] = self.alpha
        
        torch.save(checkpoint, filepath)
    
    def load(self, filepath):
        """
        Load model checkpoints.
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        
        if 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp().detach()
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        elif 'alpha' in checkpoint:
            self.alpha = checkpoint['alpha']

