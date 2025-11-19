#!/usr/bin/env python3
"""
Soft Actor-Critic (SAC) Implementation for CartPole
Adapted for discrete action spaces using categorical policy.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Optional
from collections import deque
import gymnasium as gym
import random
from dataclasses import dataclass
import os
from datetime import datetime


@dataclass
class SACConfig:
    """Configuration for SAC agent"""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005  # Soft update coefficient
    alpha: float = 0.2  # Temperature parameter (entropy coefficient)
    alpha_lr: float = 3e-4  # Learning rate for alpha
    buffer_size: int = 100000
    batch_size: int = 256
    hidden_dims: list = None
    device: str = None
    target_update_freq: int = 1  # Update target network every N steps
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class ReplayBuffer:
    """Experience replay buffer for SAC"""
    
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)  # 1D array for discrete actions
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.LongTensor(self.actions[indices]).to(self.device)  # Shape: [batch_size]
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.BoolTensor(self.dones[indices]).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return self.size


class QNetwork(nn.Module):
    """Q-network (critic) for SAC"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list):
        super(QNetwork, self).__init__()
        
        self.input_dim = state_dim
        self.output_dim = action_dim

        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),  # Output Q-value for each action
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns Q-values for all actions"""
        return self.network(state)


class PolicyNetwork(nn.Module):
    """Policy network (actor) for SAC with discrete actions"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list):
        super(PolicyNetwork, self).__init__()
        
        self.input_dim = state_dim
        self.output_dim = action_dim

        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns action logits"""
        return self.network(state)
    
    def get_action_and_log_prob(self, state: torch.Tensor, 
                                 deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy distribution and compute log probability.
        
        Args:
            state: Current state
            deterministic: If True, return deterministic action (argmax), 
                          otherwise sample from the policy distribution
        
        Returns:
            action: Sampled action from the policy distribution
            log_prob: Log probability of the action
        """
        logits = self.forward(state)
        
        # Create categorical distribution from logits
        dist = torch.distributions.Categorical(logits=logits)
        
        if deterministic:
            # Use mode (most likely action) for deterministic policy
            action = torch.argmax(logits, dim=-1)
            # Compute log prob of the mode
            log_prob = dist.log_prob(action)
        else:
            # Sample from the policy distribution
            action = dist.sample()
            # Get log probability of the sampled action
            log_prob = dist.log_prob(action)
        
        return action, log_prob


class SACAgent:
    """Soft Actor-Critic agent for discrete action spaces"""
    
    def __init__(self, state_dim: int, action_dim: int, config: SACConfig = None):
        """
        Initialize SAC agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            config: SAC configuration
        """
        if config is None:
            config = SACConfig()
        
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(config.device)
        
        # Q-networks (two critics for stability)
        self.q1 = QNetwork(state_dim, action_dim, config.hidden_dims).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, config.hidden_dims).to(self.device)
        
        # Target Q-networks
        self.q1_target = QNetwork(state_dim, action_dim, config.hidden_dims).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim, config.hidden_dims).to(self.device)
        
        # Copy weights to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Policy network
        self.policy = PolicyNetwork(state_dim, action_dim, config.hidden_dims).to(self.device)
        
        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=config.learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=config.learning_rate)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        
        # Temperature (entropy coefficient) - can be learned or fixed
        self.alpha = config.alpha
        # Ensure alpha > 0 for log_alpha initialization (use small epsilon if alpha is 0)
        alpha_init = max(config.alpha, 1e-6)
        self.log_alpha = torch.tensor(np.log(alpha_init), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = -np.log(1.0 / action_dim) * 0.98  # Target entropy for discrete actions
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size, state_dim)
        self.replay_buffer.device = self.device
        
        # Training step counter
        self.step_count = 0
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action using current policy.
        
        Args:
            state: Current state
            deterministic: If True, use deterministic policy
        
        Returns:
            action: Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _ = self.policy.get_action_and_log_prob(state_tensor, deterministic)
        
        return action.item()
    
    def select_action_with_entropy(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        """
        Select action and compute entropy for reward bonus.
        
        Args:
            state: Current state
            deterministic: If True, use deterministic policy
        
        Returns:
            action: Selected action
            entropy: Entropy of the policy distribution
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.policy.forward(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action, log_prob = self.policy.get_action_and_log_prob(state_tensor, deterministic)
            entropy = dist.entropy().item()
        
        return action.item(), entropy
    
    def compute_critic_loss(self, states: torch.Tensor, actions: torch.Tensor, 
                           rewards: torch.Tensor, next_states: torch.Tensor, 
                           dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Compute critic (Q-network) losses.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
        
        Returns:
            q1_loss: Loss for Q1 network
            q2_loss: Loss for Q2 network
            q_stats: Dictionary with Q-value statistics
        """
        # Ensure actions are 1D
        if actions.dim() > 1:
            actions = actions.squeeze()
        
        # Compute target Q-values
        with torch.no_grad():
            # Get next actions and log probs from current policy
            next_actions, next_log_probs = self.policy.get_action_and_log_prob(next_states)
            
            # Ensure next_actions is 1D
            if next_actions.dim() > 1:
                next_actions = next_actions.squeeze()
            
            # Compute target Q-values using target networks
            q1_next = self.q1_target(next_states)
            q2_next = self.q2_target(next_states)
            
            # Take Q-values for the actions chosen by policy
            q1_next = q1_next.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            q2_next = q2_next.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Use minimum of two Q-networks (clipped double Q-learning)
            q_next = torch.min(q1_next, q2_next)
            
            # Compute target: r + gamma * (min(Q1, Q2) - alpha * log_prob)
            target_q = rewards + self.config.gamma * (1 - dones.float()) * (
                q_next - self.alpha * next_log_probs
            )
        
        # Current Q-values for taken actions
        q1_values = self.q1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_values = self.q2(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Q-network losses
        q1_loss = F.mse_loss(q1_values, target_q)
        q2_loss = F.mse_loss(q2_values, target_q)
        
        # Return losses and Q-value statistics for logging
        return q1_loss, q2_loss, {
            'q1_mean': q1_values.mean().item(),
            'q2_mean': q2_values.mean().item(),
            'target_q_mean': target_q.mean().item(),
            'q1_min': q1_values.min().item(),
            'q1_max': q1_values.max().item(),
            'target_q_min': target_q.min().item(),
            'target_q_max': target_q.max().item(),
            'batch_reward_mean': rewards.mean().item(),
            'batch_reward_min': rewards.min().item(),
            'batch_reward_max': rewards.max().item(),
        }
    
    def compute_policy_loss(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute policy (actor) loss.
        
        Args:
            states: Batch of states
        
        Returns:
            policy_loss: Loss for policy network
        """
        # Get actions and log probs from current policy
        new_actions, new_log_probs = self.policy.get_action_and_log_prob(states)
        
        # Ensure new_actions is 1D
        if new_actions.dim() > 1:
            new_actions = new_actions.squeeze()
        
        # Get Q-values for the new actions (detach to prevent gradients flowing through Q during policy update)
        # Actually, we DO want gradients through Q for policy update, so don't detach
        q1_new = self.q1(states).gather(1, new_actions.unsqueeze(1)).squeeze(1)
        q2_new = self.q2(states).gather(1, new_actions.unsqueeze(1)).squeeze(1)
        q_new = torch.min(q1_new, q2_new)
        
        # Policy loss: maximize (Q - alpha * log_prob) = minimize (alpha * log_prob - Q)
        # Note: We want to maximize Q, so we minimize (alpha * log_prob - Q)
        policy_loss = (self.alpha * new_log_probs - q_new).mean()
        
        return policy_loss
    
    def update(self, batch_size: int) -> dict:
        """
        Update SAC networks using a batch of experiences.
        
        Args:
            batch_size: Batch size for training
        
        Returns:
            Dictionary with loss values
        """
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Compute and update critic losses
        q1_loss, q2_loss, q_stats = self.compute_critic_loss(states, actions, rewards, next_states, dones)
        
        # Update Q-networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2_optimizer.step()
        
        # Compute and update policy loss
        policy_loss = self.compute_policy_loss(states)
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # Update temperature (alpha) - need log_probs for this
        with torch.no_grad():
            _, new_log_probs = self.policy.get_action_and_log_prob(states)
        
        alpha_loss = -(self.log_alpha * (new_log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Clamp alpha to prevent it from becoming too small or negative
        self.alpha = max(self.log_alpha.exp().item(), 1e-6)
        
        # Soft update target networks
        if self.step_count % self.config.target_update_freq == 0:
            self._soft_update(self.q1_target, self.q1, self.config.tau)
            self._soft_update(self.q2_target, self.q2, self.config.tau)
        
        self.step_count += 1
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha,
            **q_stats  # Include Q-value statistics
        }
    
    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def save(self, filepath: str):
        """Save agent state"""
        torch.save({
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha.item(),
            'step_count': self.step_count,
            'config': self.config
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.log_alpha = torch.tensor(checkpoint['log_alpha'], requires_grad=True, device=self.device)
        self.alpha = np.exp(checkpoint['log_alpha'])
        self.step_count = checkpoint['step_count']


def train_sac_cartpole(
    env_name: str = "CartPole-v1",
    num_episodes: int = 1000,
    max_steps_per_episode: int = 500,
    warmup_steps: int = 1000,
    update_freq: int = 1,
    eval_freq: int = 50,
    eval_episodes: int = 5,
    save_path: Optional[str] = None,
    log_dir: Optional[str] = None,
    config: Optional[SACConfig] = None
):
    """
    Train SAC agent on CartPole environment.
    
    Args:
        env_name: Name of the environment
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        warmup_steps: Number of random steps before training
        update_freq: Update frequency (update every N steps)
        eval_freq: Evaluation frequency (evaluate every N episodes)
        eval_episodes: Number of episodes for evaluation
        save_path: Path to save the trained model
        log_dir: Directory for TensorBoard logs (default: runs/sac_cartpole_TIMESTAMP)
        config: SAC configuration
    """
    # Create environment
    env = gym.make(env_name)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Environment: {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Device: {config.device if config else 'auto'}\n")
    
    # Setup TensorBoard
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/sac_cartpole_{timestamp}"
    
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"View with: tensorboard --logdir {log_dir}\n")
    
    # Create agent
    if config is None:
        config = SACConfig()
    
    agent = SACAgent(state_dim, action_dim, config)
    
    # Log hyperparameters
    writer.add_hparams(
        {
            "learning_rate": config.learning_rate,
            "gamma": config.gamma,
            "tau": config.tau,
            "alpha": config.alpha,
            "batch_size": config.batch_size,
            "buffer_size": config.buffer_size,
            "warmup_steps": warmup_steps,
            "update_freq": update_freq,
        },
        {}
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    
    # Training loop
    total_steps = 0
    
    print("Starting training...\n")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps_per_episode):
            # Select action
            if total_steps < warmup_steps:
                # Random action during warmup
                action = env.action_space.sample()
                entropy = 0.0  # No entropy bonus during warmup
            else:
                # Get action and entropy from policy
                action, entropy = agent.select_action_with_entropy(state, deterministic=False)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience with ORIGINAL reward (entropy is handled in Q-target calculation)
            # In SAC, entropy is accounted for in the target: r + gamma * (Q_next - alpha * log_prob)
            # We should NOT add entropy to the reward here, as it would double-count
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            # Log reward and entropy metrics
            if total_steps >= warmup_steps:
                entropy_bonus = agent.alpha * entropy  # For logging only
                writer.add_scalar("Reward/StepReward", reward, total_steps)
                writer.add_scalar("Reward/Entropy", entropy, total_steps)
                writer.add_scalar("Reward/EntropyBonus", entropy_bonus, total_steps)
            
            # Update agent
            if total_steps >= warmup_steps and total_steps % update_freq == 0:
                losses = agent.update(config.batch_size)
                
                # Log training losses and Q-value statistics to TensorBoard
                if losses:
                    writer.add_scalar("Loss/Q1", losses.get('q1_loss', 0), total_steps)
                    writer.add_scalar("Loss/Q2", losses.get('q2_loss', 0), total_steps)
                    writer.add_scalar("Loss/Policy", losses.get('policy_loss', 0), total_steps)
                    writer.add_scalar("Loss/Alpha", losses.get('alpha_loss', 0), total_steps)
                    writer.add_scalar("Hyperparameters/Alpha", losses.get('alpha', config.alpha), total_steps)
                    
                    # Log Q-value statistics
                    writer.add_scalar("QValues/Q1_Mean", losses.get('q1_mean', 0), total_steps)
                    writer.add_scalar("QValues/Q2_Mean", losses.get('q2_mean', 0), total_steps)
                    writer.add_scalar("QValues/TargetQ_Mean", losses.get('target_q_mean', 0), total_steps)
                    writer.add_scalar("QValues/Q1_Min", losses.get('q1_min', 0), total_steps)
                    writer.add_scalar("QValues/Q1_Max", losses.get('q1_max', 0), total_steps)
                    writer.add_scalar("QValues/TargetQ_Min", losses.get('target_q_min', 0), total_steps)
                    writer.add_scalar("QValues/TargetQ_Max", losses.get('target_q_max', 0), total_steps)
                    
                    # Log reward statistics from batch
                    writer.add_scalar("Reward/BatchReward_Mean", losses.get('batch_reward_mean', 0), total_steps)
                    writer.add_scalar("Reward/BatchReward_Min", losses.get('batch_reward_min', 0), total_steps)
                    writer.add_scalar("Reward/BatchReward_Max", losses.get('batch_reward_max', 0), total_steps)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Log episode metrics to TensorBoard
        writer.add_scalar("Episode/Reward", episode_reward, episode)
        writer.add_scalar("Episode/Length", episode_length, episode)
        writer.add_scalar("Episode/TotalSteps", total_steps, episode)
        
        # Log moving averages
        if len(episode_rewards) >= 5:
            avg_reward_5 = np.mean(episode_rewards[-5:])
            avg_length_5 = np.mean(episode_lengths[-5:])
            writer.add_scalar("Episode/AvgReward_5", avg_reward_5, episode)
            writer.add_scalar("Episode/AvgLength_5", avg_length_5, episode)
        
        if len(episode_rewards) >= 10:
            avg_reward_10 = np.mean(episode_rewards[-10:])
            avg_length_10 = np.mean(episode_lengths[-10:])
            writer.add_scalar("Episode/AvgReward_10", avg_reward_10, episode)
            writer.add_scalar("Episode/AvgLength_10", avg_length_10, episode)
        
        if len(episode_rewards) >= 100:
            avg_reward_100 = np.mean(episode_rewards[-100:])
            avg_length_100 = np.mean(episode_lengths[-100:])
            writer.add_scalar("Episode/AvgReward_100", avg_reward_100, episode)
            writer.add_scalar("Episode/AvgLength_100", avg_length_100, episode)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Total Steps: {total_steps}")
        
        # Evaluation (uses deterministic policy, so typically higher than training)
        if (episode + 1) % eval_freq == 0:
            eval_reward = evaluate_agent(agent, env, eval_episodes, max_steps_per_episode)
            eval_rewards.append(eval_reward)
            writer.add_scalar("Evaluation/AverageReward", eval_reward, episode)
            
            # Compare with training averages for context
            train_avg_5 = None
            train_avg_10 = None
            
            if len(episode_rewards) >= 5:
                train_avg_5 = np.mean(episode_rewards[-5:])
                writer.add_scalar("Comparison/Eval_vs_Train5", eval_reward - train_avg_5, episode)
            
            if len(episode_rewards) >= 10:
                train_avg_10 = np.mean(episode_rewards[-10:])
                writer.add_scalar("Comparison/Eval_vs_Train10", eval_reward - train_avg_10, episode)
            
            # Print comparison
            if train_avg_5 is not None and train_avg_10 is not None:
                print(f"  Evaluation (deterministic): {eval_reward:.2f} | "
                      f"Train avg (last 5): {train_avg_5:.2f} | "
                      f"Train avg (last 10): {train_avg_10:.2f}\n")
            elif train_avg_5 is not None:
                print(f"  Evaluation (deterministic): {eval_reward:.2f} | "
                      f"Train avg (last 5): {train_avg_5:.2f}\n")
            else:
                print(f"  Evaluation (deterministic): {eval_reward:.2f} average reward over {eval_episodes} episodes\n")
    
    env.close()
    
    # Save model
    if save_path:
        agent.save(save_path)
        print(f"\nModel saved to {save_path}")
    
    # Close TensorBoard writer
    writer.close()
    
    # Print final statistics
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Final average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Best evaluation reward: {max(eval_rewards) if eval_rewards else 'N/A':.2f}")
    print(f"TensorBoard logs saved to: {log_dir}")
    print(f"View with: tensorboard --logdir {log_dir}")
    print("="*80)
    
    return agent, episode_rewards, eval_rewards


def evaluate_agent(agent: SACAgent, env: gym.Env, num_episodes: int, 
                   max_steps: int = 500) -> float:
    """
    Evaluate agent performance.
    
    Args:
        agent: SAC agent
        env: Environment
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
    
    Returns:
        Average reward
    """
    total_rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for _ in range(max_steps):
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SAC on CartPole')
    parser.add_argument('--env', type=str, default='CartPole-v0',
                        help='Environment name (default: CartPole-v1)')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes (default: 1000)')
    parser.add_argument('--max-steps', type=int, default=200,
                        help='Maximum steps per episode (default: 500)')
    parser.add_argument('--warmup', type=int, default=1000,
                        help='Warmup steps before training (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--learning-rate', type=float, default=3e-5,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--alpha', type=float, default=0.00,
                        help='Initial temperature/entropy coefficient (default: 0.2)')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update coefficient (default: 0.005)')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save model (default: None)')
    parser.add_argument('--eval-freq', type=int, default=100,
                        help='Evaluation frequency in episodes (default: 50)')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory for TensorBoard logs (default: runs/sac_cartpole_TIMESTAMP)')
    
    args = parser.parse_args()
    
    # Create config
    config = SACConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        batch_size=args.batch_size
    )
    
    # Train
    train_sac_cartpole(
        env_name=args.env,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        warmup_steps=args.warmup,
        eval_freq=args.eval_freq,
        save_path=args.save,
        log_dir=args.log_dir,
        config=config
    )


if __name__ == "__main__":
    main()

