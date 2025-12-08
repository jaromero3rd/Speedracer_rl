import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from tqdm import tqdm

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
    
    for _ in tqdm(range(num_samples), desc="Collecting random samples", unit="sample"):
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


def collect_policy(env, dataset, agent, num_samples=200, obs_buffer_max_len=4):
    """Collect samples using the agent's policy with different environment initializations.
    
    Uses the initial actor network with epsilon=0 (deterministic policy) and resets
    the environment every time an episode ends to get diverse initial states.
    
    Args:
        env: Gym environment
        dataset: ReplayBuffer to add samples to
        agent: SAC agent instance (should have epsilon=0)
        num_samples: Number of samples to collect
        obs_buffer_max_len: Length of observation buffer (default: 4)
    """
    obs_buffer = deque(maxlen=obs_buffer_max_len)
    samples_collected = 0
    
    # Reset environment initially
    obs, info = env.reset()
    
    # Initialize observation buffer
    obs_buffer.clear()
    for _ in range(obs_buffer_max_len):
        obs_buffer.append(obs)
    state = np.stack(obs_buffer, axis=0).flatten(order="C")
    
    for _ in tqdm(range(num_samples), desc="Collecting policy samples", unit="sample"):
        # Get action from agent's policy (epsilon=0, deterministic)
        action = agent.get_action(state, training=False)
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update observation buffer and create next state
        obs_buffer.append(next_obs)
        next_state = np.stack(obs_buffer, axis=0).flatten(order="C")
        
        # Add to dataset
        dataset.add(state, action, reward, next_state, done)
        
        obs = next_obs
        state = next_state
        samples_collected += 1
        
        # Reset environment every time episode ends
        if done:
            obs, info = env.reset()
            # Reinitialize observation buffer for new episode
            obs_buffer.clear()
            for _ in range(obs_buffer_max_len):
                obs_buffer.append(obs)
            state = np.stack(obs_buffer, axis=0).flatten(order="C")


class RNDTargetNetwork(nn.Module):
    """Target Network for Random Network Distillation (RND).
    
    Fixed random network that outputs features for states.
    This network is never trained - it provides a fixed target.
    """
    def __init__(self, state_size, feature_size=8, hidden_size=128):
        """Initialize RND target network.
        
        Args:
            state_size (int): Dimension of state
            feature_size (int): Size of output features (default: 128)
            hidden_size (int): Hidden layer size (default: 128)
        """
        super(RNDTargetNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_size),
        )
        # Freeze the network - it should never be trained
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, state):
        """Forward pass through the target network.
        
        Args:
            state: State tensor of shape (batch, state_size)
        
        Returns:
            features: Feature vector, shape (batch, feature_size)
        """
        return self.network(state)


class RNDPredictorNetwork(nn.Module):
    """Predictor Network for Random Network Distillation (RND).
    
    Trainable network that tries to predict the target network's output.
    High prediction error indicates novel/unexplored states.
    """
    def __init__(self, state_size, feature_size=8, hidden_size=128):
        """Initialize RND predictor network.
        
        Args:
            state_size (int): Dimension of state
            feature_size (int): Size of target features (default: 128)
            hidden_size (int): Hidden layer size (default: 128)
        """
        super(RNDPredictorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_size),
        )
    
    def forward(self, state):
        """Forward pass through the predictor network.
        
        Args:
            state: State tensor of shape (batch, state_size)
        
        Returns:
            predicted_features: Predicted feature vector, shape (batch, feature_size)
        """
        return self.network(state)


def compute_rnd_intrinsic_reward(target_network, predictor_network, state, device):
    """Compute RND intrinsic reward for a given state.
    
    This is a shared utility function used by both collect_random_RND and agent.compute_rnd_intrinsic_reward.
    
    Args:
        target_network: RNDTargetNetwork instance
        predictor_network: RNDPredictorNetwork instance
        state: State (numpy array or torch tensor)
        device: Torch device
        
    Returns:
        intrinsic_reward: Scalar intrinsic reward (prediction error = novelty measure)
    """
    # Convert to tensor if needed
    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state).float().to(device)
    if state.dim() == 1:
        state = state.unsqueeze(0)
    
    with torch.no_grad():
        target_features = target_network(state)
        predicted_features = predictor_network(state)
        prediction_error = F.mse_loss(predicted_features, target_features, reduction='none').mean(dim=1)
        return prediction_error.mean().item()


def collect_random_RND(env, dataset, num_samples=200, obs_buffer_max_len=4, 
                       device=None, train_freq=32, learning_rate=1e-3, 
                       exploration_bonus_scale=1.0, novelty_threshold=0.1):
    """Collect random samples using RND (Random Network Distillation) exploration.
    
    Creates two lightweight networks:
    1. Target network (fixed, random): outputs features for states
    2. Predictor network (trainable): tries to predict target network's output
    
    High prediction error indicates novel/unexplored states, which guides exploration.
    
    Args:
        env: Gym environment
        dataset: ReplayBuffer to add samples to
        num_samples: Number of samples to collect
        obs_buffer_max_len: Length of observation buffer (default: 4)
        device: Torch device (default: 'cpu')
        train_freq: Frequency of network training updates (default: 32)
        learning_rate: Learning rate for predictor network (default: 1e-3)
        exploration_bonus_scale: Scale factor for exploration bonus (default: 1.0)
        novelty_threshold: Threshold for considering a state novel (default: 0.1)
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    obs, info = env.reset()
    obs_buffer = deque(maxlen=obs_buffer_max_len)
    
    # Initialize buffer with first observation
    for _ in range(obs_buffer_max_len):
        obs_buffer.append(obs)
    state = np.stack(list(obs_buffer), axis=0).flatten(order="C")
    
    # Calculate state and action sizes
    state_size = len(state)
    action_size = env.action_space.n
    
    # Create RND networks: target (fixed) and predictor (trainable)
    target_network = RNDTargetNetwork(state_size, feature_size=8, hidden_size=128).to(device)
    predictor_network = RNDPredictorNetwork(state_size, feature_size=8, hidden_size=128).to(device)
    
    # Create optimizer only for predictor network
    optimizer = optim.Adam(predictor_network.parameters(), lr=learning_rate)
    
    # Internal buffer for training predictor network
    rnd_buffer = deque(maxlen=min(1000, num_samples))
    
    for step in tqdm(range(num_samples), desc="Collecting samples with RND", unit="sample"):
        action_idx = np.random.randint(0, action_size)
        action = action_idx
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update observation buffer and create next state
        obs_buffer.append(next_obs)
        next_state = np.stack(list(obs_buffer), axis=0).flatten(order="C")
        
        # Compute intrinsic reward using RND (same function as in agent.py)
        intrinsic_reward = compute_rnd_intrinsic_reward(target_network, predictor_network, next_state, device)
        
        # Add intrinsic reward to the environment reward (same as in training)
        reward = reward + intrinsic_reward
        
        # Add to main dataset with intrinsic reward included
        dataset.add(state, action_idx, reward, next_state, done)
        
        # Add to RND training buffer
        rnd_buffer.append(state.copy())
        
        # Train predictor network periodically
        if len(rnd_buffer) >= train_freq and step % train_freq == 0:
            # Sample a batch from RND buffer
            batch_size = min(32, len(rnd_buffer))
            batch_indices = np.random.choice(len(rnd_buffer), batch_size, replace=False)
            batch_states = [rnd_buffer[i] for i in batch_indices]
            
            states = torch.from_numpy(np.stack(batch_states)).float().to(device)
            
            # Get target features (fixed, from target network)
            with torch.no_grad():
                target_features = target_network(states)
            
            # Predict features using predictor network
            predicted_features = predictor_network(states)
            
            # Train predictor to minimize prediction error
            optimizer.zero_grad()
            loss = F.mse_loss(predicted_features, target_features)
            loss.backward()
            optimizer.step()
        
        obs = next_obs
        state = next_state
        
        if done:
            obs, info = env.reset()
            # Reinitialize buffer
            obs_buffer.clear()
            for _ in range(obs_buffer_max_len):
                obs_buffer.append(obs)
            state = np.stack(list(obs_buffer), axis=0).flatten(order="C")
