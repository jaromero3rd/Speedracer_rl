import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque, OrderedDict
import gymnasium as gym
from tqdm import tqdm
import os

def flatten_racecar_obs(obs):
    """Flatten racecar dictionary observation to vector."""
    if isinstance(obs, dict):
        parts = []
        for key in ['velocity', 'acceleration', 'rgb_camera']:
            if key in obs:
                parts.append(obs[key].flatten())
        return np.concatenate(parts) if parts else np.array([])
    return obs.flatten() if isinstance(obs, np.ndarray) else obs

def map_racecar_action(action_idx):
    """Map discrete action (0-5) to racecar dictionary action."""
    actions = [
        {"motor": 1, "steering": 0},
        {"motor": 1, "steering": 1},
        {"motor": 1, "steering": -1},
        {"motor": 0, "steering": 0},
        {"motor": 0, "steering": 1},
        {"motor": 0, "steering": -1},
    ]
    return OrderedDict(actions[action_idx])

def save(args, save_name, model, ep=None):
    save_dir = './trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")

def save_agent(args, save_name, agent, ep=None):
    """Save all agent networks (actor, critic1, critic2) to disk.
    
    Args:
        args: Config object with run_name attribute
        save_name: Base name for saved files
        agent: SAC agent instance containing all networks
        ep: Episode number or identifier (optional)
    
    Returns:
        str: Base path of saved files (without extension)
    """
    save_dir = './trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create base filename
    if ep is not None:
        base_name = save_dir + args.run_name + save_name + str(ep)
    else:
        base_name = save_dir + args.run_name + save_name
    
    # Save actor network
    actor_path = base_name + ".pth"
    torch.save(agent.actor_local.state_dict(), actor_path)
    
    # Save critic networks
    critic1_path = base_name + "_critic1.pth"
    critic2_path = base_name + "_critic2.pth"
    torch.save(agent.critic1.state_dict(), critic1_path)
    torch.save(agent.critic2.state_dict(), critic2_path)
    
    return base_name

def load_model(agent, model_path, device):
    """Load a pretrained model into the agent.
    
    Args:
        agent: SAC agent instance to load weights into
        model_path: Path to the saved model file (.pth)
        device: Torch device to load the model on
    
    Returns:
        bool: True if loading was successful, False otherwise
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading pretrained model from: {model_path}")
        agent.actor_local.load_state_dict(torch.load(model_path, map_location=device))
        agent.actor_local.train()  # Set back to training mode
        print(f"✓ Pretrained actor model loaded successfully")
        
        # Optionally load critic networks if they exist (for full checkpoint resume)
        critic1_path = model_path.replace('.pth', '_critic1.pth')
        critic2_path = model_path.replace('.pth', '_critic2.pth')
        if os.path.exists(critic1_path) and os.path.exists(critic2_path):
            print(f"Loading critic networks...")
            agent.critic1.load_state_dict(torch.load(critic1_path, map_location=device))
            agent.critic1_target.load_state_dict(torch.load(critic1_path, map_location=device))
            agent.critic2.load_state_dict(torch.load(critic2_path, map_location=device))
            agent.critic2_target.load_state_dict(torch.load(critic2_path, map_location=device))
            print(f"✓ Critic networks loaded successfully")
        else:
            print(f"Note: Critic networks not found. Only actor loaded.")
        
        print()
        return True
    except Exception as e:
        print(f"✗ Error loading pretrained model: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing with randomly initialized weights...\n")
        return False

def collect_random(env, dataset, num_samples=200, obs_buffer_max_len=4):
    """Collect random samples for buffer initialization.
    
    Args:
        env: Gym environment
        dataset: ReplayBuffer to add samples to
        num_samples: Number of samples to collect
        obs_buffer_max_len: Length of observation buffer (default: 4)
    """
    # Check if this is a racecar environment
    is_racecar = isinstance(env.observation_space, gym.spaces.Dict)
    
    obs, info = env.reset()
    obs_buffer = deque(maxlen=obs_buffer_max_len)
    
    # Flatten racecar observations if needed
    flat_obs = flatten_racecar_obs(obs)
    
    # Initialize buffer with first observation
    for _ in range(obs_buffer_max_len):
        obs_buffer.append(flat_obs)
    state = np.stack(obs_buffer, axis=0).flatten(order="C")
    
    for _ in tqdm(range(num_samples), desc="Collecting random samples", unit="sample"):
        if is_racecar:
            # Sample discrete action (0-5) and map to racecar action
            action_idx = np.random.randint(0, 6)
            action = map_racecar_action(action_idx)
        else:
            action = env.action_space.sample()
            action_idx = action
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Flatten racecar observations if needed
        flat_next_obs = flatten_racecar_obs(next_obs)
        
        # Update observation buffer and create next state
        obs_buffer.append(flat_next_obs)
        next_state = np.stack(obs_buffer, axis=0).flatten(order="C")
        
        dataset.add(state, action_idx, reward, next_state, done)
        
        obs = next_obs
        flat_obs = flat_next_obs
        state = next_state
        
        if done:
            obs, info = env.reset()
            # Reinitialize buffer
            obs_buffer.clear()
            flat_obs = flatten_racecar_obs(obs)
            for _ in range(obs_buffer_max_len):
                obs_buffer.append(flat_obs)
            state = np.stack(obs_buffer, axis=0).flatten(order="C")


class RNDTargetNetwork(nn.Module):
    """Target Network for Random Network Distillation (RND).
    
    Fixed random network that outputs features for states.
    This network is never trained - it provides a fixed target.
    """
    def __init__(self, state_size, feature_size=128, hidden_size=128):
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
    def __init__(self, state_size, feature_size=128, hidden_size=128):
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
    
    # Check if this is a racecar environment
    is_racecar = isinstance(env.observation_space, gym.spaces.Dict)
    
    obs, info = env.reset()
    obs_buffer = deque(maxlen=obs_buffer_max_len)
    
    # Flatten racecar observations if needed
    flat_obs = flatten_racecar_obs(obs)
    
    # Initialize buffer with first observation
    for _ in range(obs_buffer_max_len):
        obs_buffer.append(flat_obs)
    state = np.stack(list(obs_buffer), axis=0).flatten(order="C")
    
    # Calculate state and action sizes
    state_size = len(state)
    action_size = 6 if is_racecar else env.action_space.n
    
    # Create RND networks: target (fixed) and predictor (trainable)
    target_network = RNDTargetNetwork(state_size, feature_size=128, hidden_size=128).to(device)
    predictor_network = RNDPredictorNetwork(state_size, feature_size=128, hidden_size=128).to(device)
    
    # Create optimizer only for predictor network
    optimizer = optim.Adam(predictor_network.parameters(), lr=learning_rate)
    
    # Internal buffer for training predictor network
    rnd_buffer = deque(maxlen=min(1000, num_samples))
    
    for step in tqdm(range(num_samples), desc="Collecting samples with RND", unit="sample"):
        # Compute novelty/intrinsic reward using RND
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            target_features = target_network(state_tensor)
            predicted_features = predictor_network(state_tensor)
            # Prediction error = novelty measure
            prediction_error = F.mse_loss(predicted_features, target_features, reduction='none').mean(dim=1).item()
            is_novel = prediction_error > novelty_threshold
        
        # Select action: bias towards novel states
        if is_novel and np.random.random() < 0.4:
            # When state is novel, prefer random exploration (40% chance)
            action_idx = np.random.randint(0, action_size)
        else:
            # Otherwise, use random action
            action_idx = np.random.randint(0, action_size)
        
        # Map action to environment format
        if is_racecar:
            action = map_racecar_action(action_idx)
        else:
            action = action_idx
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Flatten racecar observations if needed
        flat_next_obs = flatten_racecar_obs(next_obs)
        
        # Update observation buffer and create next state
        obs_buffer.append(flat_next_obs)
        next_state = np.stack(list(obs_buffer), axis=0).flatten(order="C")
        
        # Add to main dataset (with optional exploration bonus)
        # Note: We add the original reward, but could add exploration bonus here
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
        flat_obs = flat_next_obs
        state = next_state
        
        if done:
            obs, info = env.reset()
            # Reinitialize buffer
            obs_buffer.clear()
            flat_obs = flatten_racecar_obs(obs)
            for _ in range(obs_buffer_max_len):
                obs_buffer.append(flat_obs)
            state = np.stack(list(obs_buffer), axis=0).flatten(order="C")
    
def save_model_number(args, save_name):
    """Get the number of existing model files in the trained_models directory.
    
    Args:
        args: Config object (unused, kept for API consistency)
        save_name: Base name for saved files (unused, kept for API consistency)
    
    Returns:
        int: Number of files in the trained_models directory
    """
    save_dir = './trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    number_of_models = len(os.listdir(save_dir))
    return number_of_models

