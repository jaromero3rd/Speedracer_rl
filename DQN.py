import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class DQNetwork(nn.Module):
    """
    Deep Q-Network that processes:
    - Two integer features
    - One grayscale image of any size
    """
    def __init__(self, num_actions, image_size=64, hidden_size=256):
        super(DQNetwork, self).__init__()
        
        self.image_size = image_size
        
        # CNN for processing image (adaptive to any size)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate the output size of the CNN dynamically
        self.image_feature_size = self._get_conv_output_size(image_size)
        
        # MLP for processing integer features
        self.int_encoder = nn.Sequential(
            nn.Linear(3, 3),
            nn.ReLU(),
        )
        
        # Combined network
        combined_size = self.image_feature_size + 3
        self.fc = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )
    
    def _get_conv_output_size(self, image_size):
        """Calculate the output size of the convolutional layers"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, image_size, image_size)
            dummy_output = self.image_encoder(dummy_input)
            return dummy_output.shape[1]
    
    def forward(self, int_features, image):
        """
        Args:
            int_features: Tensor of shape (batch_size, 2) - two integer features
            image: Tensor of shape (batch_size, 1, H, W) - grayscale image of any size
        
        Returns:
            Q-values for each action: Tensor of shape (batch_size, num_actions)
        """
        # Process image
        image_features = self.image_encoder(image)
        
        # Process integer features
        int_encoded = self.int_encoder(int_features)
        
        # Concatenate features
        combined = torch.cat([image_features, int_encoded], dim=1)
        
        # Get Q-values
        q_values = self.fc(combined)
        
        return q_values


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        state: tuple of (int_features, image)
        int_features: numpy array of shape (2,)
        image: numpy array of shape (1, H, W)
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Separate int features and images
        int_features = np.array([s[0] for s in states])
        images = np.array([s[1] for s in states])
        
        next_int_features = np.array([s[0] for s in next_states])
        next_images = np.array([s[1] for s in next_states])
        
        return (
            (torch.FloatTensor(int_features), torch.FloatTensor(images)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            (torch.FloatTensor(next_int_features), torch.FloatTensor(next_images)),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent with epsilon-greedy exploration"""
    def __init__(
        self,
        num_actions,
        image_size=64,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=32,
        target_update_freq=1000,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.num_actions = num_actions
        self.image_size = image_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        
        # Q-Networks
        self.q_network = DQNetwork(num_actions, image_size=image_size).to(device)
        self.target_network = DQNetwork(num_actions, image_size=image_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # Training step counter
        self.steps = 0
    
    def select_action(self, state, eval_mode=False):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: tuple of (int_features, image)
                int_features: numpy array of shape (2,)
                image: numpy array of shape (1, H, W) where H and W match image_size
            eval_mode: if True, use greedy policy (no exploration)
        
        Returns:
            action: integer action
        """
        if eval_mode or random.random() > self.epsilon:
            # Greedy action
            with torch.no_grad():
                int_features = torch.FloatTensor(state[0]).unsqueeze(0).to(self.device)
                image = torch.FloatTensor(state[1]).unsqueeze(0).to(self.device)
                q_values = self.q_network(int_features, image)
                action = q_values.argmax(dim=1).item()
        else:
            # Random action
            action = random.randrange(self.num_actions)
        
        return action
    
    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Move to device
        int_features, images = states
        int_features = int_features.to(self.device)
        images = images.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        
        next_int_features, next_images = next_states
        next_int_features = next_int_features.to(self.device)
        next_images = next_images.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(int_features, images).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_int_features, next_images).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon,
        }, path)
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']


