import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model for Racecar Environment.
    
    Processes observations by:
    1. Splitting into image (rgb_camera) and velocity/acceleration
    2. Processing image through CNN
    3. Processing velocity/acceleration through MLP to match CNN latent dimension
    4. Concatenating and reducing to action space through MLP
    """

    def __init__(self, state_size, action_size, hidden_size=256, 
                 image_shape=(128, 128, 3), vel_acc_dim=12, cnn_latent_dim=256):
        """Initialize parameters and build model.
        
        Params
        ======
            state_size (int): Total flattened state dimension (image + vel/acc) * obs_buffer_max_len
            action_size (int): Dimension of each action
            hidden_size (int): Hidden layer size for MLP
            image_shape (tuple): Shape of single image (H, W, C)
            vel_acc_dim (int): Dimension of velocity + acceleration (6 + 6 = 12)
            cnn_latent_dim (int): Latent dimension for CNN output (and vel/acc MLP output)
        """
        super(Actor, self).__init__()
        self.state_size = state_size
        self.output_size = action_size
        self.image_shape = image_shape
        self.vel_acc_dim = vel_acc_dim
        self.cnn_latent_dim = cnn_latent_dim
        
        # Calculate dimensions per observation
        self.image_dim = image_shape[0] * image_shape[1] * image_shape[2]  
        self.obs_dim = self.image_dim + vel_acc_dim  
        
        # Infer obs_buffer_max_len from state_size
        self.obs_buffer_max_len = state_size // self.obs_dim
        # Verify the division is exact
        if state_size % self.obs_dim != 0:
            raise ValueError(f"state_size ({state_size}) must be divisible by obs_dim ({self.obs_dim}). "
                           f"Expected state_size to be {self.obs_dim * self.obs_buffer_max_len} for obs_buffer_max_len={self.obs_buffer_max_len}")
        
        # CNN for image processing
        # Input: (batch, obs_buffer_max_len, H, W, C) -> reshape to (batch * obs_buffer_max_len, C, H, W)
        # Calculate output size after convolutions
        # For 128x128 input:
        # Conv1: (128, 128) -> (32, 32) with kernel=8, stride=4, padding=2
        # Conv2: (32, 32) -> (16, 16) with kernel=4, stride=2, padding=1
        # Conv3: (16, 16) -> (16, 16) with kernel=3, stride=1, padding=1
        # Final size: 16 * 16 * 64 = 16384
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(image_shape[2], 32, kernel_size=8, stride=4, padding=2),  # (128, 128) -> (32, 32)
            nn.ReLU(),
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (32, 32) -> (16, 16)
            nn.ReLU(),
            # Third conv block
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # (16, 16) -> (16, 16)
            nn.ReLU(),
            # Flatten
            nn.Flatten(),
            # Calculate flattened size: 16 * 16 * 64 = 16384
            nn.Linear(16 * 16 * 64, cnn_latent_dim),
            nn.ReLU(),
        )
        
        # MLP for velocity/acceleration to match CNN latent dimension
        self.vel_acc_mlp = nn.Sequential(
            nn.Linear(vel_acc_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, cnn_latent_dim),
            nn.ReLU(),
        )
        
        # Combine CNN and vel/acc outputs, then reduce to action space
        # Input: cnn_latent_dim * obs_buffer_max_len + cnn_latent_dim * obs_buffer_max_len
        # = 2 * cnn_latent_dim * obs_buffer_max_len
        combined_dim = 2 * cnn_latent_dim * self.obs_buffer_max_len
        self.final_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Flattened state tensor of shape (batch, state_size)
                   Contains: [image_0, vel_acc_0, image_1, vel_acc_1, ...] for obs_buffer_max_len observations
        
        Returns:
            action_probs: Action probabilities of shape (batch, action_size)
        """
        batch_size = state.shape[0]
        
        # Reshape state: (batch, state_size) -> (batch, obs_buffer_max_len, obs_dim)
        state_reshaped = state.view(batch_size, self.obs_buffer_max_len, self.obs_dim)
        
        # Split into image and vel/acc parts
        images = state_reshaped[:, :, :self.image_dim]  # (batch, obs_buffer_max_len, image_dim)
        vel_acc = state_reshaped[:, :, self.image_dim:]  # (batch, obs_buffer_max_len, vel_acc_dim)
        
        # Reshape images: (batch, obs_buffer_max_len, image_dim) -> (batch, obs_buffer_max_len, H, W, C)
        images = images.view(batch_size, self.obs_buffer_max_len, 
                            self.image_shape[0], self.image_shape[1], self.image_shape[2])
        
        # Process each observation in the buffer
        image_features = []
        vel_acc_features = []
        
        for i in range(self.obs_buffer_max_len):
            # Process image: (batch, H, W, C) -> (batch, C, H, W) for CNN
            img = images[:, i].permute(0, 3, 1, 2)  # (batch, C, H, W)
            img_feat = self.cnn(img)  # (batch, cnn_latent_dim)
            image_features.append(img_feat)
            
            # Process velocity/acceleration
            vel_acc_feat = self.vel_acc_mlp(vel_acc[:, i])  # (batch, cnn_latent_dim)
            vel_acc_features.append(vel_acc_feat)
        
        # Concatenate all features: (batch, cnn_latent_dim * obs_buffer_max_len * 2)
        image_features = torch.cat(image_features, dim=1)  # (batch, cnn_latent_dim * obs_buffer_max_len)
        vel_acc_features = torch.cat(vel_acc_features, dim=1)  # (batch, cnn_latent_dim * obs_buffer_max_len)
        combined_features = torch.cat([image_features, vel_acc_features], dim=1)  # (batch, 2 * cnn_latent_dim * obs_buffer_max_len)
        
        # Final MLP to action space
        action_probs = self.final_mlp(combined_features)
        return action_probs
    
    def evaluate(self, state, epsilon=1e-6):
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities        
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities
    
    def get_det_action(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        return action.detach().cpu()


class Critic(nn.Module):
    """Critic (Value) Model for Racecar Environment.
    
    Same architecture as Actor but outputs Q-values for each action.
    """

    def __init__(self, state_size, action_size, hidden_size=256, seed=1,
                 image_shape=(128, 128, 3), vel_acc_dim=12, cnn_latent_dim=256):
        """Initialize parameters and build model.
        
        Params
        ======
            state_size (int): Total flattened state dimension
            action_size (int): Dimension of each action
            hidden_size (int): Hidden layer size for MLP
            seed (int): Random seed
            image_shape (tuple): Shape of single image (H, W, C)
            vel_acc_dim (int): Dimension of velocity + acceleration
            cnn_latent_dim (int): Latent dimension for CNN output
        """
        super(Critic, self).__init__()
        self.state_size = state_size
        self.image_shape = image_shape
        self.vel_acc_dim = vel_acc_dim
        self.cnn_latent_dim = cnn_latent_dim
        
        # Calculate dimensions per observation
        self.image_dim = image_shape[0] * image_shape[1] * image_shape[2]
        self.obs_dim = self.image_dim + vel_acc_dim
        
        # Infer obs_buffer_max_len from state_size
        self.obs_buffer_max_len = state_size // self.obs_dim
        # Verify the division is exact
        if state_size % self.obs_dim != 0:
            raise ValueError(f"state_size ({state_size}) must be divisible by obs_dim ({self.obs_dim}). "
                           f"Expected state_size to be {self.obs_dim * self.obs_buffer_max_len} for obs_buffer_max_len={self.obs_buffer_max_len}")
        
        # CNN for image processing (same as Actor)
        self.cnn = nn.Sequential(
            nn.Conv2d(image_shape[2], 32, kernel_size=8, stride=4, padding=2),  # (128, 128) -> (32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (32, 32) -> (16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # (16, 16) -> (16, 16)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 16 * 64, cnn_latent_dim),  # 16384 -> cnn_latent_dim
            nn.ReLU(),
        )
        
        # MLP for velocity/acceleration (same as Actor)
        self.vel_acc_mlp = nn.Sequential(
            nn.Linear(vel_acc_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, cnn_latent_dim),
            nn.ReLU(),
        )
        
        # Combine and output Q-values for each action
        combined_dim = 2 * cnn_latent_dim * self.obs_buffer_max_len
        self.network = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state):
        """Build a critic (value) network that maps state -> Q-values for each action.
        
        Args:
            state: Flattened state tensor of shape (batch, state_size)
        
        Returns:
            Q-values: Q-values for each action, shape (batch, action_size)
        """
        batch_size = state.shape[0]
        
        # Reshape state: (batch, state_size) -> (batch, obs_buffer_max_len, obs_dim)
        state_reshaped = state.view(batch_size, self.obs_buffer_max_len, self.obs_dim)
        
        # Split into image and vel/acc parts
        images = state_reshaped[:, :, :self.image_dim]
        vel_acc = state_reshaped[:, :, self.image_dim:]
        
        # Reshape images
        images = images.view(batch_size, self.obs_buffer_max_len, 
                            self.image_shape[0], self.image_shape[1], self.image_shape[2])
        
        # Process each observation in the buffer
        image_features = []
        vel_acc_features = []
        
        for i in range(self.obs_buffer_max_len):
            # Process image
            img = images[:, i].permute(0, 3, 1, 2)
            img_feat = self.cnn(img)
            image_features.append(img_feat)
            
            # Process velocity/acceleration
            vel_acc_feat = self.vel_acc_mlp(vel_acc[:, i])
            vel_acc_features.append(vel_acc_feat)
        
        # Concatenate all features
        image_features = torch.cat(image_features, dim=1)
        vel_acc_features = torch.cat(vel_acc_features, dim=1)
        combined_features = torch.cat([image_features, vel_acc_features], dim=1)
        
        # Output Q-values for each action
        return self.network(combined_features)
