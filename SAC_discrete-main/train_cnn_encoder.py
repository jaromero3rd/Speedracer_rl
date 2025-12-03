#!/usr/bin/env python3
"""
Train CNN encoder to predict CartPole state from frame stacks.
Uses ResNet18 architecture trained from scratch.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
import argparse
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt


class CartPoleVisionDataset(Dataset):
    """Dataset for CartPole vision data."""

    def __init__(self, data_path, normalize_states=True):
        """
        Args:
            data_path: Path to dataset.npz file
            normalize_states: Whether to normalize state values
        """
        # Load data
        data = np.load(data_path)
        self.frame_stacks = data['frame_stacks']  # (N, 4, H, W, 3)
        self.states = data['states']  # (N, 4)

        # Load metadata and stats
        data_dir = os.path.dirname(data_path)
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)

        with open(os.path.join(data_dir, 'state_stats.json'), 'r') as f:
            self.state_stats = json.load(f)

        self.normalize_states = normalize_states

        print(f"Loaded dataset: {self.frame_stacks.shape[0]} samples")
        print(f"  Frame stacks shape: {self.frame_stacks.shape}")
        print(f"  States shape: {self.states.shape}")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # Get frame stack: (4, H, W, 3)
        frame_stack = self.frame_stacks[idx]

        # Reshape to (H, W, 12) by concatenating channels
        # (4, H, W, 3) -> (H, W, 12)
        h, w = frame_stack.shape[1], frame_stack.shape[2]
        frame_stack_concat = frame_stack.transpose(1, 2, 0, 3).reshape(h, w, -1)

        # Convert to torch tensor and normalize to [0, 1]
        # (H, W, 12) -> (12, H, W)
        frame_tensor = torch.from_numpy(frame_stack_concat).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)  # (12, H, W)

        # Get state
        state = torch.from_numpy(self.states[idx]).float()

        # Normalize states if requested
        if self.normalize_states:
            mean = torch.tensor(self.state_stats['mean'], dtype=torch.float32)
            std = torch.tensor(self.state_stats['std'], dtype=torch.float32)
            state = (state - mean) / (std + 1e-8)

        return frame_tensor, state


class ResNet18Encoder(nn.Module):
    """ResNet18 encoder for CartPole state prediction."""

    def __init__(self, input_channels=48, output_dim=4, pretrained=False):
        """
        Args:
            input_channels: Number of input channels (48 for 16 stacked RGB frames, 12 for 4 frames)
            output_dim: Output dimension (4 for CartPole state)
            pretrained: Whether to use pretrained weights (not applicable for >3 channels)
        """
        super(ResNet18Encoder, self).__init__()

        # Load ResNet18
        if pretrained and input_channels == 3:
            self.resnet = models.resnet18(pretrained=True)
        else:
            self.resnet = models.resnet18(pretrained=False)

        # Modify first conv layer for 12 input channels
        if input_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                input_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )

        # Get the feature dimension before final FC layer
        self.feature_dim = self.resnet.fc.in_features

        # Replace final FC layer
        self.resnet.fc = nn.Linear(self.feature_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) where C is input_channels
        Returns:
            state: (B, 4) - [x, x_dot, theta, theta_dot]
        """
        return self.resnet(x)

    def get_features(self, x):
        """
        Get intermediate features (embeddings) before final FC layer.
        Useful for RL agent input.

        Args:
            x: (B, C, H, W) where C is input_channels
        Returns:
            features: (B, 512)
        """
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for frames, states in tqdm(dataloader, desc="Training", leave=False):
        frames = frames.to(device)
        states = states.to(device)

        # Forward pass
        predictions = model(frames)
        loss = criterion(predictions, states)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, denormalize_fn=None):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for frames, states in tqdm(dataloader, desc="Validating", leave=False):
            frames = frames.to(device)
            states = states.to(device)

            predictions = model(frames)
            loss = criterion(predictions, states)

            total_loss += loss.item()

            all_predictions.append(predictions.cpu())
            all_targets.append(states.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute per-dimension MAE
    mae_per_dim = torch.abs(all_predictions - all_targets).mean(dim=0)

    return total_loss / len(dataloader), mae_per_dim.numpy()


def main():
    parser = argparse.ArgumentParser(description='Train CNN encoder for CartPole state prediction')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing dataset.npz files')
    parser.add_argument('--output_dir', type=str, default='trained_cnn',
                        help='Directory to save trained model')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize state values')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load datasets from all agents
    print("Loading datasets...")
    all_datasets = []
    for subdir in os.listdir(args.data_dir):
        dataset_path = os.path.join(args.data_dir, subdir, 'dataset.npz')
        if os.path.exists(dataset_path):
            print(f"Loading {subdir}...")
            dataset = CartPoleVisionDataset(dataset_path, normalize_states=args.normalize)
            all_datasets.append(dataset)

    # Concatenate all datasets
    print(f"\nCombining {len(all_datasets)} datasets...")
    combined_dataset = torch.utils.data.ConcatDataset(all_datasets)
    print(f"Total samples: {len(combined_dataset)}\n")

    # Split into train/val
    val_size = int(args.val_split * len(combined_dataset))
    train_size = len(combined_dataset) - val_size
    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Determine input channels from first dataset
    sample_frames, _ = all_datasets[0][0]
    input_channels = sample_frames.shape[0]
    print(f"Detected input channels: {input_channels}")

    # Create model
    print("Creating model...")
    model = ResNet18Encoder(input_channels=input_channels, output_dim=4, pretrained=False)
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    print("Starting training...\n")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    state_labels = ['x', 'x_dot', 'theta', 'theta_dot']

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("-" * 60)

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Validate
        val_loss, mae_per_dim = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Learning Rate: {current_lr:.6f}")
        print("MAE per dimension:")
        for label, mae in zip(state_labels, mae_per_dim):
            print(f"  {label}: {mae:.6f}")
        print()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'mae_per_dim': mae_per_dim.tolist()
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"✓ Saved best model (val_loss: {val_loss:.6f})\n")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, os.path.join(args.output_dir, f'checkpoint_epoch{epoch+1}.pth'))

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1]
    }, os.path.join(args.output_dir, 'final_model.pth'))

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training History')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'training_history.png'))
    print(f"✓ Training complete! Best val loss: {best_val_loss:.6f}")
    print(f"✓ Models saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
