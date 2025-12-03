import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import wandb

from vision_network import ResNetVAE


class ImageDataset(Dataset):
    """Dataset for loading stacked frame images."""

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Directory containing .npy or .pt files with images
            transform: Optional transform to apply to images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Find all data files
        self.data_files = list(self.data_dir.glob("*.npy")) + list(self.data_dir.glob("*.pt"))

        if len(self.data_files) == 0:
            raise ValueError(f"No data files found in {data_dir}")

        print(f"Found {len(self.data_files)} data files")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path = self.data_files[idx]

        # Load data
        if file_path.suffix == '.npy':
            data = np.load(file_path)
            data = torch.from_numpy(data).float()
        else:  # .pt file
            data = torch.load(file_path)

        # Ensure shape is (48, 224, 224) or (16, 3, 224, 224)
        if data.shape == (16, 3, 224, 224):
            data = data.reshape(48, 224, 224)

        # Normalize to [0, 1] if needed
        if data.max() > 1.0:
            data = data / 255.0

        if self.transform:
            data = self.transform(data)

        return data


def vae_loss_function(recon_x, x, mu, logvar, kld_weight=1.0):
    """
    VAE loss = Reconstruction loss + KL divergence

    Args:
        recon_x: Reconstructed images
        x: Original images
        mu: Mean from encoder
        logvar: Log variance from encoder
        kld_weight: Weight for KL divergence term (beta in beta-VAE)

    Returns:
        loss: Total loss
        recon_loss: Reconstruction loss
        kld_loss: KL divergence loss
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    loss = recon_loss + kld_weight * kld_loss

    return loss, recon_loss, kld_loss


def train_epoch(model, dataloader, optimizer, device, kld_weight=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kld = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, data in enumerate(pbar):
        data = data.to(device)

        optimizer.zero_grad()

        # Forward pass
        recon_batch, mu, logvar = model(data)

        # Compute loss
        loss, recon_loss, kld_loss = vae_loss_function(
            recon_batch, data, mu, logvar, kld_weight
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kld += kld_loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item() / len(data),
            'recon': recon_loss.item() / len(data),
            'kld': kld_loss.item() / len(data)
        })

    n_samples = len(dataloader.dataset)
    return {
        'loss': total_loss / n_samples,
        'recon_loss': total_recon / n_samples,
        'kld_loss': total_kld / n_samples
    }


def validate(model, dataloader, device, kld_weight=1.0):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kld = 0

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)

            # Forward pass
            recon_batch, mu, logvar = model(data)

            # Compute loss
            loss, recon_loss, kld_loss = vae_loss_function(
                recon_batch, data, mu, logvar, kld_weight
            )

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld_loss.item()

    n_samples = len(dataloader.dataset)
    return {
        'loss': total_loss / n_samples,
        'recon_loss': total_recon / n_samples,
        'kld_loss': total_kld / n_samples
    }


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"Checkpoint saved to {filepath}")


def main(args):
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize W&B
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=args.run_name
        )

    # Create datasets
    train_dataset = ImageDataset(args.data_dir)

    # Split into train/val if val_split > 0
    if args.val_split > 0:
        train_size = int((1 - args.val_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    else:
        val_loader = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    print(f"Train samples: {len(train_dataset)}")
    if val_loader:
        print(f"Val samples: {len(val_dataset)}")

    # Create model
    model = ResNetVAE(latent_dim=args.latent_dim).to(device)
    print(f"Model created with latent_dim={args.latent_dim}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    else:
        scheduler = None

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, args.kld_weight
        )

        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Recon: {train_metrics['recon_loss']:.4f}, "
              f"KLD: {train_metrics['kld_loss']:.4f}")

        # Validate
        if val_loader:
            val_metrics = validate(model, val_loader, device, args.kld_weight)
            print(f"Val - Loss: {val_metrics['loss']:.4f}, "
                  f"Recon: {val_metrics['recon_loss']:.4f}, "
                  f"KLD: {val_metrics['kld_loss']:.4f}")

        # Log to W&B
        if args.use_wandb:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/recon_loss': train_metrics['recon_loss'],
                'train/kld_loss': train_metrics['kld_loss'],
                'lr': optimizer.param_groups[0]['lr']
            }
            if val_loader:
                log_dict.update({
                    'val/loss': val_metrics['loss'],
                    'val/recon_loss': val_metrics['recon_loss'],
                    'val/kld_loss': val_metrics['kld_loss']
                })
            wandb.log(log_dict)

        # Update learning rate
        if scheduler:
            scheduler.step()

        # Save checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = save_dir / f"vae_epoch_{epoch}.pth"
            save_checkpoint(model, optimizer, epoch, train_metrics['loss'], checkpoint_path)

        # Save best model
        if val_loader:
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_path = save_dir / "vae_best.pth"
                save_checkpoint(model, optimizer, epoch, val_metrics['loss'], best_path)
                print(f"New best model saved with val_loss: {best_val_loss:.4f}")

    # Save final model
    final_path = save_dir / "vae_final.pth"
    save_checkpoint(model, optimizer, args.epochs, train_metrics['loss'], final_path)

    if args.use_wandb:
        wandb.finish()

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet VAE for vision encoding")

    # Data args
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing training images')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1)')

    # Model args
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent dimension size (default: 128)')
    parser.add_argument('--kld_weight', type=float, default=1.0,
                        help='Weight for KL divergence term (beta-VAE) (default: 1.0)')

    # Training args
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')

    # Scheduler args
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'none'],
                        help='Learning rate scheduler (default: step)')
    parser.add_argument('--scheduler_step', type=int, default=30,
                        help='Step size for StepLR (default: 30)')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1,
                        help='Gamma for StepLR (default: 0.1)')

    # Saving args
    parser.add_argument('--save_dir', type=str, default='./trained_models/vision',
                        help='Directory to save models (default: ./trained_models/vision)')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')

    # W&B args
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='speedracer-vision',
                        help='W&B project name (default: speedracer-vision)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='W&B run name')

    # Other args
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    main(args)
