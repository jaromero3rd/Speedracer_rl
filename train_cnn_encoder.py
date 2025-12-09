#!/usr/bin/env python3
"""
Train CNN encoder to predict CartPole state buffer from stacked frames.
Uses ResNet18 architecture trained from scratch.

Input: (B, 48, 224, 224) - 16 stacked RGB frames
Output: (B, 64) - flattened observation buffer (16 timesteps x 4 state dims)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import argparse
import os
import json
from glob import glob
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import wandb

from vision_network import ResNetStateEncoder


class ChunkedVisionDataset(Dataset):
    """Dataset that loads from multiple chunk files."""

    def __init__(self, data_dir, normalize_states=True):
        """
        Args:
            data_dir: Directory containing chunk_*.npz files
            normalize_states: Whether to normalize state values
        """
        self.data_dir = data_dir
        self.normalize_states = normalize_states

        # Find all chunk files
        self.chunk_files = sorted(glob(os.path.join(data_dir, 'chunk_*.npz')))
        if len(self.chunk_files) == 0:
            raise ValueError(f"No chunk files found in {data_dir}")

        # Load metadata and stats
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)

        with open(os.path.join(data_dir, 'state_stats.json'), 'r') as f:
            self.state_stats = json.load(f)

        # Build index: (chunk_idx, sample_idx) for each sample
        self.index = []
        self.chunk_sizes = []

        for chunk_idx, chunk_file in enumerate(self.chunk_files):
            data = np.load(chunk_file)
            n_samples = len(data['states'])
            self.chunk_sizes.append(n_samples)
            for sample_idx in range(n_samples):
                self.index.append((chunk_idx, sample_idx))

        # Cache for loaded chunks (keep last N in memory)
        self.chunk_cache = {}
        self.cache_size = 3  # Keep 3 chunks in memory

        print(f"Loaded dataset from {data_dir}")
        print(f"  Chunks: {len(self.chunk_files)}")
        print(f"  Total samples: {len(self.index)}")
        print(f"  State dim: {self.metadata.get('state_dim', 64)}")

    def __len__(self):
        return len(self.index)

    def _load_chunk(self, chunk_idx):
        """Load a chunk, using cache."""
        if chunk_idx not in self.chunk_cache:
            # Evict old chunks if cache is full
            if len(self.chunk_cache) >= self.cache_size:
                oldest = min(self.chunk_cache.keys())
                del self.chunk_cache[oldest]

            data = np.load(self.chunk_files[chunk_idx])
            self.chunk_cache[chunk_idx] = {
                'frame_stacks': data['frame_stacks'],
                'states': data['states']
            }

        return self.chunk_cache[chunk_idx]

    def __getitem__(self, idx):
        chunk_idx, sample_idx = self.index[idx]
        chunk = self._load_chunk(chunk_idx)

        # Get frame stack: (16, 224, 224, 3)
        frame_stack = chunk['frame_stacks'][sample_idx]

        # Reshape: (16, 224, 224, 3) -> (48, 224, 224)
        # Concatenate RGB channels across frames
        frame_tensor = torch.from_numpy(frame_stack).float() / 255.0
        # (16, 224, 224, 3) -> (16, 3, 224, 224) -> (48, 224, 224)
        frame_tensor = frame_tensor.permute(0, 3, 1, 2)  # (16, 3, 224, 224)
        frame_tensor = frame_tensor.reshape(48, 224, 224)  # (48, 224, 224)

        # Get state (64-dim)
        state = torch.from_numpy(chunk['states'][sample_idx]).float()

        # Normalize states if requested
        if self.normalize_states:
            mean = torch.tensor(self.state_stats['mean'], dtype=torch.float32)
            std = torch.tensor(self.state_stats['std'], dtype=torch.float32)
            state = (state - mean) / (std + 1e-8)

        return frame_tensor, state


def validate_batches(model, dataset, criterion, device, batch_size, num_batches=100, desc="Validating"):
    """Validate on a random subset of batches."""
    from torch.utils.data import DataLoader, Subset
    import random

    model.eval()
    total_loss = 0.0
    batches_done = 0
    all_predictions = []
    all_targets = []

    # Randomly sample indices for this validation run
    total_samples = len(dataset)
    num_samples = min(num_batches * batch_size, total_samples)
    random_indices = random.sample(range(total_samples), num_samples)

    subset = Subset(dataset, random_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    pbar = tqdm(loader, desc=desc, leave=False, unit="batch", total=min(num_batches, len(loader)))
    with torch.no_grad():
        for frames, states in pbar:
            if batches_done >= num_batches:
                break

            frames = frames.to(device)
            states = states.to(device)

            predictions = model(frames)
            loss = criterion(predictions, states)

            total_loss += loss.item()
            batches_done += 1

            all_predictions.append(predictions.cpu())
            all_targets.append(states.cpu())

            # Update progress bar
            avg_loss = total_loss / batches_done
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

    if len(all_predictions) == 0:
        return 0.0, np.zeros(64)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute per-dimension MAE
    mae_per_dim = torch.abs(all_predictions - all_targets).mean(dim=0)

    return total_loss / batches_done, mae_per_dim.numpy()


def train_with_interleaved_eval(model, train_dataset, val_dataset, criterion, optimizer,
                                 device, batch_size, train_batches=800, val_batches=100,
                                 epoch_pbar=None, use_wandb=False, global_step=0):
    """Train with interleaved validation checks every N batches."""
    import time
    from torch.utils.data import DataLoader

    model.train()
    total_loss = 0.0
    num_batches = 0
    total_samples = 0
    start_time = time.time()
    val_losses = []

    # Create training dataloader (shuffled each epoch)
    # Note: num_workers=0 avoids multiprocessing issues in WSL
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    total_train_batches = len(train_loader)
    pbar = tqdm(train_loader, desc="Training", leave=False, unit="batch")

    for batch_idx, (frames, states) in enumerate(pbar):
        # Run validation check before first batch and every train_batches
        if batch_idx % train_batches == 0:
            val_loss, val_mae = validate_batches(
                model, val_dataset, criterion, device,
                batch_size=batch_size, num_batches=val_batches,
                desc=f"Val check {batch_idx//train_batches + 1}"
            )
            val_losses.append(val_loss)
            model.train()  # Switch back to train mode

            if use_wandb:
                import wandb
                wandb.log({
                    'batch_val_loss': val_loss,
                    'batch_val_mae_pos': val_mae[0],
                    'global_step': global_step + batch_idx
                })

            tqdm.write(f"  [Batch {batch_idx}] Val loss: {val_loss:.4f}")

        batch_size_actual = frames.size(0)
        frames = frames.to(device)
        states = states.to(device)

        optimizer.zero_grad()
        predictions = model(frames)
        loss = criterion(predictions, states)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        total_samples += batch_size_actual

        # Calculate running stats
        avg_loss = total_loss / num_batches
        elapsed = time.time() - start_time
        samples_per_sec = total_samples / elapsed if elapsed > 0 else 0

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg': f'{avg_loss:.4f}',
            'samples/s': f'{samples_per_sec:.0f}'
        })

        # Update epoch progress bar
        if epoch_pbar is not None:
            epoch_pbar.set_postfix({'train_loss': f'{avg_loss:.4f}'})

    # Final validation check at end of epoch
    val_loss, val_mae = validate_batches(
        model, val_dataset, criterion, device,
        batch_size=batch_size, num_batches=val_batches,
        desc="Val check (end)"
    )
    val_losses.append(val_loss)

    return total_loss / num_batches, val_losses, val_mae, global_step + total_train_batches


def validate(model, dataloader, criterion, device, desc="Validating"):
    """Validate the model on full dataset."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_targets = []

    pbar = tqdm(dataloader, desc=desc, leave=False, unit="batch")
    with torch.no_grad():
        for frames, states in pbar:
            frames = frames.to(device)
            states = states.to(device)

            predictions = model(frames)
            loss = criterion(predictions, states)

            total_loss += loss.item()
            num_batches += 1

            all_predictions.append(predictions.cpu())
            all_targets.append(states.cpu())

            # Update progress bar
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute per-dimension MAE (for first 4 dims as sample)
    mae_per_dim = torch.abs(all_predictions - all_targets).mean(dim=0)

    return total_loss / num_batches, mae_per_dim.numpy()


def main():
    parser = argparse.ArgumentParser(description='Train CNN encoder for CartPole state prediction')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing agent subdirs with chunk files')
    parser.add_argument('--output_dir', type=str, default='trained_cnn',
                        help='Directory to save trained model')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34'],
                        help='Backbone architecture (default: resnet18)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128, good for RTX 4090)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1)')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Test split ratio (default: 0.1)')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize state values')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='DataLoader workers (default: 8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--wandb_project', type=str, default='cartpole-cnn',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity (team/username)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Weights & Biases run name')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--resplit', action='store_true',
                        help='Force recreate train/val/test split (overwrites existing)')
    parser.add_argument('--skip_test', action='store_true',
                        help='Skip final test evaluation (for hyperparameter tuning)')
    parser.add_argument('--train_batches', type=int, default=800,
                        help='Number of training batches between validation checks (default: 800)')
    parser.add_argument('--val_batches', type=int, default=100,
                        help='Number of validation batches per check (default: 100)')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                'backbone': args.backbone,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'learning_rate': args.lr,
                'val_split': args.val_split,
                'test_split': args.test_split,
                'normalize_states': args.normalize,
                'num_workers': args.num_workers,
                'seed': args.seed,
                'train_batches': args.train_batches,
                'val_batches': args.val_batches,
            }
        )
        print(f"Wandb run: {wandb.run.url}\n")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Load datasets from all agent subdirectories
    print("Loading datasets...")
    all_datasets = []
    for subdir in sorted(os.listdir(args.data_dir)):
        subdir_path = os.path.join(args.data_dir, subdir)
        if os.path.isdir(subdir_path) and glob(os.path.join(subdir_path, 'chunk_*.npz')):
            print(f"  Loading {subdir}...")
            dataset = ChunkedVisionDataset(subdir_path, normalize_states=args.normalize)
            all_datasets.append(dataset)

    if len(all_datasets) == 0:
        raise ValueError(f"No valid datasets found in {args.data_dir}")

    # Concatenate all datasets
    print(f"\nCombining {len(all_datasets)} datasets...")
    combined_dataset = ConcatDataset(all_datasets)
    print(f"Total samples: {len(combined_dataset):,}\n")

    # Split into train/val/test (80/10/10)
    # Use deterministic splits - save indices to file for reproducibility
    split_file = os.path.join(args.data_dir, 'split_indices.npz')
    total_len = len(combined_dataset)

    if os.path.exists(split_file) and not args.resplit:
        # Load existing split indices
        print("Loading existing split indices...")
        split_data = np.load(split_file)
        train_indices = split_data['train'].tolist()
        val_indices = split_data['val'].tolist()
        test_indices = split_data['test'].tolist()
        print(f"  Loaded from: {split_file}")
    else:
        # Create new deterministic split
        print("Creating new deterministic split...")
        rng = np.random.RandomState(args.seed)
        indices = rng.permutation(total_len)

        test_size = int(args.test_split * total_len)
        val_size = int(args.val_split * total_len)
        train_size = total_len - val_size - test_size

        train_indices = indices[:train_size].tolist()
        val_indices = indices[train_size:train_size + val_size].tolist()
        test_indices = indices[train_size + val_size:].tolist()

        # Save split indices
        np.savez(split_file,
                 train=np.array(train_indices),
                 val=np.array(val_indices),
                 test=np.array(test_indices))
        print(f"  Saved to: {split_file}")

    # Create subset datasets
    from torch.utils.data import Subset
    train_dataset = Subset(combined_dataset, train_indices)
    val_dataset = Subset(combined_dataset, val_indices)
    test_dataset = Subset(combined_dataset, test_indices)

    print(f"\nTrain samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}\n")

    # Create test dataloader (train/val loaders created per-epoch for shuffling)
    # Note: num_workers=0 avoids multiprocessing issues in WSL
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Create model
    print(f"Creating {args.backbone} encoder...")
    model = ResNetStateEncoder(
        input_channels=48,
        output_dim=64,
        backbone=args.backbone
    )
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}\n")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print("=" * 60)
    print("Starting training...")
    print(f"  Train batches between val checks: {args.train_batches}")
    print(f"  Val batches per check: {args.val_batches}")
    print("=" * 60 + "\n")

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    global_step = 0

    # Epoch-level progress bar
    epoch_pbar = tqdm(range(args.epochs), desc="Training", unit="epoch")

    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch+1}/{args.epochs}")

        # Train with interleaved validation
        train_loss, epoch_val_losses, mae_per_dim, global_step = train_with_interleaved_eval(
            model, train_dataset, val_dataset, criterion, optimizer, device,
            batch_size=args.batch_size,
            train_batches=args.train_batches,
            val_batches=args.val_batches,
            epoch_pbar=epoch_pbar,
            use_wandb=use_wandb,
            global_step=global_step
        )
        train_losses.append(train_loss)

        # Use the last validation loss from the epoch
        val_loss = epoch_val_losses[-1]
        val_losses.append(val_loss)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Update epoch progress bar with final stats
        epoch_pbar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'lr': f'{current_lr:.1e}',
            'best': f'{min(best_val_loss, val_loss):.4f}'
        })

        # Print detailed stats
        tqdm.write(f"\nEpoch {epoch+1}/{args.epochs}")
        tqdm.write(f"  Train Loss: {train_loss:.6f}")
        tqdm.write(f"  Val Loss:   {val_loss:.6f}")
        tqdm.write(f"  LR:         {current_lr:.6f}")
        tqdm.write(f"  MAE (pos, vel, angle, ang_vel): {mae_per_dim[:4]}")

        # Log to wandb
        if use_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr,
                'mae_pos': mae_per_dim[0],
                'mae_vel': mae_per_dim[1],
                'mae_angle': mae_per_dim[2],
                'mae_ang_vel': mae_per_dim[3],
            }
            wandb.log(log_dict)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'backbone': args.backbone,
                'normalize': args.normalize
            }, os.path.join(args.output_dir, 'best_model.pth'))
            tqdm.write(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'backbone': args.backbone
            }, os.path.join(args.output_dir, f'checkpoint_epoch{epoch+1}.pth'))

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'backbone': args.backbone,
        'normalize': args.normalize
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
    plt.savefig(os.path.join(args.output_dir, 'training_history.png'), dpi=150)
    plt.close()

    # Final test evaluation with best model
    test_loss = None
    test_mae = None

    if not args.skip_test:
        print("\n" + "=" * 60)
        print("Evaluating best model on test set...")
        print("=" * 60)

        # Load best model
        checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_mae = validate(model, test_loader, criterion, device, desc="Testing")
        print(f"\nTest Loss: {test_loss:.6f}")
        print(f"Test MAE (first 4 dims): {test_mae[:4]}")

    print("\n" + "=" * 60)
    print(f"✓ Training complete!")
    print(f"  Best val loss: {best_val_loss:.6f}")
    if test_loss is not None:
        print(f"  Test loss: {test_loss:.6f}")
    else:
        print(f"  Test evaluation: skipped (use without --skip_test for final eval)")
    print(f"  Models saved to: {args.output_dir}")
    print("=" * 60)

    # Finish wandb run
    if use_wandb:
        # Log final summary
        wandb.summary['best_val_loss'] = best_val_loss
        wandb.summary['final_train_loss'] = train_losses[-1]
        wandb.summary['total_epochs'] = args.epochs

        if test_loss is not None:
            wandb.summary['test_loss'] = test_loss
            wandb.summary['test_mae_pos'] = test_mae[0]
            wandb.summary['test_mae_vel'] = test_mae[1]
            wandb.summary['test_mae_angle'] = test_mae[2]
            wandb.summary['test_mae_ang_vel'] = test_mae[3]

        # Save model artifact
        artifact = wandb.Artifact('cnn-encoder', type='model')
        artifact.add_file(os.path.join(args.output_dir, 'best_model.pth'))
        wandb.log_artifact(artifact)

        wandb.finish()
        print("Wandb run finished.")


if __name__ == "__main__":
    main()
