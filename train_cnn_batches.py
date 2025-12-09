#!/usr/bin/env python3
"""
CNN training from pre-batched .npy files with 800/100 interleaved train/val.
Expects data_dir with train/, val/, test/ subdirectories.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import json
from glob import glob
from tqdm import tqdm
import wandb

from vision_network import ResNetStateEncoder


def get_batch_files(folder):
    """Get sorted list of batch files in folder."""
    frames_files = sorted(glob(os.path.join(folder, 'batch_*_frames.npy')))
    return frames_files


def load_batch(frames_file):
    """Load a batch given the frames file path."""
    states_file = frames_file.replace('_frames.npy', '_states.npy')
    frames = np.load(frames_file)
    states = np.load(states_file)
    return frames, states


def process_frames(frames_np):
    """(N, 16, 224, 224, 3) -> (N, 48, 224, 224)"""
    t = torch.from_numpy(frames_np).float() / 255.0
    t = t.permute(0, 1, 4, 2, 3)
    t = t.reshape(t.shape[0], 48, 224, 224)
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='trained_cnn')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='cartpole-cnn')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--train_batches', type=int, default=80, help='Train batches before validation check')
    parser.add_argument('--val_batches', type=int, default=10, help='Val batches per check')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Get batch files for each split
    train_files = get_batch_files(os.path.join(args.data_dir, 'train'))
    val_files = get_batch_files(os.path.join(args.data_dir, 'val'))
    test_files = get_batch_files(os.path.join(args.data_dir, 'test'))

    print(f"Train batches: {len(train_files)}")
    print(f"Val batches: {len(val_files)}")
    print(f"Test batches: {len(test_files)}")

    # Load metadata
    with open(os.path.join(args.data_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    # Compute normalization from first 20 train batches
    if args.normalize:
        print("Computing normalization stats...")
        all_states = []
        for frames_file in tqdm(train_files[:20], desc="Loading stats"):
            _, states = load_batch(frames_file)
            all_states.append(states)
        all_states = np.concatenate(all_states)
        state_mean = torch.from_numpy(all_states.mean(axis=0)).float().to(device)
        state_std = torch.from_numpy(all_states.std(axis=0) + 1e-8).float().to(device)
        print(f"  Computed from {len(all_states):,} samples")

        # Save for inference
        torch.save({'mean': state_mean.cpu(), 'std': state_std.cpu()},
                   os.path.join(args.output_dir, 'norm_stats.pt'))
        del all_states
    else:
        state_mean = torch.zeros(64).to(device)
        state_std = torch.ones(64).to(device)

    # Wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
        print(f"Wandb: {wandb.run.url}\n")

    # Model
    model = ResNetStateEncoder(input_channels=48, output_dim=64).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    global_step = 0

    print(f"\nTraining with {args.val_batches}/{args.train_batches}/{args.val_batches}/... interleaved schedule...")

    for epoch in range(args.epochs):
        train_loss_epoch = 0
        train_batches_epoch = 0

        # Indices for this epoch (deterministic order)
        train_idx = 0
        val_idx = 0

        pbar = tqdm(total=len(train_files), desc=f"Epoch {epoch+1}/{args.epochs}")

        # === Initial validation phase ===
        model.eval()
        val_loss_phase = 0
        with torch.no_grad():
            for j in tqdm(range(args.val_batches), desc="  Initial val", leave=False):
                file_idx = (val_idx + j) % len(val_files)
                frames_np, states_np = load_batch(val_files[file_idx])
                frames = process_frames(frames_np).to(device)
                states = torch.from_numpy(states_np).float().to(device)
                if args.normalize:
                    states = (states - state_mean) / state_std
                pred = model(frames)
                val_loss_phase += criterion(pred, states).item()
        val_idx = (val_idx + args.val_batches) % len(val_files)
        avg_val = val_loss_phase / args.val_batches
        pbar.set_postfix({'val': f'{avg_val:.4f}'})
        if use_wandb:
            wandb.log({'val_loss_interleaved': avg_val, 'step': global_step})

        while train_idx < len(train_files):
            # === Training phase ===
            model.train()
            phase_end = min(train_idx + args.train_batches, len(train_files))

            for i in range(train_idx, phase_end):
                frames_np, states_np = load_batch(train_files[i])

                frames = process_frames(frames_np).to(device)
                states = torch.from_numpy(states_np).float().to(device)

                if args.normalize:
                    states = (states - state_mean) / state_std

                optimizer.zero_grad()
                pred = model(frames)
                loss = criterion(pred, states)
                loss.backward()
                optimizer.step()

                train_loss_epoch += loss.item()
                train_batches_epoch += 1
                global_step += 1

                pbar.update(1)
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg': f'{train_loss_epoch/train_batches_epoch:.4f}'
                })

                if use_wandb:
                    wandb.log({'train_loss_batch': loss.item(), 'step': global_step})

            train_idx = phase_end

            # === Validation phase (interleaved) ===
            if train_idx < len(train_files):
                model.eval()
                val_loss_phase = 0

                with torch.no_grad():
                    for j in tqdm(range(args.val_batches), desc="  Val", leave=False):
                        # Cycle through val files
                        file_idx = (val_idx + j) % len(val_files)
                        frames_np, states_np = load_batch(val_files[file_idx])

                        frames = process_frames(frames_np).to(device)
                        states = torch.from_numpy(states_np).float().to(device)

                        if args.normalize:
                            states = (states - state_mean) / state_std

                        pred = model(frames)
                        val_loss_phase += criterion(pred, states).item()

                val_idx = (val_idx + args.val_batches) % len(val_files)
                avg_val = val_loss_phase / args.val_batches

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg': f'{train_loss_epoch/train_batches_epoch:.4f}',
                    'val': f'{avg_val:.4f}'
                })

                if use_wandb:
                    wandb.log({'val_loss_interleaved': avg_val, 'step': global_step})

        pbar.close()

        # End of epoch
        avg_train_loss = train_loss_epoch / train_batches_epoch

        # Full validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for frames_file in tqdm(val_files, desc="  Full validation", leave=False):
                frames_np, states_np = load_batch(frames_file)
                frames = process_frames(frames_np).to(device)
                states = torch.from_numpy(states_np).float().to(device)
                if args.normalize:
                    states = (states - state_mean) / state_std
                pred = model(frames)
                val_loss += criterion(pred, states).item()
        avg_val_loss = val_loss / len(val_files)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}: train={avg_train_loss:.4f}, val={avg_val_loss:.4f}, lr={lr:.1e}")

        if use_wandb:
            wandb.log({
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'lr': lr,
                'epoch': epoch + 1
            })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  ✓ Saved best model")

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))

    if use_wandb:
        wandb.summary['best_val_loss'] = best_val_loss
        wandb.finish()

    print(f"\nDone! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
