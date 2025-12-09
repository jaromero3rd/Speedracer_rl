#!/usr/bin/env python3
"""
Train CNN encoder on pre-batched data with interleaved validation.
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
    return sorted(glob(os.path.join(folder, 'batch_*_frames.npy')))


def load_batch(frames_file):
    states_file = frames_file.replace('_frames.npy', '_states.npy')
    return np.load(frames_file), np.load(states_file)


def process_frames(frames_np):
    # (N, 16, 224, 224, 3) -> (N, 48, 224, 224)
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
    parser.add_argument('--train_batches', type=int, default=80)
    parser.add_argument('--val_batches', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_files = get_batch_files(os.path.join(args.data_dir, 'train'))
    val_files = get_batch_files(os.path.join(args.data_dir, 'val'))
    test_files = get_batch_files(os.path.join(args.data_dir, 'test'))

    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    with open(os.path.join(args.data_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    # compute normalization from subset of training data
    if args.normalize:
        print("Computing norm stats...")
        all_states = []
        for f in tqdm(train_files[:20], desc="Loading"):
            _, states = load_batch(f)
            all_states.append(states)
        all_states = np.concatenate(all_states)
        state_mean = torch.from_numpy(all_states.mean(axis=0)).float().to(device)
        state_std = torch.from_numpy(all_states.std(axis=0) + 1e-8).float().to(device)

        torch.save({'mean': state_mean.cpu(), 'std': state_std.cpu()},
                   os.path.join(args.output_dir, 'norm_stats.pt'))
        del all_states
    else:
        state_mean = torch.zeros(64).to(device)
        state_std = torch.ones(64).to(device)

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    model = ResNetStateEncoder(input_channels=48, output_dim=64).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(args.epochs):
        train_loss_sum = 0
        train_count = 0
        train_idx = 0
        val_idx = 0

        pbar = tqdm(total=len(train_files), desc=f"Epoch {epoch+1}")

        # initial val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for j in range(args.val_batches):
                fidx = (val_idx + j) % len(val_files)
                frames_np, states_np = load_batch(val_files[fidx])
                frames = process_frames(frames_np).to(device)
                states = torch.from_numpy(states_np).float().to(device)
                if args.normalize:
                    states = (states - state_mean) / state_std
                pred = model(frames)
                val_loss += criterion(pred, states).item()
        val_idx = (val_idx + args.val_batches) % len(val_files)
        pbar.set_postfix({'val': f'{val_loss/args.val_batches:.4f}'})
        if use_wandb:
            wandb.log({'val_loss_interleaved': val_loss/args.val_batches, 'step': global_step})

        while train_idx < len(train_files):
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

                train_loss_sum += loss.item()
                train_count += 1
                global_step += 1
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                if use_wandb:
                    wandb.log({'train_loss': loss.item(), 'step': global_step})

            train_idx = phase_end

            # interleaved validation
            if train_idx < len(train_files):
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for j in range(args.val_batches):
                        fidx = (val_idx + j) % len(val_files)
                        frames_np, states_np = load_batch(val_files[fidx])
                        frames = process_frames(frames_np).to(device)
                        states = torch.from_numpy(states_np).float().to(device)
                        if args.normalize:
                            states = (states - state_mean) / state_std
                        pred = model(frames)
                        val_loss += criterion(pred, states).item()
                val_idx = (val_idx + args.val_batches) % len(val_files)

                if use_wandb:
                    wandb.log({'val_loss_interleaved': val_loss/args.val_batches, 'step': global_step})

        pbar.close()

        # full validation at end of epoch
        model.eval()
        full_val_loss = 0
        with torch.no_grad():
            for f in val_files:
                frames_np, states_np = load_batch(f)
                frames = process_frames(frames_np).to(device)
                states = torch.from_numpy(states_np).float().to(device)
                if args.normalize:
                    states = (states - state_mean) / state_std
                pred = model(frames)
                full_val_loss += criterion(pred, states).item()
        avg_val = full_val_loss / len(val_files)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}: train={train_loss_sum/train_count:.4f}, val={avg_val:.4f}, lr={lr:.1e}")

        if use_wandb:
            wandb.log({'val_loss': avg_val, 'lr': lr, 'epoch': epoch+1})

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  Saved best model")

    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))

    if use_wandb:
        wandb.finish()

    print(f"\nDone! Best val: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
