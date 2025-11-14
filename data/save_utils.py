import os
import pickle
import json

def create_run_directory(base_path="/data"):
    """
    Create a new run directory with incrementing number.
    Returns the path to the created directory.
    """
    # Create base data directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Find existing run directories
    existing_runs = [d for d in os.listdir(base_path) if d.startswith("run_")]
    
    if existing_runs:
        # Extract run numbers and find the max
        run_numbers = [int(d.split("_")[1]) for d in existing_runs if d.split("_")[1].isdigit()]
        next_run = max(run_numbers) + 1 if run_numbers else 0
    else:
        next_run = 0
    
    # Create new run directory
    run_dir = os.path.join(base_path, f"run_{next_run:04d}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories for organization
    os.makedirs(os.path.join(run_dir, "episodes"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    
    print(f"Created run directory: {run_dir}")
    return run_dir


def save_episode_data(run_dir, episode, episode_data):
    """
    Save episode data to disk.
    
    Args:
        run_dir: Path to the run directory
        episode: Episode number
        episode_data: Dictionary containing episode information
    """
    episode_file = os.path.join(run_dir, "episodes", f"episode_{episode:04d}.pkl")
    
    with open(episode_file, 'wb') as f:
        pickle.dump(episode_data, f)


def save_step_data(run_dir, episode, step, step_data):
    """
    Save individual step data to disk (optional, for detailed logging).
    
    Args:
        run_dir: Path to the run directory
        episode: Episode number
        step: Step number
        step_data: Dictionary containing step information
    """
    step_dir = os.path.join(run_dir, "episodes", f"episode_{episode:04d}_steps")
    os.makedirs(step_dir, exist_ok=True)
    
    step_file = os.path.join(step_dir, f"step_{step:04d}.pkl")
    
    with open(step_file, 'wb') as f:
        pickle.dump(step_data, f)


def save_run_metadata(run_dir, metadata):
    """
    Save metadata about the run configuration.
    
    Args:
        run_dir: Path to the run directory
        metadata: Dictionary containing run configuration
    """
    metadata_file = os.path.join(run_dir, "run_metadata.json")
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)


def save_training_log(run_dir, log_data):
    """
    Append training log entry to the log file.
    
    Args:
        run_dir: Path to the run directory
        log_data: Dictionary containing log information
    """
    log_file = os.path.join(run_dir, "logs", "training_log.json")
    
    # Read existing log if it exists
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    # Append new log entry
    logs.append(log_data)
    
    # Write back to file
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4)