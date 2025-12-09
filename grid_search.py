#!/usr/bin/env python
"""
Hyperparameter grid search for SAC discrete agent.
Tests combinations of obs_buffer_max_len, learning_rate, entropy_bonus, and epsilon.
"""

import itertools
import subprocess
import os
import json
from datetime import datetime
import argparse

def run_training(config_dict, run_id, run_name, base_log_dir="./logs", env="CartPole-v1", seed=1):
    """Run a single training configuration.
    
    Args:
        config_dict: Dictionary with hyperparameters
        run_id: Unique identifier for this run
        run_name: Name for this run
        base_log_dir: Base directory for logs
        env: Environment name
        seed: Random seed
    
    Returns:
        Dictionary with results
    """
    # Build command
    cmd = [
        "python", "train.py",
        "--run_name", run_name,
        "--episodes", "100",
        "--env", env,
        "--seed", str(seed),
        "--learning_rate", str(config_dict["learning_rate"]),
        "--epsilon", str(config_dict["epsilon"]),
        "--obs_buffer_max_len", str(config_dict["obs_buffer_max_len"]),
        "--buffer_init_method", config_dict["buffer_init_method"],
    ]
    
    # Add entropy_bonus (always add it, use "None" string if None)
    if config_dict["entropy_bonus"] is not None:
        cmd.extend(["--entropy_bonus", str(config_dict["entropy_bonus"])])
    else:
        cmd.extend(["--entropy_bonus", "None"])
    
    print(f"\n{'='*80}")
    print(f"Running configuration {run_id}:")
    print(f"  run_name: {run_name}")
    print(f"  obs_buffer_max_len: {config_dict['obs_buffer_max_len']}")
    print(f"  learning_rate: {config_dict['learning_rate']}")
    print(f"  entropy_bonus: {config_dict['entropy_bonus']}")
    print(f"  epsilon: {config_dict['epsilon']}")
    print(f"  buffer_init_method: {config_dict['buffer_init_method']}")
    print(f"{'='*80}\n")
    
    # Run training
    try:
        # Don't capture output so tqdm progress bars are visible in real-time
        result = subprocess.run(
            cmd,
            check=True
        )
        
        return {
            "status": "success",
            "output": "Training completed (output not captured to show tqdm progress)",
            "config": config_dict,
            "run_id": run_id
        }
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Training failed for configuration {run_id}")
        return {
            "status": "failed",
            "error": str(e),
            "config": config_dict,
            "run_id": run_id
        }

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter grid search')
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--log_dir", type=str, default="./grid_search_logs", help="Base log directory")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    
    args = parser.parse_args()
    
    # Define hyperparameter grids
    obs_buffer_max_lens = [1, 2, 4, 8, 16]
    learning_rates = [5e-4]  # Default learning rate
    entropy_bonuses = [0, 0.2]
    epsilons = [0, 0.1]
    buffer_init_methods = ["random"]  # Fixed to random actor initialization
    
    # Create grid of all combinations
    grid = list(itertools.product(
        obs_buffer_max_lens,
        learning_rates,
        entropy_bonuses,
        epsilons,
        buffer_init_methods
    ))
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER GRID SEARCH")
    print(f"{'='*80}")
    print(f"Total configurations: {len(grid)}")
    print(f"Episodes per configuration: 100")
    print(f"Total training episodes: {len(grid) * 100}")
    print(f"\nHyperparameter ranges:")
    print(f"  obs_buffer_max_len: {obs_buffer_max_lens}")
    print(f"  learning_rate: {learning_rates}")
    print(f"  entropy_bonus: {entropy_bonuses}")
    print(f"  epsilon: {epsilons}")
    print(f"  buffer_init_method: {buffer_init_methods} (fixed)")
    print(f"{'='*80}\n")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.log_dir, f"grid_search_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save grid configuration
    grid_config = {
        "obs_buffer_max_lens": obs_buffer_max_lens,
        "learning_rates": learning_rates,
        "entropy_bonuses": entropy_bonuses,
        "epsilons": epsilons,
        "buffer_init_methods": buffer_init_methods,
        "total_configs": len(grid),
        "episodes_per_config": 100,
        "timestamp": timestamp,
        "env": args.env,
        "seed": args.seed
    }
    
    with open(os.path.join(results_dir, "grid_config.json"), "w") as f:
        json.dump(grid_config, f, indent=2)
    
    # Create hyperparameter mapping (run_id -> hyperparameters)
    hyperparameter_mapping = {}
    
    # Run each configuration
    results = []
    for idx, (obs_buf_len, lr, ent_bonus, eps, buffer_method) in enumerate(grid, 1):
        config_dict = {
            "obs_buffer_max_len": obs_buf_len,
            "learning_rate": lr,
            "entropy_bonus": ent_bonus,
            "epsilon": eps,
            "buffer_init_method": buffer_method
        }
        
        run_id = f"{idx:03d}"
        # Create run name with descriptive parameters
        run_name = f"grid_search_obs{obs_buf_len}_eps{eps}_ent{ent_bonus}_{run_id}"
        
        # Store hyperparameter mapping
        # The actual log directory will be created by train.py with format: {run_name}_{timestamp}
        hyperparameter_mapping[run_id] = {
            "run_id": run_id,
            "obs_buffer_max_len": obs_buf_len,
            "learning_rate": lr,
            "entropy_bonus": ent_bonus if ent_bonus is not None else "learnable",
            "epsilon": eps,
            "buffer_init_method": buffer_method,
            "run_name": run_name,
            "note": "TensorBoard log directory format: {run_name}_{timestamp} in base log_dir"
        }
        
        result = run_training(config_dict, run_id, run_name, args.log_dir, args.env, args.seed)
        results.append(result)
        
        # Save intermediate results and hyperparameter mapping
        with open(os.path.join(results_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        with open(os.path.join(results_dir, "hyperparameter_mapping.json"), "w") as f:
            json.dump(hyperparameter_mapping, f, indent=2)
        
        # Also save as CSV for easy viewing
        import csv
        csv_file = os.path.join(results_dir, "hyperparameter_mapping.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['run_id', 'obs_buffer_max_len', 'learning_rate', 
                                                   'entropy_bonus', 'epsilon', 'buffer_init_method', 'run_name'])
            writer.writeheader()
            for run_id, params in sorted(hyperparameter_mapping.items(), key=lambda x: int(x[0])):
                # Create a row without the 'note' field
                row = {k: v for k, v in params.items() if k != 'note'}
                writer.writerow(row)
        
        print(f"\nCompleted {idx}/{len(grid)} configurations\n")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"GRID SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Total configurations: {len(grid)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"\nResults saved to: {results_dir}")
    print(f"{'='*80}\n")
    
    # Save final results
    with open(os.path.join(results_dir, "final_results.json"), "w") as f:
        json.dump({
            "grid_config": grid_config,
            "results": results,
            "summary": {
                "total": len(grid),
                "successful": sum(1 for r in results if r['status'] == 'success'),
                "failed": sum(1 for r in results if r['status'] == 'failed')
            }
        }, f, indent=2)

if __name__ == "__main__":
    main()

