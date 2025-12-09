#!/usr/bin/env python
"""
Hyperparameter grid search for SAC discrete.
"""

import itertools
import subprocess
import os
import json
from datetime import datetime
import argparse

def run_training(config, run_id, log_dir, env, seed):
    cmd = [
        "python", "train.py",
        "--run_name", f"grid_search_{run_id}",
        "--episodes", "100",
        "--log_dir", log_dir,
        "--env", env,
        "--seed", str(seed),
        "--learning_rate", str(config["learning_rate"]),
        "--epsilon", str(config["epsilon"]),
        "--obs_buffer_max_len", str(config["obs_buffer_max_len"]),
    ]

    if config["entropy_bonus"] is not None:
        cmd.extend(["--entropy_bonus", str(config["entropy_bonus"])])
    else:
        cmd.extend(["--entropy_bonus", "None"])

    print(f"\nConfig {run_id}: obs_buf={config['obs_buffer_max_len']}, lr={config['learning_rate']}, ent={config['entropy_bonus']}, eps={config['epsilon']}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {"status": "success", "config": config, "run_id": run_id}
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {e.stderr[:200]}")
        return {"status": "failed", "error": str(e), "config": config, "run_id": run_id}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--log_dir", type=str, default="./grid_search_logs")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    # hyperparameter grid
    obs_buffer_lens = [2, 8, 16]
    learning_rates = [1e-4, 5e-4, 1e-3]
    entropy_bonuses = [None, 0.2, 0.4]  # None = learnable
    epsilons = [0.0, 0.2, 0.4]

    grid = list(itertools.product(obs_buffer_lens, learning_rates, entropy_bonuses, epsilons))

    print(f"Grid search: {len(grid)} configs, 100 eps each")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.log_dir, f"grid_search_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # save config
    with open(os.path.join(results_dir, "grid_config.json"), "w") as f:
        json.dump({
            "obs_buffer_lens": obs_buffer_lens,
            "learning_rates": learning_rates,
            "entropy_bonuses": [str(x) for x in entropy_bonuses],
            "epsilons": epsilons,
        }, f, indent=2)

    results = []
    hparam_map = {}

    for idx, (obs_len, lr, ent, eps) in enumerate(grid, 1):
        config = {
            "obs_buffer_max_len": obs_len,
            "learning_rate": lr,
            "entropy_bonus": ent,
            "epsilon": eps
        }

        run_id = f"{idx:03d}"
        hparam_map[run_id] = config

        result = run_training(config, run_id, args.log_dir, args.env, args.seed)
        results.append(result)

        # save progress
        with open(os.path.join(results_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        print(f"Done {idx}/{len(grid)}")

    # final summary
    success = sum(1 for r in results if r['status'] == 'success')
    print(f"\nGrid search done: {success}/{len(grid)} successful")
    print(f"Results: {results_dir}")

if __name__ == "__main__":
    main()
