
"""
Benchmark Script

This script runs evaluations for our model and the SOTA baselines (PressLight, CoLight)
and saves the results for comparison.
"""

import argparse
import sys
import yaml
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.phase1.evaluate import evaluate_model

def run_benchmarks(config_path: str, checkpoint: str, episodes: int):
    """Run all benchmarks and save results."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    if "evaluation" not in config:
        config["evaluation"] = {}
    config["evaluation"]["num_episodes"] = episodes
    
    if "output" not in config:
        config["output"] = {}
    config["output"]["final_model_path"] = checkpoint

    results = {}

    # Evaluate our model
    print("Evaluating MAPPO-STGNN (Ours)...")
    our_model_results = evaluate_model(config, "PPO")
    results["MAPPO-STGNN"] = our_model_results

    # Evaluate PressLight
    print("Evaluating PressLight...")
    presslight_results = evaluate_model(config, "PressLight")
    results["presslight"] = presslight_results

    # Evaluate CoLight
    print("Evaluating CoLight...")
    colight_results = evaluate_model(config, "CoLight")
    results["colight"] = colight_results

    # Save results
    output_path = Path("outputs/benchmark_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Benchmark results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOTA benchmarks.")
    parser.add_argument("--config", type=str, default="configs/phase1.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained model zip")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per baseline")
    args = parser.parse_args()
    run_benchmarks(args.config, args.checkpoint, args.episodes)
