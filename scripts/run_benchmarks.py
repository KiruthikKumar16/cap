
"""
Benchmark Script

This script runs evaluations for our model and the SOTA baselines (PressLight, CoLight)
and saves the results for comparison.
"""

import argparse
import yaml
import json
from pathlib import Path

from src.phase1.evaluate import evaluate_model

def run_benchmarks(config_path: str):
    """Run all benchmarks and save results."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    results = {}

    # Evaluate our model
    print("Evaluating Our Model...")
    our_model_results = evaluate_model(config, "PPO")
    results["our_model"] = our_model_results

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
    parser.add_argument("--config", type=str, default="configs/phase2_10x10.yaml", help="Path to config file")
    args = parser.parse_args()
    run_benchmarks(args.config)
