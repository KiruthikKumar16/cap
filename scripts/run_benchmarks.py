
"""
Benchmark Script

This script runs evaluations for MAPPO vs NSTLight baseline
and saves results for comparison (with latency summary if available).
"""

print("Debug: Starting run_benchmarks.py...", flush=True)
import argparse
import sys
import yaml
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print("Debug: Importing evaluate_model...", flush=True)
from src.phase1.evaluate import evaluate_model
print("Debug: evaluate_model imported.", flush=True)

def run_benchmarks(config_path: str, checkpoint: str, episodes: int):
    """Run all benchmarks and save results."""
    print(f"Debug: Entering run_benchmarks with config={config_path}, episodes={episodes}", flush=True)
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
    print("Evaluating MAPPO-STGNN (Ours)...", flush=True)
    results["MAPPO-STGNN"] = evaluate_model(config, "PPO")

    # Evaluate SOTA Heuristic: MaxPressure
    print("Evaluating MaxPressure...", flush=True)
    results["MaxPressure"] = evaluate_model(config, "MaxPressure")

    # Evaluate SOTA MARL: PressLight
    print("Evaluating PressLight...", flush=True)
    results["PressLight"] = evaluate_model(config, "PressLight")

    # Evaluate SOTA GNN: CoLight
    print("Evaluating CoLight...", flush=True)
    results["CoLight"] = evaluate_model(config, "CoLight")

    # Evaluate SOTA GNN: NSTLight
    print("Evaluating NSTLight...", flush=True)
    results["NSTLight"] = evaluate_model(config, "NSTLight")

    # Keep fixed-time for hardware-independent sanity check.
    print("Evaluating Fixed-Time...", flush=True)
    results["FixedTime"] = evaluate_model(config, "FixedTime")

    # Evaluate Random baseline
    print("Evaluating Random...", flush=True)
    results["Random"] = evaluate_model(config, "Random")

    # Append latency outputs to benchmark summary when available.
    latency_path = Path("outputs/latency/inference_latency.json")
    if latency_path.exists():
        with open(latency_path, "r", encoding="utf-8") as f:
            results["latency_ms_per_step"] = json.load(f)

    # Save results
    output_path = Path("outputs/benchmark_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Benchmark results saved to {output_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOTA benchmarks.")
    parser.add_argument("--config", type=str, default="configs/phase1.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained model zip")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per baseline")
    args = parser.parse_args()
    run_benchmarks(args.config, args.checkpoint, args.episodes)
