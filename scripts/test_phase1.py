"""
Test Phase 1: Benchmark the pure GNN-RL MAPPO policy against CoLight/PressLight metrics.
"""
import subprocess
import json

print("\n" + "="*50)
print("PHASE 1 TEST: RL & GNN Benchmarking")
print("="*50)

# Run the benchmark against 2 episodes (Fast test mode)
subprocess.run([
    "python", "scripts/run_benchmarks.py", 
    "--config", "configs/phase1.yaml", 
    "--checkpoint", "best_model_stage_2.zip",
    "--episodes", "2"
])

print("\n--- Benchmark JSON Output ---")
try:
    with open("outputs/benchmark_results.json", "r") as f:
        print(json.dumps(json.load(f), indent=2))
except Exception as e:
    print(f"Failed to read results: {e}")
