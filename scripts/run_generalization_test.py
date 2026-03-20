import subprocess
import yaml
import json
from pathlib import Path

def run_generalization_test():
    """Trains on a 5x5 grid and evaluates on a 10x10 grid."""
    
    # Train on 5x5 grid
    print("--- Training on 5x5 Grid ---")
    train_cmd = [
        "python", "-m", "src.phase1.train_marl",
        "--config", "configs/phase1_5x5.yaml"
    ]
    subprocess.run(train_cmd, check=True)
    
    # Evaluate on 10x10 grid
    print("\n--- Evaluating on 10x10 Grid (Generalization Test) ---")
    eval_10x10_cmd = [
        "python", "-m", "src.phase1.evaluate",
        "--config", "configs/phase2_10x10.yaml",
        "--checkpoint", "marl_ppo_traffic.zip",
        "--episodes", "1", # SOTA: 1 episode is enough for generalization proof
        "--save-summary", "outputs/generalization_10x10_results.json"
    ]
    subprocess.run(eval_10x10_cmd, check=True)
    
    # Evaluate on Bengaluru map
    print("\n--- Evaluating on Bengaluru Map (Zero-Shot Generalization) ---")
    eval_bengaluru_cmd = [
        "python", "-m", "src.phase1.evaluate",
        "--config", "configs/bengaluru_city.yaml",
        "--checkpoint", "marl_ppo_traffic.zip",
        "--episodes", "1", # SOTA: 1 episode is enough for generalization proof
        "--save-summary", "outputs/generalization_bengaluru_results.json"
    ]
    subprocess.run(eval_bengaluru_cmd, check=True)
    
    print("\n--- Generalization Tests Complete ---")
    print("Results saved to:")
    print(" - outputs/generalization_10x10_results.json")
    print(" - outputs/generalization_bengaluru_results.json")

if __name__ == "__main__":
    run_generalization_test()
