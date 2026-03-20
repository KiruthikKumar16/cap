import subprocess
import yaml
import json
from pathlib import Path
import os

SEED = 42

def run_ablation_study():
    """Runs the full ablation study for the PR-MARL model."""
    
    ablation_configs = {
        "rl_only": {
            "model": {"use_gnn": False}},
        "gnn_rl": {
            "model": {"use_gnn": True}},
        "gnn_rl_forecast": {
            "model": {"use_gnn": True}},
        "full_model": {
            "model": {"use_gnn": True}},
    }
    
    results = {}
    
    for model_name, config_override in ablation_configs.items():
        print(f"\n--- Running Ablation: {model_name} ---")
        
        with open("configs/phase2_10x10.yaml", 'r') as f:
            base_config = yaml.safe_load(f)
        
        base_config.update(config_override)
        base_config["experiment"] = {"seed": SEED}
        
        temp_config_path = f"configs/temp_{model_name}_config.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(base_config, f)
            
        train_cmd = [
            "python", "-m", "src.phase1.train_marl",
            "--config", temp_config_path,
            "--total-timesteps", "10000"
        ]
        subprocess.run(train_cmd, check=True)
        
        eval_cmd = [
            "python", "-m", "src.phase1.evaluate",
            "--config", temp_config_path,
            "--save-summary", f"outputs/{model_name}_eval.json"
        ]
        subprocess.run(eval_cmd, check=True)
        
        with open(f"outputs/{model_name}_eval.json", 'r') as f:
            results[model_name] = json.load(f)
            
        os.remove(temp_config_path)
            
    with open("outputs/ablation_results.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n--- Ablation Study Complete ---")
    print("Results saved to outputs/ablation_results.json")

if __name__ == "__main__":
    run_ablation_study()
