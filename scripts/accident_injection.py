import argparse
import sys
from pathlib import Path
import json

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.phase1.train_rl import load_config
from src.phase1.evaluate import evaluate_model
from src.baselines.presslight import PresslightAgent

def main():
    print("="*60)
    print("Phase 3: SOTA Adversarial Stress Test (Risk-Aware Engine)")
    print("="*60)
    
    config = load_config(project_root / "configs" / "phase1.yaml")
    
    # Inject adversarial parameters into evaluation dictionary
    if "evaluation" not in config:
        config["evaluation"] = {}
    config["evaluation"]["adversarial_accidents"] = True
    config["evaluation"]["sensor_noise"] = True
    config["evaluation"]["num_episodes"] = 1
    
    if "output" not in config:
        config["output"] = {}
    config["output"]["final_model_path"] = str(project_root / "best_model_stage_2.zip")
    
    print("\n[!] Triggering Severe Validation Protocols:")
    print("    -> 5 Phantom Crashes at Step 500")
    print("    -> 10% Observation Sensor Masking")
    print("-" * 60)
    
    results = {}
    
    # 1. Evaluate MAPPO-STGNN (Risk-Aware)
    print("Evaluating MAPPO-STGNN (Ours) under Adversarial Stress...")
    our_results = evaluate_model(config, "PPO")
    results["MAPPO-STGNN"] = our_results
    
    # 2. Evaluate NSTLight Baseline (Fallback representing 2025 traditional routing)
    print("Evaluating NSTLight under Adversarial Stress...")
    nst_results = evaluate_model(config, "NSTLight")
    results["nstlight"] = nst_results
    
    # Dump metrics to output json
    out_dir = project_root / "outputs" / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "adversarial_benchmark.json"
    
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "="*60)
    print(f"[OK] Adversarial Stress Test Complete. Degradation matrix saved to {out_file}")
    
    # Log Degradation Differences intuitively
    our_wt = our_results["mean_waiting_time"]
    nst_wt = nst_results["mean_waiting_time"]
    diff = ((nst_wt - our_wt) / max(1, nst_wt)) * 100
    print(f"\n[SOTA METRIC] MAPPO-STGNN reduced accident waiting time by {diff:.1f}% compared to baseline NSTLight!")
    print("="*60)

if __name__ == "__main__":
    main()
