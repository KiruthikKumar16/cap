import sys
from pathlib import Path
import json

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.phase1.train_rl import load_config
from src.phase1.evaluate import evaluate_model
import torch

def map_generalization():
    print("="*60)
    print("Phase 4: SOTA Zero-Shot Routing Generalization Matrix")
    print("="*60)
    
    config = load_config(project_root / "configs" / "phase1.yaml")
    
    # Define test geometries mapping (Base Map A vs Target Map B)
    geometries = {
        "Map_A_Training_Grid": "data/raw/grid_10x10",
        "Map_B_Bengaluru_OSM": "data/raw/bengaluru_osm",
        "Map_C_Fallback_Scaling": "data/raw/grid_6x6"
    }
    
    # Hardcode evaluation model PPO
    config["output"] = {"final_model_path": str(project_root / "best_model_stage_2.zip")}
    config["evaluation"] = {"num_episodes": 1, "adversarial_accidents": False, "sensor_noise": False}
    
    results = {}
    
    for map_name, map_prefix in geometries.items():
        print(f"\n[Validation] Targeting Zero-Shot execution on {map_name} ...")
        
        net_path = project_root / f"{map_prefix}.net.xml"
        rou_path = project_root / f"{map_prefix}.rou.xml"
        
        if not net_path.exists():
            print(f"  [!] Missing Geometry Array: {net_path}")
            print(f"      (For Bengaluru test, please run osmWebWizard.py and drop bengaluru_osm.net.xml into data/raw/)")
            continue
            
        # Hook maps to active environment configuration
        config["sumo"]["net_file"] = str(net_path)
        config["sumo"]["route_file"] = str(rou_path)
        
        print(f"  -> Routing Graph: {net_path.name}")
        metrics = evaluate_model(config, "PPO")
        results[map_name] = metrics
        
    out_dir = project_root / "outputs" / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "zero_shot_generalization.json"
    
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "="*60)
    print(f"[OK] Generalization Metrics locked and exported to {out_file}")
    
    # Calculate generalization penalty 
    if "Map_A_Training_Grid" in results and "Map_C_Fallback_Scaling" in results:
        base_queue = results["Map_A_Training_Grid"]["mean_queue_length"]
        target_queue = results["Map_C_Fallback_Scaling"]["mean_queue_length"]
        degradation = ((target_queue - base_queue) / max(1, base_queue)) * 100
        print(f"\n[SOTA RESILIENCE] ST-GNN PPO Topology Generalization Penalty: {degradation:.2f}% queue divergence on unseen map.")
    
    print("="*60)

if __name__ == "__main__":
    map_generalization()
