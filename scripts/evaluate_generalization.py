import sys
from pathlib import Path
import json
import copy
import argparse

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.phase1.train_rl import load_config
from src.phase1.evaluate import evaluate_model

def _drop_pct(source: float, target: float, lower_is_better: bool) -> float:
    if source == 0:
        return 0.0
    if lower_is_better:
        return ((target - source) / abs(source)) * 100.0
    return ((source - target) / abs(source)) * 100.0


def map_generalization(config_path: str, checkpoint: str, episodes: int):
    print("="*60)
    print("Phase 4: Baseline Zero-Shot Routing Generalization Matrix")
    print("="*60)
    
    config = load_config(project_root / config_path)
    
    # Formal Bengaluru zero-shot protocol: Map A (train distribution) vs Map B (Bengaluru OSM).
    geometries = {
        "Map_A_Training_Grid_5x5": "data/raw/grid_5x5",
        "Map_B_Large_Grid_10x10": "data/raw/grid_10x10",
    }
    
    # Use user-provided checkpoint/config.
    cp_path = Path(checkpoint)
    if not cp_path.is_absolute():
        cp_path = project_root / cp_path
    config["output"] = {"final_model_path": str(cp_path)}
    config["evaluation"] = {"num_episodes": episodes, "adversarial_accidents": False, "sensor_noise": False}
    
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
        map_cfg = copy.deepcopy(config)
        map_cfg["sumo"]["net_file"] = str(net_path)
        map_cfg["sumo"]["route_file"] = str(rou_path)
        
        print(f"  -> Routing Graph: {net_path.name}")
        metrics = evaluate_model(map_cfg, "PPO")
        results[map_name] = metrics
        
    out_dir = project_root / "outputs" / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "zero_shot_generalization.json"
    
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "="*60)
    print(f"[OK] Generalization Metrics locked and exported to {out_file}")
    
    # Calculate Map A -> Map B zero-shot performance drop.
    map_a = "Map_A_Training_Grid_5x5"
    map_b = "Map_B_Large_Grid_10x10"
    if map_a in results and map_b in results:
        drop = {
            "throughput_drop_pct": _drop_pct(results[map_a]["mean_throughput"], results[map_b]["mean_throughput"], lower_is_better=False),
            "waiting_time_increase_pct": _drop_pct(results[map_a]["mean_waiting_time"], results[map_b]["mean_waiting_time"], lower_is_better=True),
            "queue_length_increase_pct": _drop_pct(results[map_a]["mean_queue_length"], results[map_b]["mean_queue_length"], lower_is_better=True),
        }
        results["map_a_to_b_drop_pct"] = {k: round(v, 3) for k, v in drop.items()}
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(
            "\n[Baseline RESILIENCE] Zero-shot Map A->B drop | "
            f"throughput: {drop['throughput_drop_pct']:.2f}% | "
            f"waiting: {drop['waiting_time_increase_pct']:.2f}% | "
            f"queue: {drop['queue_length_increase_pct']:.2f}%"
        )
    
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map-transfer generalization benchmark")
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/phase1/dqn_traffic_final.zip")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()
    map_generalization(args.config, args.checkpoint, args.episodes)
