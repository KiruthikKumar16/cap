
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
from src.phase1.evaluate import collect_action_trace, evaluate_model, generate_eval_gif
print("Debug: evaluate_model imported.", flush=True)


def _trace_similarity(trace_a, trace_b):
    vecs_a = trace_a.get("trace_vectors", []) if isinstance(trace_a, dict) else []
    vecs_b = trace_b.get("trace_vectors", []) if isinstance(trace_b, dict) else []
    common = min(len(vecs_a), len(vecs_b))
    if common == 0:
        return None
    same = sum(1 for idx in range(common) if vecs_a[idx] == vecs_b[idx])
    return round(same / common, 4)


def _media_slug(model_name: str) -> str:
    return (
        model_name.lower()
        .replace("+", "plus")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

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
    action_diagnostics = {}
    media_manifest = {}
    media_dir = Path("outputs/dashboard_media")
    media_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate our model
    print("[MODEL_START] MAPPO-STGNN", flush=True)
    print("Evaluating MAPPO-STGNN (Ours)...", flush=True)
    results["MAPPO-STGNN"] = evaluate_model(config, "PPO")
    action_diagnostics["MAPPO-STGNN"] = collect_action_trace(config, "PPO", config.get("evaluation", {}).get("action_trace_steps", 32))
    media_manifest["MAPPO-STGNN"] = generate_eval_gif(config, "PPO", media_dir / f"{_media_slug('MAPPO-STGNN')}.gif")
    print(f"[VISUAL_READY] MAPPO-STGNN {media_manifest['MAPPO-STGNN']['gif_path']}", flush=True)
    print("[MODEL_DONE] MAPPO-STGNN", flush=True)

    # Evaluate SOTA Heuristic: MaxPressure
    print("Evaluating MaxPressure...", flush=True)
    results["MaxPressure"] = evaluate_model(config, "MaxPressure")
    action_diagnostics["MaxPressure"] = collect_action_trace(config, "MaxPressure", config.get("evaluation", {}).get("action_trace_steps", 32))

    # Evaluate SOTA MARL: PressLight
    print("Evaluating PressLight...", flush=True)
    results["PressLight"] = evaluate_model(config, "PressLight")
    action_diagnostics["PressLight"] = collect_action_trace(config, "PressLight", config.get("evaluation", {}).get("action_trace_steps", 32))

    # Evaluate SOTA GNN: CoLight
    print("[MODEL_START] CoLight", flush=True)
    print("Evaluating CoLight...", flush=True)
    results["CoLight"] = evaluate_model(config, "CoLight")
    action_diagnostics["CoLight"] = collect_action_trace(config, "CoLight", config.get("evaluation", {}).get("action_trace_steps", 32))
    media_manifest["CoLight"] = generate_eval_gif(config, "CoLight", media_dir / f"{_media_slug('CoLight')}.gif")
    print(f"[VISUAL_READY] CoLight {media_manifest['CoLight']['gif_path']}", flush=True)
    print("[MODEL_DONE] CoLight", flush=True)

    # Evaluate SOTA GNN: NSTLight
    print("[MODEL_START] NSTLight", flush=True)
    print("Evaluating NSTLight...", flush=True)
    results["NSTLight"] = evaluate_model(config, "NSTLight")
    action_diagnostics["NSTLight"] = collect_action_trace(config, "NSTLight", config.get("evaluation", {}).get("action_trace_steps", 32))
    media_manifest["NSTLight"] = generate_eval_gif(config, "NSTLight", media_dir / f"{_media_slug('NSTLight')}.gif")
    print(f"[VISUAL_READY] NSTLight {media_manifest['NSTLight']['gif_path']}", flush=True)
    print("[MODEL_DONE] NSTLight", flush=True)

    # Keep fixed-time for hardware-independent sanity check.
    print("Evaluating Fixed-Time...", flush=True)
    results["FixedTime"] = evaluate_model(config, "FixedTime")
    action_diagnostics["FixedTime"] = collect_action_trace(config, "FixedTime", config.get("evaluation", {}).get("action_trace_steps", 32))

    # Evaluate Random baseline
    print("Evaluating Random...", flush=True)
    results["Random"] = evaluate_model(config, "Random")
    action_diagnostics["Random"] = collect_action_trace(config, "Random", config.get("evaluation", {}).get("action_trace_steps", 32))

    similarity = {}
    reference_models = ["MAPPO-STGNN", "FixedTime"]
    for ref in reference_models:
        if ref not in action_diagnostics:
            continue
        similarity[ref] = {}
        for model_name, diag in action_diagnostics.items():
            similarity[ref][model_name] = _trace_similarity(action_diagnostics[ref], diag)

    # Append latency outputs to benchmark summary when available.
    latency_path = Path("outputs/latency/inference_latency.json")
    if latency_path.exists():
        with open(latency_path, "r", encoding="utf-8") as f:
            results["latency_ms_per_step"] = json.load(f)

    results["action_diagnostics"] = action_diagnostics
    results["action_similarity"] = similarity
    results["dashboard_media"] = media_manifest

    # Save results
    output_path = Path("outputs/benchmark_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    diagnostics_path = Path("outputs/action_diagnostics.json")
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump({"action_diagnostics": action_diagnostics, "action_similarity": similarity}, f, indent=4)

    media_path = Path("outputs/dashboard_media.json")
    with open(media_path, "w", encoding="utf-8") as f:
        json.dump(media_manifest, f, indent=4)

    print(f"Benchmark results saved to {output_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOTA benchmarks.")
    parser.add_argument("--config", type=str, default="configs/phase1.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained model zip")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per baseline")
    args = parser.parse_args()
    run_benchmarks(args.config, args.checkpoint, args.episodes)
