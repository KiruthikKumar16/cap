
"""
Benchmark Script

This script runs evaluations for MAPPO vs NSTLight baseline
and saves results for comparison (with latency summary if available).
"""

print("Debug: Starting run_benchmarks.py...", flush=True)
import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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

CONTROLLERS = ["MAPPO-STGNN", "MaxPressure", "PressLight", "CoLight", "NSTLight", "FixedTime", "Random"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def _aggregate_controller_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    metric_names = [
        "mean_reward",
        "mean_throughput",
        "mean_travel_time",
        "mean_waiting_time",
        "mean_queue_length",
    ]
    for metric in metric_names:
        values = [float(run[metric]) for run in runs if metric in run]
        if not values:
            continue
        mean_value = sum(values) / len(values)
        aggregate[metric] = mean_value
        if len(values) > 1:
            variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
            aggregate[f"{metric}_std"] = variance ** 0.5
        else:
            aggregate[f"{metric}_std"] = 0.0
    aggregate["num_seed_runs"] = len(runs)
    aggregate["per_seed"] = runs
    return aggregate


def _base_metadata(config: Dict[str, Any], config_path: str, checkpoint: str, episodes: int, seeds: List[int]) -> Dict[str, Any]:
    checkpoint_path = ROOT / checkpoint
    evidence_status = "simulation_smoke" if episodes < 5 or len(seeds) < 3 else "simulation_benchmark_candidate"
    metadata: Dict[str, Any] = {
        "artifact_type": "benchmark_run",
        "evidence_status": evidence_status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config_path": config_path,
        "checkpoint": checkpoint,
        "checkpoint_exists": checkpoint_path.exists(),
        "episodes_per_seed": episodes,
        "seeds": seeds,
        "controllers": CONTROLLERS,
        "scenario": {
            "net_file": config.get("sumo", {}).get("net_file"),
            "route_file": config.get("sumo", {}).get("route_file"),
            "config_file": config.get("sumo", {}).get("config_file"),
            "simulation_steps": config.get("sumo", {}).get("simulation_steps"),
        },
    }
    if checkpoint_path.exists():
        metadata["checkpoint_sha256"] = _sha256(checkpoint_path)
    return metadata


def _run_controller_set(config: Dict[str, Any], write_media: bool) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    results: Dict[str, Any] = {}
    action_diagnostics: Dict[str, Any] = {}
    media_manifest: Dict[str, Any] = {}
    media_dir = Path("outputs/dashboard_media")
    media_dir.mkdir(parents=True, exist_ok=True)

    print("[MODEL_START] MAPPO-STGNN", flush=True)
    print("Evaluating MAPPO-STGNN (Ours)...", flush=True)
    results["MAPPO-STGNN"] = evaluate_model(config, "PPO")
    action_diagnostics["MAPPO-STGNN"] = collect_action_trace(config, "PPO", config.get("evaluation", {}).get("action_trace_steps", 32))
    if write_media:
        media_manifest["MAPPO-STGNN"] = generate_eval_gif(config, "PPO", media_dir / f"{_media_slug('MAPPO-STGNN')}.gif")
        print(f"[VISUAL_READY] MAPPO-STGNN {media_manifest['MAPPO-STGNN']['gif_path']}", flush=True)
    print("[MODEL_DONE] MAPPO-STGNN", flush=True)

    print("Evaluating MaxPressure...", flush=True)
    results["MaxPressure"] = evaluate_model(config, "MaxPressure")
    action_diagnostics["MaxPressure"] = collect_action_trace(config, "MaxPressure", config.get("evaluation", {}).get("action_trace_steps", 32))

    print("Evaluating PressLight...", flush=True)
    results["PressLight"] = evaluate_model(config, "PressLight")
    action_diagnostics["PressLight"] = collect_action_trace(config, "PressLight", config.get("evaluation", {}).get("action_trace_steps", 32))

    print("[MODEL_START] CoLight", flush=True)
    print("Evaluating CoLight...", flush=True)
    results["CoLight"] = evaluate_model(config, "CoLight")
    action_diagnostics["CoLight"] = collect_action_trace(config, "CoLight", config.get("evaluation", {}).get("action_trace_steps", 32))
    if write_media:
        media_manifest["CoLight"] = generate_eval_gif(config, "CoLight", media_dir / f"{_media_slug('CoLight')}.gif")
        print(f"[VISUAL_READY] CoLight {media_manifest['CoLight']['gif_path']}", flush=True)
    print("[MODEL_DONE] CoLight", flush=True)

    print("[MODEL_START] NSTLight", flush=True)
    print("Evaluating NSTLight...", flush=True)
    results["NSTLight"] = evaluate_model(config, "NSTLight")
    action_diagnostics["NSTLight"] = collect_action_trace(config, "NSTLight", config.get("evaluation", {}).get("action_trace_steps", 32))
    if write_media:
        media_manifest["NSTLight"] = generate_eval_gif(config, "NSTLight", media_dir / f"{_media_slug('NSTLight')}.gif")
        print(f"[VISUAL_READY] NSTLight {media_manifest['NSTLight']['gif_path']}", flush=True)
    print("[MODEL_DONE] NSTLight", flush=True)

    print("Evaluating Fixed-Time...", flush=True)
    results["FixedTime"] = evaluate_model(config, "FixedTime")
    action_diagnostics["FixedTime"] = collect_action_trace(config, "FixedTime", config.get("evaluation", {}).get("action_trace_steps", 32))

    print("Evaluating Random...", flush=True)
    results["Random"] = evaluate_model(config, "Random")
    action_diagnostics["Random"] = collect_action_trace(config, "Random", config.get("evaluation", {}).get("action_trace_steps", 32))

    return results, action_diagnostics, media_manifest


def _trace_similarity_matrix(action_diagnostics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    similarity: Dict[str, Dict[str, Any]] = {}
    reference_models = ["MAPPO-STGNN", "FixedTime"]
    for ref in reference_models:
        if ref not in action_diagnostics:
            continue
        similarity[ref] = {}
        for model_name, diag in action_diagnostics.items():
            similarity[ref][model_name] = _trace_similarity(action_diagnostics[ref], diag)
    return similarity


def run_benchmarks(config_path: str, checkpoint: str, episodes: int, seeds: List[int]):
    """Run all benchmarks and save self-describing results."""
    print(f"Debug: Entering run_benchmarks with config={config_path}, episodes={episodes}, seeds={seeds}", flush=True)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    checkpoint_path = ROOT / checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if "evaluation" not in config:
        config["evaluation"] = {}
    config["evaluation"]["num_episodes"] = episodes
    
    if "output" not in config:
        config["output"] = {}
    config["output"]["final_model_path"] = checkpoint

    per_controller_runs: Dict[str, List[Dict[str, Any]]] = {name: [] for name in CONTROLLERS}
    all_action_diagnostics: Dict[str, Any] = {}
    media_manifest: Dict[str, Any] = {}

    for seed_index, seed in enumerate(seeds):
        seed_config = copy.deepcopy(config)
        seed_config.setdefault("experiment", {})["seed"] = seed
        seed_config.setdefault("sumo", {})["seed"] = seed
        print(f"[SEED_START] {seed}", flush=True)
        seed_results, seed_diagnostics, seed_media = _run_controller_set(seed_config, write_media=seed_index == 0)
        for controller_name, metrics in seed_results.items():
            if controller_name in per_controller_runs:
                per_controller_runs[controller_name].append({"seed": seed, **metrics})
        all_action_diagnostics[str(seed)] = seed_diagnostics
        media_manifest.update(seed_media)
        print(f"[SEED_DONE] {seed}", flush=True)

    results: Dict[str, Any] = {
        controller_name: _aggregate_controller_runs(runs)
        for controller_name, runs in per_controller_runs.items()
        if runs
    }

    first_seed = str(seeds[0])
    similarity = _trace_similarity_matrix(all_action_diagnostics.get(first_seed, {}))

    # Append latency outputs to benchmark summary when available.
    latency_path = Path("outputs/latency/inference_latency.json")
    if latency_path.exists():
        with open(latency_path, "r", encoding="utf-8") as f:
            results["latency_ms_per_step"] = json.load(f)

    results["artifact_metadata"] = _base_metadata(config, config_path, checkpoint, episodes, seeds)
    results["action_diagnostics"] = all_action_diagnostics
    results["action_similarity"] = similarity
    results["dashboard_media"] = media_manifest

    # Save results
    output_path = Path("outputs/benchmark_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    diagnostics_path = Path("outputs/action_diagnostics.json")
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump({"action_diagnostics": all_action_diagnostics, "action_similarity": similarity}, f, indent=4)

    media_path = Path("outputs/dashboard_media.json")
    with open(media_path, "w", encoding="utf-8") as f:
        json.dump(media_manifest, f, indent=4)

    print(f"Benchmark results saved to {output_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOTA benchmarks.")
    parser.add_argument("--config", type=str, default="configs/phase1.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained model zip")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per baseline")
    parser.add_argument("--seeds", type=int, default=1, help="Number of config evaluation seeds to run")
    parser.add_argument("--seed-values", type=int, nargs="*", default=None, help="Explicit seed values to run")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    configured_seeds = cfg.get("evaluation", {}).get("seeds", [42])
    if args.seed_values:
        seed_values = args.seed_values
    else:
        seed_values = configured_seeds[: args.seeds] if isinstance(configured_seeds, list) else [42]
    run_benchmarks(args.config, args.checkpoint, args.episodes, seed_values)
