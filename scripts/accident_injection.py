import argparse
import copy
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.phase1.train_rl import load_config
from src.phase1.evaluate import evaluate_model


def _safe_drop(base: float, stressed: float, lower_is_better: bool) -> float:
    if base == 0:
        return 0.0
    if lower_is_better:
        return ((stressed - base) / abs(base)) * 100.0
    return ((base - stressed) / abs(base)) * 100.0


def main():
    parser = argparse.ArgumentParser(description="Adversarial accident injection benchmark")
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    parser.add_argument("--checkpoint", type=str, default="marl_ppo_traffic.zip")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--sensor-noise-rate", type=float, default=0.10)
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 3: Adversarial Stress Test (Risk-Aware Engine)")
    print("=" * 60)

    config = load_config(project_root / args.config)
    config.setdefault("evaluation", {})
    config.setdefault("output", {})
    config["evaluation"]["num_episodes"] = args.episodes
    config["output"]["final_model_path"] = str(project_root / args.checkpoint)

    normal_cfg = copy.deepcopy(config)
    normal_cfg["evaluation"].update(
        {"adversarial_accidents": False, "sensor_noise": False, "sensor_noise_rate": 0.0}
    )
    stress_cfg = copy.deepcopy(config)
    stress_cfg["evaluation"].update(
        {"adversarial_accidents": True, "sensor_noise": True, "sensor_noise_rate": args.sensor_noise_rate}
    )

    print("\n[!] Stress protocol:")
    print("    -> Simulated crashes freeze vehicles at step 500")
    print(f"    -> {int(args.sensor_noise_rate * 100)}% sensor failure mask on MAPPO observations")
    print("-" * 60)

    results = {"normal": {}, "stress": {}, "degradation_limits_pct": {}}

    print("Evaluating MAPPO-STGNN (normal)...")
    results["normal"]["mappo"] = evaluate_model(normal_cfg, "PPO")
    print("Evaluating MAPPO-STGNN (stress)...")
    results["stress"]["mappo"] = evaluate_model(stress_cfg, "PPO")

    # NSTLight stress test excludes sensor masking wrapper to keep requirement MAPPO-specific.
    nst_stress_cfg = copy.deepcopy(stress_cfg)
    nst_stress_cfg["evaluation"]["sensor_noise"] = False
    print("Evaluating NSTLight (normal)...")
    results["normal"]["nstlight"] = evaluate_model(normal_cfg, "NSTLight")
    print("Evaluating NSTLight (stress)...")
    results["stress"]["nstlight"] = evaluate_model(nst_stress_cfg, "NSTLight")

    for model in ("mappo", "nstlight"):
        base = results["normal"][model]
        stressed = results["stress"][model]
        results["degradation_limits_pct"][model] = {
            "throughput_drop_pct": round(_safe_drop(base["mean_throughput"], stressed["mean_throughput"], lower_is_better=False), 3),
            "waiting_time_increase_pct": round(_safe_drop(base["mean_waiting_time"], stressed["mean_waiting_time"], lower_is_better=True), 3),
            "queue_length_increase_pct": round(_safe_drop(base["mean_queue_length"], stressed["mean_queue_length"], lower_is_better=True), 3),
        }

    out_dir = project_root / "outputs" / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "adversarial_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"[OK] Saved adversarial degradation report to {out_file}")
    print(
        f"MAPPO waiting-time increase: {results['degradation_limits_pct']['mappo']['waiting_time_increase_pct']:.2f}% | "
        f"NSTLight waiting-time increase: {results['degradation_limits_pct']['nstlight']['waiting_time_increase_pct']:.2f}%"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
