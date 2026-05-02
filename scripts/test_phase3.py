"""
Smoke test Phase 3 anomaly-aware benchmark wiring with a short SUMO horizon.
"""

import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
config_src = ROOT / "configs" / "phase1.yaml"
config_test = ROOT / "configs" / "phase3_smoke.yaml"

print("\n" + "=" * 50)
print("PHASE 3 SMOKE TEST: Anomaly-Aware Traffic Routing")
print("=" * 50)

with config_src.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

config.setdefault("sumo", {})["simulation_steps"] = 120
config.setdefault("evaluation", {})["action_trace_steps"] = 8
phase3 = config.setdefault("phase3", {})
phase3["enable_anomaly_awareness"] = True
phase3["anomaly_model_path"] = "outputs/phase2/st_gnn_anomaly_detector.pt"
phase3["anomaly_threshold"] = 0.5

try:
    with config_test.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    subprocess.run(
        [
            sys.executable,
            "scripts/accident_injection.py",
            "--config",
            str(config_test.relative_to(ROOT)),
            "--checkpoint",
            "marl_ppo_traffic.zip",
            "--episodes",
            "1",
            "--sensor-noise-rate",
            "0.10",
            "--mappo-only",
        ],
        cwd=ROOT,
        check=True,
    )
finally:
    if config_test.exists():
        config_test.unlink()
