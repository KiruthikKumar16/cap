"""
Smoke test Phase 1 benchmark wiring with a short SUMO horizon.

This test checks that benchmark execution, checkpoint loading, and result
serialization work. It is not a publication-quality evaluation run.
"""

import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
config_src = ROOT / "configs" / "phase1.yaml"
config_test = ROOT / "configs" / "phase1_smoke.yaml"

print("\n" + "=" * 50)
print("PHASE 1 SMOKE TEST: RL & GNN Benchmarking")
print("=" * 50)

with config_src.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

config.setdefault("sumo", {})["simulation_steps"] = 120
config.setdefault("evaluation", {})["action_trace_steps"] = 8

try:
    with config_test.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    subprocess.run(
        [
            sys.executable,
            "src/phase1/evaluate.py",
            "--config",
            str(config_test.relative_to(ROOT)),
            "--checkpoint",
            "marl_ppo_traffic.zip",
            "--episodes",
            "1",
            "--fixed-time",
            "--random",
            "--save-summary",
            "outputs/phase1/evaluation_summary.json",
        ],
        cwd=ROOT,
        check=True,
    )

    print("\n--- Evaluation Summary JSON Output ---")
    with (ROOT / "outputs" / "phase1" / "evaluation_summary.json").open("r", encoding="utf-8") as f:
        print(json.dumps(json.load(f), indent=2)[:4000])
finally:
    if config_test.exists():
        config_test.unlink()
