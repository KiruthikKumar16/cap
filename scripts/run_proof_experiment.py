#!/usr/bin/env python3
"""Run the first proof-of-capability experiment.

This is the command to use before writing papers or approaching external
partners. It intentionally produces evidence metadata and refuses to run when
required files are missing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def _run(command: list[str]) -> None:
    print(f"[RUN] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _write_manifest(args: argparse.Namespace) -> Path:
    benchmark_path = ROOT / "outputs" / "benchmark_results.json"
    summary_path = ROOT / "results" / "summary.md"
    benchmark = _load_json(benchmark_path)
    metadata = benchmark.get("artifact_metadata", {}) if isinstance(benchmark, dict) else {}

    manifest = {
        "artifact_type": "proof_experiment_manifest",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "seeds": args.seeds,
        "benchmark_results": str(benchmark_path.relative_to(ROOT)),
        "publication_summary": str(summary_path.relative_to(ROOT)),
        "benchmark_metadata": metadata,
        "honesty_note": (
            "This proof run is simulation evidence only. It is not real-world field "
            "evidence and must not be described as a deployed traffic-control result."
        ),
    }

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run simulation proof experiment and reporting pipeline.")
    parser.add_argument("--config", default="configs/phase1.yaml", help="SUMO/evaluation config path.")
    parser.add_argument("--checkpoint", default="marl_ppo_traffic.zip", help="Compatible PPO checkpoint.")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per seed.")
    parser.add_argument("--seeds", type=int, default=3, help="Number of configured seeds to run.")
    parser.add_argument("--output", default="results/proof_manifest.json", help="Manifest output path.")
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Only regenerate artifacts/manifest from existing benchmark_results.json.",
    )
    args = parser.parse_args()

    config_path = ROOT / args.config
    checkpoint_path = ROOT / args.checkpoint
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not args.skip_benchmark and not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    _run([sys.executable, "-m", "compileall", "-q", "src", "scripts"])
    _run([sys.executable, "scripts/ci_validate_evidence.py"])

    if not args.skip_benchmark:
        _run(
            [
                sys.executable,
                "scripts/run_benchmarks.py",
                "--config",
                args.config,
                "--checkpoint",
                args.checkpoint,
                "--episodes",
                str(args.episodes),
                "--seeds",
                str(args.seeds),
            ]
        )

    _run([sys.executable, "scripts/generate_publication_artifacts.py", "--mode", "quick"])
    manifest_path = _write_manifest(args)
    print(f"[OK] Proof manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
