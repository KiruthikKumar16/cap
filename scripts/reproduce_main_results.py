import argparse
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import List


ROOT = Path(__file__).resolve().parent.parent


def _run_step(name: str, command: List[str]) -> str:
    start = perf_counter()
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    elapsed = perf_counter() - start
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    ok = proc.returncode == 0
    if not ok:
        raise RuntimeError(f"[{name}] failed ({elapsed:.1f}s)\n{output}")
    return f"[OK] {name} ({elapsed:.1f}s)"


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command reproducibility runner")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--checkpoint", type=str, default="marl_ppo_traffic.zip")
    args = parser.parse_args()
    checkpoint_path = ROOT / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("=" * 72)
    print("Reproducibility Pipeline")
    print(f"Mode: {args.mode}")
    print("=" * 72)

    if args.mode == "quick":
        steps = [
            ("Python compile check", [sys.executable, "-m", "compileall", "-q", "src", "scripts"]),
            ("Setup smoke test", [sys.executable, "scripts/test_setup.py"]),
            ("Phase 1 smoke evaluation", [sys.executable, "scripts/test_phase1.py"]),
            ("Phase 2 anomaly smoke evaluation", [sys.executable, "scripts/test_phase2.py"]),
            ("Phase 3 integration smoke test", [sys.executable, "scripts/test_phase3_integration.py"]),
            ("Phase 3 stress smoke test", [sys.executable, "scripts/test_phase3.py"]),
            ("Publication artifacts", [sys.executable, "scripts/generate_publication_artifacts.py", "--mode", "quick"]),
        ]
    else:
        steps = [
            ("Python compile check", [sys.executable, "-m", "compileall", "-q", "src", "scripts"]),
            ("Environment check", [sys.executable, "scripts/check_sumo.py"]),
            ("Phase 1 training", [sys.executable, "src/phase1/train_marl.py"]),
            (
                "Benchmark comparison",
                [
                    sys.executable,
                    "scripts/run_benchmarks.py",
                    "--config",
                    "configs/phase1.yaml",
                    "--checkpoint",
                    args.checkpoint,
                    "--episodes",
                    "3",
                ],
            ),
            (
                "Detailed evaluation summary",
                [
                    sys.executable,
                    "src/phase1/evaluate.py",
                    "--config",
                    "configs/phase1.yaml",
                    "--checkpoint",
                    args.checkpoint,
                    "--episodes",
                    "50",
                    "--fixed-time",
                    "--random",
                    "--require-sumo",
                    "--save-summary",
                    "outputs/phase1/evaluation_summary.json",
                ],
            ),
            (
                "Adversarial stress benchmark",
                [
                    sys.executable,
                    "scripts/accident_injection.py",
                    "--config",
                    "configs/phase1.yaml",
                    "--checkpoint",
                    args.checkpoint,
                    "--episodes",
                    "3",
                    "--sensor-noise-rate",
                    "0.10",
                ],
            ),
            ("Statistical summary table", [sys.executable, "scripts/generate_statistical_tables.py"]),
            ("Publication artifacts", [sys.executable, "scripts/generate_publication_artifacts.py", "--mode", "full"]),
        ]

    for name, cmd in steps:
        print(_run_step(name, cmd))

    print("=" * 72)
    print("Pipeline completed. See results/ for generated artifacts.")
    print("=" * 72)


if __name__ == "__main__":
    main()
