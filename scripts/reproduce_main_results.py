import argparse
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import List, Tuple


ROOT = Path(__file__).resolve().parent.parent


def _run_step(name: str, command: List[str], allow_fail: bool = False) -> Tuple[bool, str]:
    start = perf_counter()
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    elapsed = perf_counter() - start
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    ok = proc.returncode == 0
    if not ok and not allow_fail:
        raise RuntimeError(f"[{name}] failed ({elapsed:.1f}s)\n{output}")
    status = "OK" if ok else "FAILED (allowed)"
    return ok, f"[{status}] {name} ({elapsed:.1f}s)"


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command reproducibility runner")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--checkpoint", type=str, default="outputs/phase1/dqn_traffic_final.zip")
    args = parser.parse_args()

    print("=" * 72)
    print("Reproducibility Pipeline")
    print(f"Mode: {args.mode}")
    print("=" * 72)

    steps: List[Tuple[str, List[str], bool]] = [
        ("Environment check", [sys.executable, "scripts/check_sumo.py"], True),
        ("Phase 1 training", [sys.executable, "src/phase1/train_marl.py"], True),
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
                "1" if args.mode == "quick" else "3",
            ],
            True,
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
                "10" if args.mode == "quick" else "50",
                "--fixed-time",
                "--random",
                "--save-summary",
                "outputs/phase1/evaluation_summary.json",
            ],
            True,
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
                "1" if args.mode == "quick" else "3",
                "--sensor-noise-rate",
                "0.10",
            ],
            True,
        ),
        (
            "Statistical summary table",
            [sys.executable, "scripts/generate_statistical_tables.py"],
            True,
        ),
        (
            "Publication artifacts",
            [sys.executable, "scripts/generate_publication_artifacts.py", "--mode", args.mode],
            False,
        ),
    ]

    for name, cmd, allow_fail in steps:
        _, line = _run_step(name, cmd, allow_fail=allow_fail)
        print(line)

    print("=" * 72)
    print("Pipeline completed. See results/ for generated artifacts.")
    print("=" * 72)


if __name__ == "__main__":
    main()
