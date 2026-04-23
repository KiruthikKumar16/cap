import argparse
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import List, Tuple


ROOT = Path(__file__).resolve().parent.parent


def _run_step(name: str, command: List[str], allow_fail: bool = True) -> Tuple[bool, str]:
    start = perf_counter()
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    elapsed = perf_counter() - start
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    ok = proc.returncode == 0
    if not ok and not allow_fail:
        raise RuntimeError(f"[{name}] failed ({elapsed:.1f}s)\n{output}")
    if not ok:
        print(f"\n[WARN] {name} failed ({elapsed:.1f}s). Continuing.\n")
        print(output[:2000])
    return ok, f"{name}: {'OK' if ok else 'FAILED (allowed)'} ({elapsed:.1f}s)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run publication suite end-to-end")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--checkpoint", default="outputs/phase1/dqn_traffic_final.zip")
    args = parser.parse_args()

    episodes = "1" if args.mode == "quick" else "3"
    detailed_eps = "10" if args.mode == "quick" else "50"

    steps: List[Tuple[str, List[str], bool]] = [
        ("SUMO check", [sys.executable, "scripts/check_sumo.py"], True),
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
                episodes,
            ],
            True,
        ),
        (
            "Detailed evaluation",
            [
                sys.executable,
                "src/phase1/evaluate.py",
                "--config",
                "configs/phase1.yaml",
                "--checkpoint",
                args.checkpoint,
                "--episodes",
                detailed_eps,
                "--fixed-time",
                "--random",
                "--save-summary",
                "outputs/phase1/evaluation_summary.json",
            ],
            True,
        ),
        (
            "Ablation study",
            [sys.executable, "scripts/run_ablation_study.py"],
            True,
        ),
        (
            "Stress benchmark",
            [
                sys.executable,
                "scripts/accident_injection.py",
                "--config",
                "configs/phase1.yaml",
                "--checkpoint",
                args.checkpoint,
                "--episodes",
                episodes,
                "--sensor-noise-rate",
                "0.10",
            ],
            True,
        ),
        (
            "Generalization benchmark",
            [sys.executable, "scripts/evaluate_generalization.py"],
            True,
        ),
        (
            "Latency benchmark",
            [sys.executable, "scripts/latency_benchmark.py", "--gpu"],
            True,
        ),
        ("Statistical tables", [sys.executable, "scripts/generate_statistical_tables.py"], True),
        (
            "Publication artifacts",
            [sys.executable, "scripts/generate_publication_artifacts.py", "--mode", args.mode],
            False,
        ),
    ]

    print("=" * 72)
    print("Publication Suite")
    print(f"Mode: {args.mode}")
    print("=" * 72)
    for name, cmd, allow_fail in steps:
        _, line = _run_step(name, cmd, allow_fail=allow_fail)
        print(line)
    print("=" * 72)
    print("Suite complete. Review generated artifacts under results/.")
    print("=" * 72)


if __name__ == "__main__":
    main()
