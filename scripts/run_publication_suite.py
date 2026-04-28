import argparse
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import List, Tuple
import yaml


ROOT = Path(__file__).resolve().parent.parent


def _run_step(name: str, command: List[str], allow_fail: bool = True) -> Tuple[bool, str]:
    start = perf_counter()
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    lines: List[str] = []
    if proc.stdout is None:
        raise RuntimeError(f"[{name}] failed to attach to subprocess output.")

    print(f"\n>>> {name}", flush=True)
    for line in proc.stdout:
        line = line.rstrip()
        lines.append(line)
        print(line, flush=True)

    return_code = proc.wait()
    elapsed = perf_counter() - start
    output = "\n".join(lines).strip()
    ok = return_code == 0
    if not ok and not allow_fail:
        raise RuntimeError(f"[{name}] failed ({elapsed:.1f}s)\n{output}")
    if not ok:
        print(f"\n[WARN] {name} failed ({elapsed:.1f}s). Continuing.\n")
        print(output[:2000])
    return ok, f"{name}: {'OK' if ok else 'FAILED (allowed)'} ({elapsed:.1f}s)"


def _ensure_route_file(config_path: str) -> None:
    cfg_path = (ROOT / config_path).resolve()
    if not cfg_path.exists():
        return
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    route_file = cfg.get("sumo", {}).get("route_file")
    if not route_file:
        return
    route_path = (ROOT / route_file).resolve()
    if route_path.exists():
        return
    # Attempt to generate standard 5x5 medium scenario when route file is missing.
    print(f"[INFO] Missing route file: {route_file}. Generating fallback scenario...")
    subprocess.run(
        [sys.executable, "scripts/create_sumo_scenario.py", "--grid-size", "5", "--demand", "medium"],
        cwd=ROOT,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run publication suite end-to-end")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--checkpoint", default="outputs/phase1/dqn_traffic_final.zip")
    parser.add_argument("--benchmark-episodes", type=int, default=None)
    parser.add_argument("--detailed-episodes", type=int, default=None)
    parser.add_argument("--stress-episodes", type=int, default=None)
    parser.add_argument("--latency-device", choices=["gpu", "cpu"], default="gpu")
    args = parser.parse_args()
    _ensure_route_file(args.config)

    episodes = str(args.benchmark_episodes if args.benchmark_episodes is not None else (1 if args.mode == "quick" else 3))
    detailed_eps = str(args.detailed_episodes if args.detailed_episodes is not None else (10 if args.mode == "quick" else 50))
    stress_eps = str(args.stress_episodes if args.stress_episodes is not None else episodes)
    latency_cmd = [sys.executable, "scripts/latency_benchmark.py"]
    if args.latency_device == "gpu":
        latency_cmd.append("--gpu")

    # Keep quick mode genuinely interactive for the Streamlit dashboard.
    if args.mode == "quick":
        steps: List[Tuple[str, List[str], bool]] = [
            ("SUMO check", [sys.executable, "scripts/check_sumo.py"], True),
            (
                "Benchmark comparison",
                [
                    sys.executable,
                    "scripts/run_benchmarks.py",
                    "--config",
                    args.config,
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
                    args.config,
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
                "Stress benchmark",
                [
                    sys.executable,
                    "scripts/accident_injection.py",
                    "--config",
                    args.config,
                    "--checkpoint",
                    args.checkpoint,
                    "--episodes",
                    stress_eps,
                    "--sensor-noise-rate",
                    "0.10",
                ],
                True,
            ),
            ("Statistical tables", [sys.executable, "scripts/generate_statistical_tables.py"], True),
            (
                "Publication artifacts",
                [sys.executable, "scripts/generate_publication_artifacts.py", "--mode", args.mode],
                True,
            ),
        ]
    else:
        steps = [
            ("SUMO check", [sys.executable, "scripts/check_sumo.py"], True),
            (
                "Benchmark comparison",
                [
                    sys.executable,
                    "scripts/run_benchmarks.py",
                    "--config",
                    args.config,
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
                    args.config,
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
                    args.config,
                    "--checkpoint",
                    args.checkpoint,
                    "--episodes",
                    stress_eps,
                    "--sensor-noise-rate",
                    "0.10",
                ],
                True,
            ),
            (
                "Generalization benchmark",
                [
                    sys.executable,
                    "scripts/evaluate_generalization.py",
                    "--config",
                    args.config,
                    "--checkpoint",
                    args.checkpoint,
                    "--episodes",
                    episodes,
                ],
                True,
            ),
            (
                "Latency benchmark",
                latency_cmd,
                True,
            ),
            ("Statistical tables", [sys.executable, "scripts/generate_statistical_tables.py"], True),
            (
                "Publication artifacts",
                [sys.executable, "scripts/generate_publication_artifacts.py", "--mode", args.mode],
                True,
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
