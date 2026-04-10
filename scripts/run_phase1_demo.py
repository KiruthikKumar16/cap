"""
One-command Phase 1 demo: train → evaluate → generate figures for panel.

Usage (from project root, with venv activated):

  python scripts/run_phase1_demo.py

  python scripts/run_phase1_demo.py --quick   # short training (10k steps) for fast demo

Then open outputs/phase1/figures/ to show the panel.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Phase 1 full demo: train, evaluate, generate figures")
    parser.add_argument("--quick", action="store_true", help="Quick demo: 10k training steps only")
    parser.add_argument("--config", type=str, default="configs/phase1.yaml", help="Config file")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    if Path.cwd() != root:
        print(f"[INFO] Changing to project root: {root}")
        import os
        os.chdir(root)

    config = args.config
    py = sys.executable

    # 0) Create SUMO network if missing (default phase1 config uses grid_3x3)
    cfg_path = root / config
    net_rel = "data/raw/grid_3x3.net.xml"
    if cfg_path.is_file():
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            _cfg = yaml.safe_load(f)
        net_rel = _cfg.get("sumo", {}).get("net_file", net_rel)
    net_file = root / net_rel
    if not net_file.exists():
        print("\n" + "=" * 60)
        print("Step 0/3: Creating SUMO grid network (data/raw/)")
        print("=" * 60)
        r0 = subprocess.run([py, "scripts/create_sumo_network.py"], cwd=str(root))
        if r0.returncode != 0:
            print("[WARN] SUMO network creation failed; training may error without net files.")
        else:
            print("[OK] SUMO network ready.")
    else:
        print(f"[OK] SUMO network found: {net_rel}")

    # 1) Train (configs/phase1.yaml is MAPPO / PPO — use train_marl, not train_rl)
    train_cmd = [py, "-m", "src.phase1.train_marl", "--config", config]
    if args.quick:
        train_cmd += ["--total-timesteps", "2048"]
        print("[QUICK] Training for 2,048 steps only (smoke run).")
    print("\n" + "=" * 60)
    print("Step 1/3: Training")
    print("=" * 60)
    r = subprocess.run(train_cmd, cwd=str(root))
    if r.returncode != 0:
        print("[ERROR] Training failed. Exiting.")
        return r.returncode

    # 2) Evaluate (save summary for comparison charts)
    print("\n" + "=" * 60)
    print("Step 2/3: Evaluation (MAPPO vs baselines)")
    print("=" * 60)
    eval_config = args.config if not args.quick else config
    summary_path = root / "outputs" / "phase1" / "evaluation_summary.json"
    eval_ep, eval_seeds = ("1", "1") if args.quick else ("5", "2")
    r = subprocess.run(
        [
            py,
            "-m",
            "src.phase1.evaluate",
            "--config",
            eval_config,
            "--checkpoint",
            "marl_ppo_traffic.zip",
            "--episodes",
            eval_ep,
            "--seeds",
            eval_seeds,
            "--actuated",
            "--save-summary",
            str(summary_path),
        ],
        cwd=str(root),
    )
    if r.returncode != 0:
        print("[WARN] Evaluation failed; continuing to figures.")

    # 3) Generate figures
    print("\n" + "=" * 60)
    print("Step 3/3: Generating figures")
    print("=" * 60)
    r = subprocess.run([py, "scripts/phase1_generate_figures.py"], cwd=str(root))
    if r.returncode != 0:
        print("[WARN] Figure generation had issues.")

    fig_dir = root / "outputs" / "phase1" / "figures"
    print("\n" + "=" * 60)
    print("Done. Figures for your panel:")
    print("=" * 60)
    print(f"  {fig_dir}")
    if fig_dir.exists():
        for f in sorted(fig_dir.glob("*.png")):
            print(f"    - {f.name}")
    print("\nComparison charts (why ours is better):")
    for name in ["phase1_comparison_reward.png", "phase1_comparison_throughput.png", "phase1_comparison_travel_time.png", "phase1_comparison_improvement.png"]:
        if (fig_dir / name).exists():
            print(f"    - {name}")
    print("\nOne-line for panel: Phase 1 trains MAPPO+ST-GNN traffic control, evaluates vs baselines, and produces figures and comparison charts.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
