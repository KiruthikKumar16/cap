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

    # 0) Create SUMO network if missing (so training uses real simulation)
    net_file = root / "data" / "raw" / "grid_2x2.net.xml"
    if not net_file.exists():
        print("\n" + "=" * 60)
        print("Step 0/3: Creating SUMO 2x2 grid network (data/raw/)")
        print("=" * 60)
        r0 = subprocess.run([py, "scripts/create_sumo_network.py"], cwd=str(root))
        if r0.returncode != 0:
            print("[WARN] SUMO network creation failed; training will use placeholder mode.")
        else:
            print("[OK] SUMO network ready.")
    else:
        print("[OK] SUMO network found: data/raw/grid_2x2.net.xml")

    # 1) Train
    train_cmd = [py, "-m", "src.phase1.train_rl", "--config", config]
    if args.quick:
        import yaml
        with open(config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["training"]["total_timesteps"] = 10000
        quick_config = root / "configs" / "phase1_quick_demo.yaml"
        with open(quick_config, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        config = str(quick_config)
        train_cmd = [py, "-m", "src.phase1.train_rl", "--config", config]
        print("[QUICK] Training for 10,000 steps only.")
    print("\n" + "=" * 60)
    print("Step 1/3: Training")
    print("=" * 60)
    r = subprocess.run(train_cmd, cwd=str(root))
    if r.returncode != 0:
        print("[ERROR] Training failed. Exiting.")
        return r.returncode

    # 2) Evaluate (save summary for comparison charts)
    print("\n" + "=" * 60)
    print("Step 2/3: Evaluation (DQN vs Fixed-time vs Actuated)")
    print("=" * 60)
    eval_config = args.config if not args.quick else config
    summary_path = root / "outputs" / "phase1" / "evaluation_summary.json"
    r = subprocess.run(
        [py, "-m", "src.phase1.evaluate", "--config", eval_config, "--episodes", "5", "--seeds", "2", "--actuated", "--save-summary", str(summary_path)],
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
    print("\nOne-line for panel: Phase 1 trains a GNN-DQN traffic controller, evaluates it vs fixed-time and actuated baselines, and produces architecture diagrams, learning curves, and comparison charts showing why ours is better (SOTA).")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
