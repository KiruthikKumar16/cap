#!/usr/bin/env python3
"""Convenience wrapper for training, evaluation, and figure generation.

Usage examples:

# full/big experiment (default config)
python scripts/run_phase1_pipeline.py

# small/quick demo
python scripts/run_phase1_pipeline.py --quick

# or explicitly specify a config file
python scripts/run_phase1_pipeline.py --config configs/phase1.yaml
"""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run Phase1 train, eval, and plot in one go")
    parser.add_argument(
        "--config",
        help="Path to RL config file (overridden by --quick)",
        default="configs/phase1.yaml",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the quick-demo config (configs/phase1_quick_demo.yaml)",
    )
    args = parser.parse_args()

    if args.quick:
        config_file = "configs/phase1_quick_demo.yaml"
    else:
        config_file = args.config

    steps = [
        [sys.executable, "-m", "src.phase1.train_rl", "--config", config_file],
        [sys.executable, "scripts/real_sumo_evaluation.py"],
        [sys.executable, "scripts/phase1_generate_figures.py"],
    ]

    for cmd in steps:
        print("Running:", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"Command failed with exit code {rc}, aborting.")
            sys.exit(rc)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
