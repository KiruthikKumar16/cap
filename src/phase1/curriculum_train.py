
"""
Curriculum Learning Script for MARL Traffic Control

Trains agents on 3x3 -> 5x5 -> 10x10 with optional adaptive gating, mid-training
evaluation, and early stopping to avoid wasting time on bad hyperparameters.

Legacy mode (default): one training subprocess per stage, no evaluation — same as before.

Adaptive mode: pass --enable-adaptive and/or define a ``curriculum:`` block in the base YAML.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Project root for imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.phase1.evaluate_marl import evaluate_mean_reward

DEFAULT_MODEL_PATH = "marl_ppo_traffic.zip"

CURRICULUM_MAPS: List[Dict[str, Any]] = [
    {
        "name": "3x3",
        "net": "data/raw/grid_3x3.net.xml",
        "rou": "data/raw/grid_3x3.rou.xml",
        "steps": 100000,
    },
    {
        "name": "5x5",
        "net": "data/raw/grid_5x5.net.xml",
        "rou": "data/raw/grid_5x5.rou.xml",
        "steps": 300000,
    },
    {
        "name": "10x10",
        "net": "data/raw/grid_10x10.net.xml",
        "rou": "data/raw/grid_10x10.rou.xml",
        "steps": 1000000,
    },
]

FAST_DEV_TIMESTEPS = 10_000
FAST_DEV_EVAL_FREQ = 2048


@dataclass
class StageSpec:
    index: int
    name: str
    net: str
    rou: str
    timesteps: int
    reward_threshold: Optional[float] = None  # if None, post-stage gating skipped for this stage


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(config: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)


def merge_curriculum_from_yaml(
    base_config: dict, curriculum_maps: List[Dict[str, Any]]
) -> List[StageSpec]:
    """Build stage specs from CURRICULUM_MAPS + optional base_config['curriculum']."""
    cur = base_config.get("curriculum") or {}
    specs: List[StageSpec] = []
    for i, stage in enumerate(curriculum_maps):
        key = f"stage_{i}"
        y = cur.get(key) or {}
        timesteps = y.get("timesteps", stage["steps"])
        reward_threshold = y.get("reward_threshold")
        if reward_threshold is not None:
            reward_threshold = float(reward_threshold)
        specs.append(
            StageSpec(
                index=i,
                name=stage["name"],
                net=stage["net"],
                rou=stage["rou"],
                timesteps=int(timesteps),
                reward_threshold=reward_threshold,
            )
        )
    return specs


def moving_average(values: List[float], window: int = 3) -> float:
    if len(values) < window:
        return sum(values) / len(values) if values else 0.0
    return sum(values[-window:]) / window


def compute_slope(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = list(range(len(values)))
    y = values
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)

    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    den = sum((xi - x_mean) ** 2 for xi in x)

    return num / den if den != 0 else 0.0


def set_config_safely(config: dict, net_file: str, route_file: str, timesteps: int) -> None:
    if "sumo" in config:
        config["sumo"]["net_file"] = net_file
        config["sumo"]["route_file"] = route_file
    elif "data" in config and "sumo" in config["data"]:
        config["data"]["sumo"]["net_file"] = net_file
        config["data"]["sumo"]["route_file"] = route_file
    else:
        config["sumo"] = {"net_file": net_file, "route_file": route_file}
        
    if "training" not in config:
        config["training"] = {}
    config["training"]["total_timesteps"] = timesteps


def train_subprocess(
    python_executable: str,
    config_path: str,
    load_model: Optional[str],
    total_timesteps: int,
    subprocess_env: dict,
    require_cuda: bool,
) -> None:
    cmd = [
        python_executable,
        "src/phase1/train_marl.py",
        "--config",
        config_path,
        "--total-timesteps",
        str(total_timesteps),
    ]
    if load_model:
        cmd.extend(["--load-model", load_model])
    if require_cuda:
        cmd.append("--require-cuda")
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=subprocess_env)


def maybe_save_best(
    save_best_only: bool,
    stage_index: int,
    mean_reward: float,
    best_so_far: float,
    min_improvement: float,
) -> Tuple[float, bool]:
    """If improved, copy DEFAULT_MODEL_PATH to best_model_stage_{i}.zip. Returns (new_best, improved)."""
    improved = mean_reward > best_so_far + min_improvement
    new_best = float(mean_reward) if improved else float(best_so_far)
    if save_best_only and improved and os.path.isfile(DEFAULT_MODEL_PATH):
        out = f"best_model_stage_{stage_index}.zip"
        shutil.copy2(DEFAULT_MODEL_PATH, out)
        print(f"  [Best] Saved {out} (eval reward {mean_reward:.4f})")
    return new_best, improved


def run_stage_adaptive(
    *,
    python_executable: str,
    subprocess_env: dict,
    base_config_path: str,
    stage: StageSpec,
    temp_config_path: str,
    load_model: Optional[str],
    eval_episodes: int,
    reward_threshold_override: Optional[float],
    early_stop_patience: int,
    min_improvement: float,
    eval_freq: int,
    min_reward: Optional[float],
    save_best_only: bool,
    require_cuda: bool,
    fast_dev: bool,
    eval_ema_alpha: Optional[float],
    trend_window: int,
    stop_on_negative_trend: bool,
    min_passes: int,
) -> Tuple[bool, float, Optional[str]]:
    """
    Train with optional chunked evals; return (passed_gating, last_mean_reward, reason_if_fail).
    """
    stage_threshold = reward_threshold_override
    if stage_threshold is None:
        stage_threshold = stage.reward_threshold

    total_steps = FAST_DEV_TIMESTEPS if fast_dev else stage.timesteps
    freq = (FAST_DEV_EVAL_FREQ if fast_dev else eval_freq) if eval_freq > 0 else total_steps

    eval_rewards: List[float] = []
    ema_smoothed: Optional[float] = None
    best_eval = -float("inf")
    no_improve_evals = 0
    trained = 0
    current_load = load_model
    last_mean = 0.0
    consecutive_passes = 0
    metric_for_gate = 0.0

    while trained < total_steps:
        chunk = min(freq, total_steps - trained)
        base = load_yaml(base_config_path)
        set_config_safely(base, stage.net, stage.rou, chunk)
        save_yaml(base, temp_config_path)

        train_subprocess(
            python_executable,
            temp_config_path,
            current_load,
            chunk,
            subprocess_env,
            require_cuda,
        )
        trained += chunk
        current_load = DEFAULT_MODEL_PATH

        if not os.path.isfile(DEFAULT_MODEL_PATH):
            return False, last_mean, "missing_checkpoint"

        last_mean = evaluate_mean_reward(
            temp_config_path,
            DEFAULT_MODEL_PATH,
            episodes=eval_episodes,
            require_cuda=require_cuda,
            verbose=False,
        )
        eval_rewards.append(last_mean)

        smoothed_reward = moving_average(eval_rewards, window=3)

        if eval_ema_alpha is not None and 0 < eval_ema_alpha < 1:
            ema_smoothed = (
                eval_ema_alpha * last_mean + (1 - eval_ema_alpha) * (ema_smoothed or last_mean)
            )
            metric_for_gate = ema_smoothed
        else:
            metric_for_gate = smoothed_reward

        slope = compute_slope(eval_rewards[-trend_window:])

        print(f"  [Stage {stage.index}] Step {trained}/{total_steps} -> Raw: {last_mean:.4f} | Smoothed: {smoothed_reward:.4f} | Slope: {slope:.4f}")

        if stop_on_negative_trend and len(eval_rewards) >= trend_window and slope < 0:
            print(f"  [Stage {stage.index}] STOPPED (negative trend)")
            return False, metric_for_gate, "negative_trend"

        if min_reward is not None and metric_for_gate < min_reward:
            print(f"  [Stage {stage.index}] FAILED (min_reward: {min_reward}) -> stopping")
            return False, metric_for_gate, "below_min_reward"

        best_eval, improved = maybe_save_best(
            save_best_only, stage.index, metric_for_gate, best_eval, min_improvement
        )
        if improved:
            no_improve_evals = 0
        else:
            no_improve_evals += 1

        if no_improve_evals >= early_stop_patience and early_stop_patience > 0:
            print(
                f"  [Stage {stage.index}] Early stop: no improvement for {early_stop_patience} evals "
                f"(best {best_eval:.4f}, min_improvement {min_improvement})"
            )
            break

        if stage_threshold is not None:
            if metric_for_gate >= stage_threshold:
                consecutive_passes += 1
                if consecutive_passes >= min_passes:
                    print(f"  [Stage {stage.index}] PASS {consecutive_passes}/{min_passes} -> advancing")
                    break
                else:
                    print(f"  [Stage {stage.index}] PASS {consecutive_passes}/{min_passes}")
            else:
                consecutive_passes = 0

    if stage_threshold is not None:
        if consecutive_passes < min_passes:
            print(
                f"  [Stage {stage.index}] FAILED (threshold: {stage_threshold}, min_passes: {min_passes}, consecutive: {consecutive_passes})"
            )
            return False, metric_for_gate, "below_threshold_passes"
        print(
            f"  [Stage {stage.index}] PASSED (threshold: {stage_threshold}, metric: {metric_for_gate:.4f})"
        )

    return True, metric_for_gate, None


def build_stage_config_and_train_once(
    *,
    python_executable: str,
    subprocess_env: dict,
    base_config: dict,
    stage: StageSpec,
    temp_config_path: str,
    load_model: Optional[str],
    timesteps_override: Optional[int],
    require_cuda: bool,
    fast_dev: bool,
) -> None:
    steps = FAST_DEV_TIMESTEPS if fast_dev else (
        timesteps_override if timesteps_override is not None else stage.timesteps
    )
    set_config_safely(base_config, stage.net, stage.rou, steps)
    save_yaml(base_config, temp_config_path)
    train_subprocess(
        python_executable,
        temp_config_path,
        load_model,
        steps,
        subprocess_env,
        require_cuda,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curriculum MARL: 3x3 -> 5x5 -> 10x10 with optional adaptive gating."
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/phase2_10x10.yaml",
        help="YAML template; may include optional `curriculum:` block.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="Run only this stage index (0=3x3, 1=5x5, 2=10x10).",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Warm-start checkpoint. In full curriculum, only used if previous stage passed (or stage 0).",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override timesteps for this stage (single-stage legacy mode only). Ignored in full legacy curriculum.",
    )
    parser.add_argument("--keep-config", action="store_true", help="Keep generated temp YAML.")
    # Adaptive / tuning
    parser.add_argument(
        "--enable-adaptive",
        action="store_true",
        help="Enable eval, gating, chunked training, early stop (also auto-on if base YAML defines curriculum:).",
    )
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes per evaluation.")
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=None,
        help="Override reward threshold for all stages in this run (higher = stricter if rewards are negative, tune to your scale).",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help="Stop chunk loop after N evals without improvement (adaptive mode).",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=1e-4,
        help="Minimum reward improvement to reset early-stop counter.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=0,
        help="Run training in chunks of this many timesteps, then eval (0 = one chunk = full stage timesteps).",
    )
    parser.add_argument(
        "--save-best-only",
        action="store_true",
        help="When eval improves, copy checkpoint to best_model_stage_{i}.zip.",
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=None,
        help="Stop immediately if eval mean reward falls below this (adaptive chunk evals).",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Pass through to train_marl / fail eval if no CUDA.",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Small timesteps (~10k) and eval_freq ~2k for quick sanity checks.",
    )
    parser.add_argument(
        "--post-training-eval",
        action="store_true",
        help="After legacy (non-adaptive) training, run eval and print mean reward (no gating).",
    )
    parser.add_argument(
        "--eval-ema-alpha",
        type=float,
        default=None,
        help="Optional exponential moving average (0,1) for gating metric stability.",
    )
    parser.add_argument("--trend-window", type=int, default=5, help="Window for slope detection.")
    parser.add_argument("--stop-on-negative-trend", action="store_true", help="Stop if negative trend detected.")
    parser.add_argument("--min-passes", type=int, default=2, help="Consecutive evals above threshold needed.")
    args = parser.parse_args()

    if args.total_timesteps is not None and args.stage is None and not args.enable_adaptive:
        parser.error(
            "--total-timesteps without --stage is ambiguous in legacy full curriculum; "
            "use --stage N or --enable-adaptive, or edit YAML timesteps."
        )

    base_config_path = args.base_config
    base_config = load_yaml(base_config_path)
    curriculum_yaml = base_config.get("curriculum")
    adaptive = args.enable_adaptive or bool(curriculum_yaml)

    stage_specs = merge_curriculum_from_yaml(base_config, CURRICULUM_MAPS)

    if args.stage is not None:
        stages_to_run: List[Tuple[int, StageSpec]] = [(args.stage, stage_specs[args.stage])]
    else:
        stages_to_run = list(enumerate(stage_specs))

    python_executable = sys.executable
    subprocess_env = dict(os.environ, PYTHONPATH=os.getcwd())

    previous_passed = True
    current_model: Optional[str] = args.load_model

    for i, stage in stages_to_run:
        print(f"\n{'=' * 20} STAGE {i + 1} ({stage.name}): {stage.net} {'=' * 20}")

        temp_config_path = f"configs/temp_curriculum_stage_{i}.yaml"

        # Safe warm-start: deny loading if previous curriculum stage failed
        if current_model and i > 0 and not previous_passed:
            print(
                f"[WARN] Previous stage did not pass gating — ignoring --load-model for stage {i} "
                f"(train from scratch). Use an explicit checkpoint only after a PASSED stage."
            )
            current_model = None

        if args.stage is not None and args.total_timesteps is not None:
            stage = StageSpec(
                index=stage.index,
                name=stage.name,
                net=stage.net,
                rou=stage.rou,
                timesteps=args.total_timesteps,
                reward_threshold=stage.reward_threshold,
            )

        if adaptive:
            gate_ok, mean_r, reason = run_stage_adaptive(
                python_executable=python_executable,
                subprocess_env=subprocess_env,
                base_config_path=base_config_path,
                stage=stage,
                temp_config_path=temp_config_path,
                load_model=current_model,
                eval_episodes=args.eval_episodes,
                reward_threshold_override=args.reward_threshold,
                early_stop_patience=args.early_stop_patience,
                min_improvement=args.min_improvement,
                eval_freq=args.eval_freq,
                min_reward=args.min_reward,
                save_best_only=args.save_best_only,
                require_cuda=args.require_cuda,
                fast_dev=args.fast_dev_run,
                eval_ema_alpha=args.eval_ema_alpha,
                trend_window=args.trend_window,
                stop_on_negative_trend=args.stop_on_negative_trend,
                min_passes=args.min_passes,
            )
            previous_passed = gate_ok
            if not gate_ok:
                print(f"\n[STOP] Curriculum halted at stage {i} ({stage.name}). Reason: {reason or 'gate'}")
                if not args.keep_config and os.path.exists(temp_config_path):
                    os.remove(temp_config_path)
                sys.exit(1)
            # Post-stage: optionally re-eval for logging
            print(f"[Stage {i}] Final eval mean reward: {mean_r:.4f} -> PASSED")
        else:
            # Legacy: reload fresh base_config so we do not accumulate edits
            cfg = load_yaml(base_config_path)
            build_stage_config_and_train_once(
                python_executable=python_executable,
                subprocess_env=subprocess_env,
                base_config=cfg,
                stage=stage,
                temp_config_path=temp_config_path,
                load_model=current_model,
                timesteps_override=(
                    args.total_timesteps if args.stage is not None else None
                ),
                require_cuda=args.require_cuda,
                fast_dev=args.fast_dev_run,
            )
            if (
                args.post_training_eval
                and args.eval_episodes > 0
                and os.path.isfile(DEFAULT_MODEL_PATH)
            ):
                mean_r = evaluate_mean_reward(
                    temp_config_path,
                    DEFAULT_MODEL_PATH,
                    episodes=args.eval_episodes,
                    require_cuda=args.require_cuda,
                    verbose=False,
                )
                print(
                    f"[Stage {i}] Post-training eval mean reward: {mean_r:.4f} (legacy, informational)"
                )

        current_model = DEFAULT_MODEL_PATH

        if not args.keep_config and os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    if args.stage is None:
        print("\n[OK] Curriculum learning finished successfully!")
    else:
        last_name = stages_to_run[0][1].name
        print(f"\n[OK] Stage {args.stage} ({last_name}) finished.")


if __name__ == "__main__":
    main()
