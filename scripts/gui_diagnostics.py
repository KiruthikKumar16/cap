import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO

from src.phase1.train_rl import load_config, create_environment
from src.phase1.evaluate import evaluate_fixed_time, evaluate_random


def _unwrap_info(info: Any) -> Dict[str, Any]:
    if isinstance(info, (list, tuple)) and len(info) > 0:
        item = info[0]
        return item if isinstance(item, dict) else {}
    return info if isinstance(info, dict) else {}


def _compute_recovery_step(queue_series: List[float]) -> int:
    if not queue_series:
        return -1
    arr = np.asarray(queue_series, dtype=float)
    peak_idx = int(np.argmax(arr))
    if peak_idx >= len(arr) - 1:
        return -1
    tail = arr[peak_idx + 1 :]
    baseline = float(np.percentile(arr, 30))
    for i, v in enumerate(tail, start=peak_idx + 1):
        if v <= baseline:
            return i
    return -1


def run_ppo_diagnostics(config: Dict[str, Any], checkpoint: Path, max_steps: int) -> pd.DataFrame:
    env = create_environment(config)
    model = PPO.load(str(checkpoint), env=env)
    vec_env = model.get_env() if model.get_env() is not None else env
    reset_out = vec_env.reset()
    obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out

    rows: List[Dict[str, Any]] = []
    done = False
    step_idx = 0
    while not done and step_idx < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        step_out = vec_env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
        else:
            obs, reward, done_arr, info = step_out[0], step_out[1], step_out[2], step_out[3]
            terminated = done_arr
            truncated = np.array([False]) if np.ndim(done_arr) > 0 else False
        info = _unwrap_info(info)
        r = float(np.asarray(reward).flatten()[0]) if np.ndim(reward) > 0 else float(reward)
        rows.append(
            {
                "step": step_idx,
                "reward": r,
                "arrived_vehicles": float(info.get("step_arrived_vehicles", 0.0)),
                "stopped_vehicles": float(info.get("step_stopped_vehicles", 0.0)),
                "total_waiting_time": float(info.get("step_total_waiting_time", 0.0)),
                "total_queue_length": float(info.get("step_total_queue_length", 0.0)),
                "placeholder_mode": bool(info.get("placeholder_mode", False)),
            }
        )
        done = bool(np.any(terminated) or np.any(truncated))
        step_idx += 1

    env.close()
    return pd.DataFrame(rows)


def run_baseline_diagnostics(config: Dict[str, Any], controller: str, max_steps: int) -> pd.DataFrame:
    env = create_environment(config)
    num_episodes = 1
    if controller == "FixedTime":
        evaluate_fixed_time(env, num_episodes=num_episodes, max_steps_per_episode=max_steps)
    else:
        evaluate_random(env, num_episodes=num_episodes, max_steps_per_episode=max_steps)
    # evaluate_* currently does not expose step-wise logs, so run a thin manual loop for logs.
    reset_out = env.reset()
    _ = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
    rows: List[Dict[str, Any]] = []
    num_intersections = getattr(env, "num_intersections", getattr(env, "num_envs", 1))
    done = False
    step_idx = 0
    while not done and step_idx < max_steps:
        if controller == "FixedTime":
            phase = (step_idx // 30) % 4
            action = np.array([phase] * num_intersections, dtype=np.int32)
        else:
            action = env.action_space.sample()
        step_out = env.step(action)
        if len(step_out) == 5:
            _, reward, terminated, truncated, info = step_out
        else:
            _, reward, terminated, info = step_out[0], step_out[1], step_out[2], step_out[3]
            truncated = np.array([False]) if np.ndim(terminated) > 0 else False
        info = _unwrap_info(info)
        r = float(np.asarray(reward).flatten()[0]) if np.ndim(reward) > 0 else float(reward)
        rows.append(
            {
                "step": step_idx,
                "reward": r,
                "arrived_vehicles": float(info.get("step_arrived_vehicles", 0.0)),
                "stopped_vehicles": float(info.get("step_stopped_vehicles", 0.0)),
                "total_waiting_time": float(info.get("step_total_waiting_time", 0.0)),
                "total_queue_length": float(info.get("step_total_queue_length", 0.0)),
                "placeholder_mode": bool(info.get("placeholder_mode", False)),
            }
        )
        done = bool(np.any(terminated) or np.any(truncated))
        step_idx += 1
    env.close()
    return pd.DataFrame(rows)


def save_diagnostics(df: pd.DataFrame, output_dir: Path, controller: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{controller.lower()}_step_log.csv"
    df.to_csv(csv_path, index=False)

    queue = df["total_queue_length"].tolist() if "total_queue_length" in df.columns else []
    recovery_step = _compute_recovery_step(queue)
    summary = {
        "controller": controller,
        "steps_logged": int(len(df)),
        "peak_queue_length": float(np.max(queue)) if queue else 0.0,
        "mean_queue_length": float(np.mean(queue)) if queue else 0.0,
        "recovery_step_after_peak": int(recovery_step),
        "mean_waiting_time": float(df["total_waiting_time"].mean()) if "total_waiting_time" in df else 0.0,
        "mean_arrived_vehicles": float(df["arrived_vehicles"].mean()) if "arrived_vehicles" in df else 0.0,
    }
    (output_dir / f"{controller.lower()}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(df["step"], df["total_queue_length"], label="Total Queue Length")
    axes[0].plot(df["step"], df["total_waiting_time"], label="Total Waiting Time")
    if recovery_step >= 0:
        axes[0].axvline(recovery_step, color="green", linestyle="--", label="Recovery Step")
    axes[0].set_ylabel("Traffic Load")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["step"], df["arrived_vehicles"], label="Arrived Vehicles/Step")
    axes[1].plot(df["step"], df["stopped_vehicles"], label="Stopped Vehicles/Step")
    axes[1].set_xlabel("Simulation Step")
    axes[1].set_ylabel("Vehicles")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{controller.lower()}_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Diagnostics CSV: {csv_path}")
    print(f"[OK] Diagnostics summary: {output_dir / f'{controller.lower()}_summary.json'}")
    print(f"[OK] Diagnostics plot: {output_dir / f'{controller.lower()}_diagnostics.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SUMO GUI diagnostics logger and analyzer")
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--checkpoint", default="outputs/phase1/dqn_traffic_final.zip")
    parser.add_argument("--controller", choices=["PPO", "FixedTime", "Random"], default="PPO")
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--output-dir", default="outputs/gui_diagnostics")
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("sumo", {})
    config["sumo"]["gui"] = True
    config.setdefault("output", {})
    config["output"]["final_model_path"] = args.checkpoint

    print("=" * 68)
    print(f"GUI diagnostics started | controller={args.controller} | max_steps={args.max_steps}")
    print("=" * 68)

    if args.controller == "PPO":
        df = run_ppo_diagnostics(config, Path(args.checkpoint), args.max_steps)
    else:
        df = run_baseline_diagnostics(config, args.controller, args.max_steps)
    save_diagnostics(df, Path(args.output_dir), args.controller)


if __name__ == "__main__":
    main()
