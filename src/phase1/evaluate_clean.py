"""
Phase 1 Evaluation — Clean implementation from core concepts.

Core concepts:
1. Run N episodes with policy A (fixed-time): action = (step // phase_duration) % 4 per intersection.
2. Run N episodes with policy B (DQN): action = model.predict(obs)[0] on wrapped env.
3. Collect per-step: reward, departed, travel_time, waiting_time, queue_length; aggregate per episode.
4. Use one seed per episode (round-robin) so results vary and we get std > 0.
5. Save summary JSON for figures. No placeholders; all values from SUMO.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import yaml

from stable_baselines3 import DQN
from gymnasium import spaces

from src.phase1.train_rl import load_config, create_environment
from src.phase1.dqn_agent import MultiDiscreteToDiscreteWrapper, GNNObservationWrapper


def _unwrap_info(info: Any) -> Dict:
    if isinstance(info, (list, tuple)) and len(info) > 0:
        return info[0] if isinstance(info[0], dict) else {}
    return info if isinstance(info, dict) else {}


def _scalar(x: Any) -> float:
    if np.ndim(x) > 0:
        return float(np.asarray(x).flatten()[0])
    return float(x)


def run_episodes(
    env,
    get_action: Callable[[np.ndarray, int], np.ndarray],
    num_episodes: int,
    seeds: List[int],
    max_steps: int,
    policy_name: str,
    log_first_actions: int = 5,
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], bool]:
    """
    Run num_episodes with the given policy. get_action(obs, step) returns action.
    Returns: episode_rewards, episode_lengths, episode_throughputs, episode_travel_times,
             episode_waiting_times, episode_queue_lengths, used_sumo.
    """
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_throughputs: List[float] = []
    episode_travel_times: List[float] = []
    episode_waiting_times: List[float] = []
    episode_queue_lengths: List[float] = []
    used_sumo = False

    for ep in range(num_episodes):
        seed = seeds[ep % len(seeds)]
        np.random.seed(seed)
        out = env.reset(seed=seed)
        obs = out[0] if isinstance(out, (tuple, list)) else out

        total_reward = 0.0
        total_departed = 0.0
        total_travel_time = 0.0
        total_waiting_time = 0.0
        total_queue_length = 0.0
        step_count = 0
        done = False

        while not done and step_count < max_steps:
            action = get_action(obs, step_count)
            if ep == 0 and step_count < log_first_actions:
                print(f"  [{policy_name}] episode 0 step {step_count} action = {action}")

            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
            else:
                obs, reward, done, info = step_out[0], step_out[1], step_out[2], step_out[3]
                terminated = done
                truncated = False
            info = _unwrap_info(info)

            total_reward += _scalar(reward)
            total_departed += _scalar(info.get("departed", 0))
            total_travel_time += _scalar(info.get("travel_time", 0.0))
            total_waiting_time += _scalar(info.get("waiting_time", 0.0))
            total_queue_length += _scalar(info.get("queue_length", 0.0))
            step_count += 1
            done = terminated or truncated
            if not used_sumo and step_count == 1:
                used_sumo = info.get("sumo_running", False)

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        episode_throughputs.append(total_departed)
        episode_travel_times.append(total_travel_time)
        steps = max(step_count, 1)
        episode_waiting_times.append(total_waiting_time / steps)
        episode_queue_lengths.append(total_queue_length / steps)

    return (
        episode_rewards,
        episode_lengths,
        episode_throughputs,
        episode_travel_times,
        episode_waiting_times,
        episode_queue_lengths,
        used_sumo,
    )


def fixed_time_action(env, phase_duration: int) -> Callable:
    """Return get_action(obs, step) for fixed-time: phase = (step // phase_duration) % 4 per intersection."""
    n = env.num_intersections if hasattr(env, "num_intersections") else getattr(env, "env", env).num_intersections

    def get_action(_obs: np.ndarray, step: int) -> np.ndarray:
        phase = (step // phase_duration) % 4
        return np.array([phase] * n, dtype=np.int32)

    return get_action


def random_action(env) -> Callable:
    """Return get_action for random policy (random phase per intersection each step)."""
    n = env.num_intersections if hasattr(env, "num_intersections") else getattr(env, "env", env).num_intersections

    def get_action(_obs: np.ndarray, _step: int) -> np.ndarray:
        return np.array([np.random.randint(0, 4) for _ in range(n)], dtype=np.int32)

    return get_action


def main():
    parser = argparse.ArgumentParser(description="Phase 1 evaluation (clean)")
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/phase1/dqn_traffic_final.zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds (round-robin over episodes)")
    parser.add_argument("--phase-duration", type=int, default=30)
    parser.add_argument("--baseline", type=str, default="fixed-time", choices=["fixed-time", "random"], help="Baseline: fixed-time or random")
    parser.add_argument("--save-summary", type=str, default="outputs/phase1/evaluation_summary.json")
    parser.add_argument("--no-verify-actions", action="store_true", help="Do not log first 5 actions per policy")
    args = parser.parse_args()
    log_actions = 5 if not args.no_verify_actions else 0

    config = load_config(args.config)
    sumo_cfg = config["sumo"]
    max_steps = sumo_cfg.get("simulation_steps", 3600)
    seeds = list(range(42, 42 + args.seeds))

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return

    # ----- Baseline: fixed-time or random -----
    baseline_name = args.baseline
    print(f"Running {baseline_name} baseline...")
    env_ft = create_environment(config)
    if baseline_name == "random":
        get_ft = random_action(env_ft)
        baseline_label = "Random"
    else:
        get_ft = fixed_time_action(env_ft, args.phase_duration)
        baseline_label = "Fixed-time"
    ft_rewards, ft_lengths, ft_tput, ft_tt, ft_wt, ft_q, used_sumo = run_episodes(
        env_ft,
        get_ft,
        args.episodes,
        seeds,
        max_steps,
        baseline_name,  # policy_name
        log_actions,  # log_first_actions
    )
    env_ft.close()

    # ----- DQN: wrapped env -----
    print("Running DQN (GNN-RL)...")
    env_raw = create_environment(config)
    wrapped = MultiDiscreteToDiscreteWrapper(env_raw)
    wrapped = GNNObservationWrapper(wrapped)
    model = DQN.load(str(checkpoint_path), env=wrapped)

    def get_dqn(obs: np.ndarray, _step: int) -> np.ndarray:
        a, _ = model.predict(obs, deterministic=True)
        return a

    dqn_rewards, dqn_lengths, dqn_tput, dqn_tt, dqn_wt, dqn_q, _ = run_episodes(
        wrapped,
        get_dqn,
        args.episodes,
        seeds,
        max_steps,
        "dqn",  # policy_name
        log_actions,  # log_first_actions
    )
    wrapped.close()

    # ----- Aggregate -----
    dqn_r, ft_r = np.array(dqn_rewards), np.array(ft_rewards)
    dqn_mean_r, dqn_std_r = float(np.mean(dqn_r)), float(np.std(dqn_r))
    ft_mean_r, ft_std_r = float(np.mean(ft_r)), float(np.std(ft_r))
    dqn_mean_tput = float(np.mean(dqn_tput))
    ft_mean_tput = float(np.mean(ft_tput))
    dqn_mean_tt = float(np.mean(dqn_tt))
    ft_mean_tt = float(np.mean(ft_tt))

    print("\n" + "=" * 60)
    print("Phase 1 Evaluation (clean)")
    print("=" * 60)
    print(f"  Episodes: {args.episodes}  Seeds: {seeds}")
    print(f"  Checkpoint: {checkpoint_path}")
    print("-" * 60)
    print(f"  DQN (GNN-RL):     mean_reward = {dqn_mean_r:+.2f} +/- {dqn_std_r:.2f}")
    print(f"  {baseline_label}:       mean_reward = {ft_mean_r:+.2f} +/- {ft_std_r:.2f}")
    print(f"  DQN throughput (departed/episode): {dqn_mean_tput:.1f}")
    print(f"  {baseline_label} throughput:              {ft_mean_tput:.1f}")
    if dqn_mean_tt > 0 or ft_mean_tt > 0:
        print(f"  DQN mean travel_time (sum/episode): {dqn_mean_tt:.1f}")
        print(f"  Fixed-time mean travel_time:        {ft_mean_tt:.1f}")
    print("-" * 60)
    if ft_mean_r != 0:
        pct = 100 * (dqn_mean_r - ft_mean_r) / abs(ft_mean_r)
        print(f"  DQN vs {baseline_label} reward: {pct:+.1f}% (positive = DQN better)")
    try:
        from scipy import stats
        if len(dqn_r) >= 2 and len(ft_r) >= 2:
            _, p = stats.ttest_ind(dqn_r, ft_r)
            print(f"  t-test (DQN vs {baseline_label}): p = {p:.4f}")
    except ImportError:
        pass
    print("=" * 60)
    if abs(dqn_mean_r - ft_mean_r) < 1e-6:
        if baseline_name == "fixed-time":
            print(f"[Note] DQN and {baseline_label} rewards are identical. This means:")
            print("       1. DQN hasn't learned yet (5k steps too short; try 20k-100k), OR")
            print("       2. Try --baseline random to compare DQN vs random (easier to beat).")
        else:
            print(f"[Note] DQN and {baseline_label} rewards are identical - DQN hasn't learned yet.")

    # ----- Save summary (same format as before for figures) -----
    out_path = Path(args.save_summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "num_episodes": args.episodes,
        "num_seeds": len(seeds),
        "total_runs": len(dqn_rewards),
        "used_sumo": used_sumo,
        "dqn": {
            "mean_reward": dqn_mean_r,
            "std_reward": dqn_std_r,
            "mean_throughput": dqn_mean_tput,
            "std_throughput": float(np.std(dqn_tput)),
            "mean_travel_time": dqn_mean_tt,
            "std_travel_time": float(np.std(dqn_tt)),
            "mean_waiting_time": float(np.mean(dqn_wt)),
            "std_waiting_time": float(np.std(dqn_wt)),
            "mean_queue_length": float(np.mean(dqn_q)),
            "std_queue_length": float(np.std(dqn_q)),
            "rewards": [float(x) for x in dqn_rewards],
            "throughputs": [float(x) for x in dqn_tput],
            "travel_times": [float(x) for x in dqn_tt],
            "waiting_times": [float(x) for x in dqn_wt],
            "queue_lengths": [float(x) for x in dqn_q],
        },
        "fixed_time": {
            "mean_reward": ft_mean_r,
            "std_reward": ft_std_r,
            "mean_throughput": ft_mean_tput,
            "std_throughput": float(np.std(ft_tput)),
            "mean_travel_time": ft_mean_tt,
            "std_travel_time": float(np.std(ft_tt)),
            "mean_waiting_time": float(np.mean(ft_wt)),
            "std_waiting_time": float(np.std(ft_wt)),
            "mean_queue_length": float(np.mean(ft_q)),
            "std_queue_length": float(np.std(ft_q)),
            "rewards": [float(x) for x in ft_rewards],
            "throughputs": [float(x) for x in ft_tput],
            "travel_times": [float(x) for x in ft_tt],
            "waiting_times": [float(x) for x in ft_wt],
            "queue_lengths": [float(x) for x in ft_q],
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Summary saved to {out_path}")


if __name__ == "__main__":
    main()
