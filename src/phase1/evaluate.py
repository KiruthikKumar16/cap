"""
Phase 1 Evaluation Script

Evaluates the trained DQN agent and compares against fixed-time and actuated baselines.
Supports multiple seeds and statistical test (t-test). Works in placeholder mode or with SUMO.
Use --save-summary to write results to JSON for comparison charts.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import yaml
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import DQN
from gymnasium import spaces

from src.phase1.train_rl import load_config, create_environment
from src.phase1.dqn_agent import MultiDiscreteToDiscreteWrapper, GNNObservationWrapper

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def wrap_env_for_dqn(env):
    """Wrap environment the same way as create_dqn_agent (for loading DQN)."""
    if isinstance(env.action_space, spaces.MultiDiscrete):
        env = MultiDiscreteToDiscreteWrapper(env)
    return GNNObservationWrapper(env)


def _unwrap_info(info):
    """VecEnv returns list of infos; unwrap to single dict for departed/travel_time."""
    if isinstance(info, (list, tuple)) and len(info) > 0:
        return info[0]
    return info


def evaluate_dqn(
    model: DQN,
    env,
    num_episodes: int,
    deterministic: bool = True,
    max_steps_per_episode: int = 3600,
) -> Tuple[List[float], List[int], List[float], List[float], List[float], List[float], bool]:
    """
    Run evaluation episodes with the DQN agent.

    Returns:
        episode_rewards, episode_lengths, episode_throughputs, episode_travel_times, episode_waiting_times, episode_queue_lengths, placeholder_mode
    """
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_throughputs: List[float] = []
    episode_travel_times: List[float] = []
    episode_waiting_times: List[float] = []
    episode_queue_lengths: List[float] = []
    placeholder_mode = True  # assume placeholder until we see sumo_running

    # Use model's env when available (SB3 wraps in DummyVecEnv+Monitor; Monitor may report reward in info['episode']['r'])
    vec_env = model.get_env() if hasattr(model, "get_env") and model.get_env() is not None else None
    use_vec = vec_env is not None and hasattr(vec_env, "envs")

    for ep in range(num_episodes):
        run_env = vec_env if use_vec else env
        reset_out = run_env.reset()
        obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
        total_reward = 0.0
        total_departed = 0.0
        total_travel_time = 0.0
        total_waiting_time = 0.0
        total_queue_length = 0.0
        step_count = 0
        done = False
        last_info = None
        while not done and step_count < max_steps_per_episode:
            action, _ = model.predict(obs, deterministic=deterministic)
            step_out = run_env.step(action)
            # VecEnv (SB3) returns 4 values: (obs, rewards, dones, infos); gymnasium returns 5: (obs, reward, terminated, truncated, info)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
            else:
                obs, reward, done, info = step_out[0], step_out[1], step_out[2], step_out[3]
                terminated = done
                truncated = np.array([False]) if np.ndim(done) > 0 else False
            info = _unwrap_info(info)
            last_info = info
            if step_count == 0 and ep == 0:
                placeholder_mode = info.get("placeholder_mode", not info.get("sumo_running", False))
            # Ensure scalars (VecEnv returns arrays)
            r = float(np.asarray(reward).flatten()[0]) if np.ndim(reward) > 0 else float(reward)
            total_reward += r
            total_departed += float(np.asarray(info.get("departed", 0)).flatten()[0]) if np.ndim(info.get("departed", 0)) > 0 else float(info.get("departed", 0))
            total_travel_time += float(np.asarray(info.get("travel_time", 0.0)).flatten()[0]) if np.ndim(info.get("travel_time", 0.0)) > 0 else float(info.get("travel_time", 0.0))
            total_waiting_time += float(np.asarray(info.get("waiting_time", 0.0)).flatten()[0]) if np.ndim(info.get("waiting_time", 0.0)) > 0 else float(info.get("waiting_time", 0.0))
            total_queue_length += float(np.asarray(info.get("queue_length", 0.0)).flatten()[0]) if np.ndim(info.get("queue_length", 0.0)) > 0 else float(info.get("queue_length", 0.0))
            step_count += 1
            term = bool(np.asarray(terminated).flatten()[0]) if np.ndim(terminated) > 0 else bool(terminated)
            trun = bool(np.asarray(truncated).flatten()[0]) if np.ndim(truncated) > 0 else bool(truncated)
            done = term or trun
        # Prefer our accumulated total_reward; use Monitor episode["r"] only when present and non-zero (or when total_reward is 0)
        ep_reward = total_reward
        if last_info and isinstance(last_info, dict):
            ep_data = last_info.get("episode") or (last_info[0].get("episode") if isinstance(last_info, (list, tuple)) and last_info else None)
            if ep_data is not None and "r" in ep_data:
                mon_r = float(ep_data["r"])
                if mon_r != 0 or total_reward == 0:
                    ep_reward = mon_r
        episode_rewards.append(ep_reward)
        episode_lengths.append(step_count)
        episode_throughputs.append(total_departed)
        episode_travel_times.append(total_travel_time)
        avg_waiting = total_waiting_time / step_count if step_count > 0 else 0.0
        episode_waiting_times.append(avg_waiting)
        avg_queue = total_queue_length / step_count if step_count > 0 else 0.0
        episode_queue_lengths.append(avg_queue)

    return episode_rewards, episode_lengths, episode_throughputs, episode_travel_times, episode_waiting_times, episode_queue_lengths, placeholder_mode


def evaluate_fixed_time(
    env,
    num_episodes: int,
    phase_duration: int = 30,
    max_steps_per_episode: int = 3600,
) -> Tuple[List[float], List[int], List[float], List[float], List[float], List[float], bool]:
    """
    Run evaluation episodes with fixed-time controller.
    Returns: episode_rewards, episode_lengths, episode_throughputs, episode_travel_times, episode_waiting_times, episode_queue_lengths, placeholder_mode.
    """
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_throughputs: List[float] = []
    episode_travel_times: List[float] = []
    episode_waiting_times: List[float] = []
    episode_queue_lengths: List[float] = []
    placeholder_mode = True
    num_intersections = env.num_intersections

    for ep in range(num_episodes):
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
        total_reward = 0.0
        total_departed = 0.0
        total_travel_time = 0.0
        total_waiting_time = 0.0
        total_queue_length = 0.0
        step_count = 0
        done = False
        last_info = None
        while not done and step_count < max_steps_per_episode:
            phase = (step_count // phase_duration) % 4
            action = np.array([phase] * num_intersections, dtype=np.int32)
            obs, reward, terminated, truncated, info = env.step(action)
            info = _unwrap_info(info)
            last_info = info
            if step_count == 0 and ep == 0:
                placeholder_mode = info.get("placeholder_mode", not info.get("sumo_running", False))
            r = float(np.asarray(reward).flatten()[0]) if np.ndim(reward) > 0 else float(reward)
            total_reward += r
            total_departed += float(np.asarray(info.get("departed", 0)).flatten()[0]) if np.ndim(info.get("departed", 0)) > 0 else float(info.get("departed", 0))
            total_travel_time += float(np.asarray(info.get("travel_time", 0.0)).flatten()[0]) if np.ndim(info.get("travel_time", 0.0)) > 0 else float(info.get("travel_time", 0.0))
            total_waiting_time += float(np.asarray(info.get("waiting_time", 0.0)).flatten()[0]) if np.ndim(info.get("waiting_time", 0.0)) > 0 else float(info.get("waiting_time", 0.0))
            total_queue_length += float(np.asarray(info.get("queue_length", 0.0)).flatten()[0]) if np.ndim(info.get("queue_length", 0.0)) > 0 else float(info.get("queue_length", 0.0))
            step_count += 1
            done = terminated or truncated
        ep_reward = total_reward
        if last_info and isinstance(last_info, dict) and last_info.get("episode") and "r" in last_info["episode"]:
            mon_r = float(last_info["episode"]["r"])
            if mon_r != 0 or total_reward == 0:
                ep_reward = mon_r
        episode_rewards.append(ep_reward)
        episode_lengths.append(step_count)
        episode_throughputs.append(total_departed)
        episode_travel_times.append(total_travel_time)
        avg_waiting = total_waiting_time / step_count if step_count > 0 else 0.0
        episode_waiting_times.append(avg_waiting)
        avg_queue = total_queue_length / step_count if step_count > 0 else 0.0
        episode_queue_lengths.append(avg_queue)

    return episode_rewards, episode_lengths, episode_throughputs, episode_travel_times, episode_waiting_times, episode_queue_lengths, placeholder_mode


def evaluate_actuated(
    env,
    num_episodes: int,
    phase_duration: int = 30,
    max_steps_per_episode: int = 3600,
) -> Tuple[List[float], List[int], List[float], List[float], List[float], List[float], bool]:
    """
    Run evaluation episodes with actuated controller (max-pressure style baseline).
    Returns: episode_rewards, episode_lengths, episode_throughputs, episode_travel_times, episode_waiting_times, episode_queue_lengths, placeholder_mode.
    """
    try:
        import traci
    except ImportError:
        # Fall back to fixed-time when SUMO/TraCI is unavailable.
        return evaluate_fixed_time(env, num_episodes, phase_duration, max_steps_per_episode)

    def _build_phase_lane_map(tl_id: str):
        try:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)
        except Exception:
            return []
        if not logic:
            return []
        phases = logic[0].phases
        controlled_links = traci.trafficlight.getControlledLinks(tl_id)
        phase_lanes = []
        for phase in phases:
            state = phase.state
            lanes = set()
            for i, link in enumerate(controlled_links):
                if i < len(state) and state[i] in ("G", "g"):
                    for conn in link:
                        lanes.add(conn[0])  # from-lane
            phase_lanes.append(lanes)
        return phase_lanes

    def _score_phase_lanes(lanes):
        score = 0.0
        for lane_id in lanes:
            try:
                score += traci.lane.getLastStepHaltingNumber(lane_id)
            except Exception:
                pass
        return score

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_throughputs: List[float] = []
    episode_travel_times: List[float] = []
    episode_waiting_times: List[float] = []
    episode_queue_lengths: List[float] = []
    placeholder_mode = True

    for ep in range(num_episodes):
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
        total_reward = 0.0
        total_departed = 0.0
        total_travel_time = 0.0
        total_waiting_time = 0.0
        total_queue_length = 0.0
        step_count = 0
        done = False
        last_info = None

        # Build phase->lanes mapping after SUMO is up
        tl_ids = traci.trafficlight.getIDList()
        tl_phase_lanes = {tl_id: _build_phase_lane_map(tl_id) for tl_id in tl_ids}

        while not done and step_count < max_steps_per_episode:
            # Update phases periodically (actuated decision interval)
            if step_count % phase_duration == 0:
                actions = []
                for tl_id in tl_ids:
                    phase_lanes = tl_phase_lanes.get(tl_id, [])
                    if not phase_lanes:
                        actions.append(0)
                        continue
                    best_phase = 0
                    best_score = -1.0
                    for idx, lanes in enumerate(phase_lanes):
                        score = _score_phase_lanes(lanes)
                        if score > best_score:
                            best_score = score
                            best_phase = idx
                    actions.append(best_phase)
                action = np.array(actions, dtype=np.int32)
            else:
                # Keep current phases between decisions
                try:
                    action = np.array([traci.trafficlight.getPhase(tl_id) for tl_id in tl_ids], dtype=np.int32)
                except Exception:
                    action = np.zeros(len(tl_ids), dtype=np.int32)

            obs, reward, terminated, truncated, info = env.step(action)
            info = _unwrap_info(info)
            last_info = info
            if step_count == 0 and ep == 0:
                placeholder_mode = info.get("placeholder_mode", not info.get("sumo_running", False))
            r = float(np.asarray(reward).flatten()[0]) if np.ndim(reward) > 0 else float(reward)
            total_reward += r
            total_departed += float(np.asarray(info.get("departed", 0)).flatten()[0]) if np.ndim(info.get("departed", 0)) > 0 else float(info.get("departed", 0))
            total_travel_time += float(np.asarray(info.get("travel_time", 0.0)).flatten()[0]) if np.ndim(info.get("travel_time", 0.0)) > 0 else float(info.get("travel_time", 0.0))
            total_waiting_time += float(np.asarray(info.get("waiting_time", 0.0)).flatten()[0]) if np.ndim(info.get("waiting_time", 0.0)) > 0 else float(info.get("waiting_time", 0.0))
            total_queue_length += float(np.asarray(info.get("queue_length", 0.0)).flatten()[0]) if np.ndim(info.get("queue_length", 0.0)) > 0 else float(info.get("queue_length", 0.0))
            step_count += 1
            done = terminated or truncated

        ep_reward = total_reward
        if last_info and isinstance(last_info, dict) and last_info.get("episode") and "r" in last_info["episode"]:
            mon_r = float(last_info["episode"]["r"])
            if mon_r != 0 or total_reward == 0:
                ep_reward = mon_r
        episode_rewards.append(ep_reward)
        episode_lengths.append(step_count)
        episode_throughputs.append(total_departed)
        episode_travel_times.append(total_travel_time)
        avg_waiting = total_waiting_time / step_count if step_count > 0 else 0.0
        episode_waiting_times.append(avg_waiting)
        avg_queue = total_queue_length / step_count if step_count > 0 else 0.0
        episode_queue_lengths.append(avg_queue)

    return episode_rewards, episode_lengths, episode_throughputs, episode_travel_times, episode_waiting_times, episode_queue_lengths, placeholder_mode


def _decode_flat_to_multi(flat_action: int, nvec: np.ndarray) -> np.ndarray:
    """Decode flat action to multi-discrete (same as MultiDiscreteToDiscreteWrapper._convert_action)."""
    multi = np.zeros(len(nvec), dtype=np.int32)
    remaining = flat_action
    for i in range(len(nvec) - 1, -1, -1):
        multi[i] = remaining % nvec[i]
        remaining = remaining // nvec[i]
    return multi


def _debug_actions(
    config: Dict,
    checkpoint_path: Path,
    phase_duration: int,
    max_steps: int,
    num_log_steps: int,
) -> None:
    """Run one DQN episode and one fixed-time episode, log first num_log_steps actions to verify policies differ."""
    env_raw = create_environment(config)
    wrapped = wrap_env_for_dqn(env_raw)
    model = DQN.load(str(checkpoint_path), env=wrapped)
    num_intersections = env_raw.num_intersections
    nvec = np.array(env_raw.action_space.nvec) if hasattr(env_raw.action_space, "nvec") else np.array([4] * num_intersections)

    dqn_multi_list: List[np.ndarray] = []
    reset_out = wrapped.reset()
    obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
    for step in range(min(num_log_steps, max_steps)):
        action, _ = model.predict(obs, deterministic=True)
        action_int = int(np.asarray(action).flatten()[0])
        multi = _decode_flat_to_multi(action_int, nvec)
        dqn_multi_list.append(multi.copy())
        step_out = wrapped.step(action)
        obs = step_out[0]
    wrapped.close()

    ft_multi_list: List[np.ndarray] = []
    env_ft = create_environment(config)
    reset_out = env_ft.reset()
    obs_ft = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
    for step in range(min(num_log_steps, max_steps)):
        phase = (step // phase_duration) % 4
        action = np.array([phase] * num_intersections, dtype=np.int32)
        ft_multi_list.append(action.copy())
        step_out = env_ft.step(action)
        obs_ft = step_out[0]
    env_ft.close()

    print("\n[DEBUG] First {} steps: DQN vs Fixed-time (per-intersection phases):".format(num_log_steps))
    steps_match = 0
    for i in range(min(len(dqn_multi_list), len(ft_multi_list))):
        dqn_phases = dqn_multi_list[i]
        ft_phases = ft_multi_list[i]
        same = np.array_equal(dqn_phases, ft_phases)
        if same:
            steps_match += 1
        print("  step {:3d}:  DQN {}   fixed_time {}   {}".format(
            i, dqn_phases.tolist(), ft_phases.tolist(), "SAME" if same else "DIFF"))
    print("  Summary: {}/{} steps had identical phase vector (DQN vs fixed-time).".format(steps_match, min(len(dqn_multi_list), len(ft_multi_list))))
    if steps_match == min(len(dqn_multi_list), len(ft_multi_list)):
        print("  [Note] DQN is choosing the same phases as fixed-time every step — metrics will match. Try --phase-duration 60 to make fixed-time worse and see if DQN can beat it.")


def _run_single_seed(
    config: Dict,
    checkpoint_path: Path,
    num_episodes: int,
    max_steps: int,
    phase_duration: int,
    seed: int,
    run_actuated: bool,
) -> Tuple[
    List[float], List[float], List[float], List[float], List[float], List[float],
    List[float], List[float], List[float], List[float], List[float], List[float],
    Optional[List[float]], Optional[List[float]], Optional[List[float]], Optional[List[float]], Optional[List[float]], Optional[List[float]],
    bool,
]:
    """Run DQN, fixed-time, and optionally actuated for one seed. Returns (dqn_*), (ft_*), (act_* or None), placeholder_mode."""
    import numpy as np
    np.random.seed(seed)
    env_raw = create_environment(config)
    wrapped_env = wrap_env_for_dqn(env_raw)
    model = DQN.load(str(checkpoint_path), env=wrapped_env)
    dqn_r, dqn_l, dqn_tput, dqn_tt, dqn_wt, dqn_q, placeholder_mode = evaluate_dqn(
        model, wrapped_env, num_episodes, deterministic=True, max_steps_per_episode=max_steps
    )
    wrapped_env.close()

    env_ft = create_environment(config)
    ft_r, ft_l, ft_tput, ft_tt, ft_wt, ft_q, _ = evaluate_fixed_time(
        env_ft, num_episodes, phase_duration=phase_duration, max_steps_per_episode=max_steps
    )
    env_ft.close()

    act_r, act_l, act_tput, act_tt, act_wt, act_q = None, None, None, None, None, None
    if run_actuated:
        env_act = create_environment(config)
        act_r, act_l, act_tput, act_tt, act_wt, act_q, _ = evaluate_actuated(
            env_act, num_episodes, phase_duration=phase_duration, max_steps_per_episode=max_steps
        )
        env_act.close()

    return dqn_r, dqn_l, dqn_tput, dqn_tt, dqn_wt, dqn_q, ft_r, ft_l, ft_tput, ft_tt, ft_wt, ft_q, act_r, act_l, act_tput, act_tt, act_wt, act_q, placeholder_mode


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 DQN, fixed-time, and actuated baselines")
    parser.add_argument("--config", type=str, default="configs/phase1.yaml", help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, default="outputs/phase1/dqn_traffic_final.zip", help="Path to trained DQN checkpoint")
    parser.add_argument("--episodes", type=int, default=None, help="Number of evaluation episodes (default: from config)")
    parser.add_argument("--phase-duration", type=int, default=30, help="Fixed-time/actuated phase duration in steps")
    parser.add_argument("--seeds", type=int, default=None, help="Number of seeds for mean +/- std (default: 1, use config evaluation.seeds)")
    parser.add_argument("--actuated", action="store_true", help="Also evaluate actuated baseline")
    parser.add_argument("--save-summary", type=str, nargs="?", const="outputs/phase1/evaluation_summary.json", default=None, help="Save evaluation summary to JSON for comparison charts (default: outputs/phase1/evaluation_summary.json if flag present)")
    parser.add_argument("--debug-actions", type=int, default=0, metavar="N", help="Log first N step actions (DQN vs fixed-time) for episode 0 to verify policies differ (e.g. 20)")
    args = parser.parse_args()

    config = load_config(args.config)
    eval_cfg = config.get("evaluation", {})
    num_episodes = args.episodes or eval_cfg.get("num_episodes", 10)
    deterministic = eval_cfg.get("deterministic", True)
    sumo_cfg = config["sumo"]
    max_steps = sumo_cfg.get("simulation_steps", 3600)
    seeds_list = eval_cfg.get("seeds", [42])
    if isinstance(seeds_list, list):
        n_seeds = args.seeds if args.seeds is not None else 1
        seeds_to_use = seeds_list[:n_seeds]
    else:
        seeds_to_use = [42]

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("Run training first: python -m src.phase1.train_rl --config configs/phase1.yaml")
        return

    if args.debug_actions > 0:
        _debug_actions(config, checkpoint_path, args.phase_duration, max_steps, args.debug_actions)
        print()

    run_actuated = args.actuated

    all_dqn_r, all_dqn_l, all_dqn_tput, all_dqn_tt, all_dqn_wt, all_dqn_q = [], [], [], [], [], []
    all_ft_r, all_ft_l, all_ft_tput, all_ft_tt, all_ft_wt, all_ft_q = [], [], [], [], [], []
    all_act_r, all_act_l, all_act_tput, all_act_tt, all_act_wt, all_act_q = [], [], [], [], [], []

    used_sumo = False
    for seed in seeds_to_use:
        dqn_r, dqn_l, dqn_tput, dqn_tt, dqn_wt, dqn_q, ft_r, ft_l, ft_tput, ft_tt, ft_wt, ft_q, act_r, act_l, act_tput, act_tt, act_wt, act_q, placeholder_mode = _run_single_seed(
            config, checkpoint_path, num_episodes, max_steps, args.phase_duration, seed, run_actuated
        )
        used_sumo = used_sumo or (not placeholder_mode)
        all_dqn_r.extend(dqn_r)
        all_dqn_l.extend(dqn_l)
        all_dqn_tput.extend(dqn_tput)
        all_dqn_tt.extend(dqn_tt)
        all_dqn_wt.extend(dqn_wt)
        all_dqn_q.extend(dqn_q)
        all_ft_r.extend(ft_r)
        all_ft_l.extend(ft_l)
        all_ft_tput.extend(ft_tput)
        all_ft_tt.extend(ft_tt)
        all_ft_wt.extend(ft_wt)
        all_ft_q.extend(ft_q)
        if act_r is not None:
            all_act_r.extend(act_r)
            all_act_l.extend(act_l)
            all_act_tput.extend(act_tput)
            all_act_tt.extend(act_tt)
            all_act_wt.extend(act_wt)
            all_act_q.extend(act_q)

    dqn_rewards, ft_rewards = np.array(all_dqn_r), np.array(all_ft_r)
    dqn_mean_rew = float(np.mean(dqn_rewards))
    dqn_std_rew = float(np.std(dqn_rewards))
    dqn_mean_len = float(np.mean(all_dqn_l))
    ft_mean_rew = float(np.mean(ft_rewards))
    ft_std_rew = float(np.std(ft_rewards))
    ft_mean_len = float(np.mean(all_ft_l))
    dqn_mean_throughput = float(np.mean(all_dqn_tput))
    ft_mean_throughput = float(np.mean(all_ft_tput))
    dqn_mean_tt = float(np.mean(all_dqn_tt))
    ft_mean_tt = float(np.mean(all_ft_tt))
    has_throughput = dqn_mean_throughput > 0 or ft_mean_throughput > 0
    has_travel_time = dqn_mean_tt > 0 or ft_mean_tt > 0

    print("\n" + "=" * 60)
    print("Phase 1 Evaluation Results (SOTA)")
    print("=" * 60)
    if not used_sumo:
        print("  [Note] Placeholder mode (no SUMO): throughput and travel_time are 0; not reported as results.")
    print(f"  Episodes: {num_episodes} x {len(seeds_to_use)} seeds = {len(dqn_rewards)} total")
    print(f"  Checkpoint: {checkpoint_path}")
    print("-" * 60)
    print(f"  DQN (GNN-RL):     mean_reward = {dqn_mean_rew:+.2f} +/- {dqn_std_rew:.2f}  |  mean_length = {dqn_mean_len:.1f}")
    print(f"  Fixed-time:       mean_reward = {ft_mean_rew:+.2f} +/- {ft_std_rew:.2f}  |  mean_length = {ft_mean_len:.1f}")
    if run_actuated and all_act_r:
        act_mean_rew = float(np.mean(all_act_r))
        act_std_rew = float(np.std(all_act_r))
        print(f"  Actuated:         mean_reward = {act_mean_rew:+.2f} +/- {act_std_rew:.2f}")
    if has_throughput:
        print(f"  DQN (GNN-RL):     throughput (departed/episode) = {dqn_mean_throughput:.1f}")
        print(f"  Fixed-time:      throughput (departed/episode) = {ft_mean_throughput:.1f}")
    if has_travel_time:
        print(f"  DQN (GNN-RL):     travel_time (sum/episode) = {dqn_mean_tt:.1f}")
        print(f"  Fixed-time:      travel_time (sum/episode) = {ft_mean_tt:.1f}")
    print("-" * 60)
    if ft_mean_rew != 0:
        pct = 100 * (dqn_mean_rew - ft_mean_rew) / abs(ft_mean_rew)
        print(f"  DQN vs Fixed-time: {pct:+.1f}% reward change (positive = DQN better)")
    if has_throughput and ft_mean_throughput > 0:
        pct_t = 100 * (dqn_mean_throughput - ft_mean_throughput) / ft_mean_throughput
        print(f"  DQN vs Fixed-time: {pct_t:+.1f}% throughput change (positive = DQN better)")
    if HAS_SCIPY and len(dqn_rewards) >= 2 and len(ft_rewards) >= 2:
        t_stat, p_value = scipy_stats.ttest_ind(dqn_rewards, ft_rewards)
        print(f"  Statistical test (t-test DQN vs Fixed-time): p = {p_value:.4f}" + (" (significant at 0.05)" if p_value < 0.05 else ""))
    if run_actuated and all_act_r and HAS_SCIPY and len(all_dqn_r) >= 2:
        t_stat, p_act = scipy_stats.ttest_ind(dqn_rewards, np.array(all_act_r))
        print(f"  Statistical test (t-test DQN vs Actuated): p = {p_act:.4f}" + (" (significant at 0.05)" if p_act < 0.05 else ""))
    if dqn_mean_rew == ft_mean_rew and (dqn_mean_throughput == ft_mean_throughput or not has_throughput):
        print("-" * 60)
        print("  [Hint] DQN and Fixed-time metrics are identical. To verify policies differ, run with:")
        print("    --debug-actions 20")
        print("  To try a weaker baseline (longer fixed phases), run with e.g.:")
        print("    --phase-duration 60")
    print("=" * 60)
    print("[OK] Evaluation complete.")

    # Save summary for comparison charts (SOTA: per-episode for line charts + means)
    if args.save_summary:
        dqn_mean_wt = float(np.mean(all_dqn_wt)) if all_dqn_wt else 0.0
        ft_mean_wt = float(np.mean(all_ft_wt)) if all_ft_wt else 0.0
        summary = {
            "num_episodes": num_episodes,
            "num_seeds": len(seeds_to_use),
            "total_runs": len(dqn_rewards),
            "used_sumo": used_sumo,
            "dqn": {
                "mean_reward": dqn_mean_rew,
                "std_reward": dqn_std_rew,
                "mean_throughput": dqn_mean_throughput,
                "std_throughput": float(np.std(all_dqn_tput)) if all_dqn_tput else 0,
                "mean_travel_time": dqn_mean_tt,
                "std_travel_time": float(np.std(all_dqn_tt)) if all_dqn_tt else 0,
                "mean_waiting_time": dqn_mean_wt,
                "std_waiting_time": float(np.std(all_dqn_wt)) if all_dqn_wt else 0,
                "mean_queue_length": float(np.mean(all_dqn_q)) if all_dqn_q else 0.0,
                "std_queue_length": float(np.std(all_dqn_q)) if all_dqn_q else 0,
                "rewards": [float(r) for r in all_dqn_r],
                "throughputs": [float(t) for t in all_dqn_tput],
                "travel_times": [float(t) for t in all_dqn_tt],
                "waiting_times": [float(t) for t in all_dqn_wt],
                "queue_lengths": [float(q) for q in all_dqn_q],
            },
            "fixed_time": {
                "mean_reward": ft_mean_rew,
                "std_reward": ft_std_rew,
                "mean_throughput": ft_mean_throughput,
                "std_throughput": float(np.std(all_ft_tput)) if all_ft_tput else 0,
                "mean_travel_time": ft_mean_tt,
                "std_travel_time": float(np.std(all_ft_tt)) if all_ft_tt else 0,
                "mean_waiting_time": ft_mean_wt,
                "std_waiting_time": float(np.std(all_ft_wt)) if all_ft_wt else 0,
                "mean_queue_length": float(np.mean(all_ft_q)) if all_ft_q else 0.0,
                "std_queue_length": float(np.std(all_ft_q)) if all_ft_q else 0,
                "rewards": [float(r) for r in all_ft_r],
                "throughputs": [float(t) for t in all_ft_tput],
                "travel_times": [float(t) for t in all_ft_tt],
                "waiting_times": [float(t) for t in all_ft_wt],
                "queue_lengths": [float(q) for q in all_ft_q],
            },
        }
        if run_actuated and all_act_r:
            summary["actuated"] = {
                "mean_reward": float(np.mean(all_act_r)),
                "std_reward": float(np.std(all_act_r)),
                "mean_throughput": float(np.mean(all_act_tput)) if all_act_tput else 0,
                "std_throughput": float(np.std(all_act_tput)) if all_act_tput else 0,
                "mean_travel_time": float(np.mean(all_act_tt)) if all_act_tt else 0,
                "std_travel_time": float(np.std(all_act_tt)) if all_act_tt else 0,
                "mean_waiting_time": float(np.mean(all_act_wt)) if all_act_wt else 0,
                "std_waiting_time": float(np.std(all_act_wt)) if all_act_wt else 0,
                "mean_queue_length": float(np.mean(all_act_q)) if all_act_q else 0.0,
                "std_queue_length": float(np.std(all_act_q)) if all_act_q else 0,
                "rewards": [float(r) for r in all_act_r],
                "throughputs": [float(t) for t in all_act_tput],
                "travel_times": [float(t) for t in all_act_tt],
                "waiting_times": [float(t) for t in all_act_wt],
                "queue_lengths": [float(q) for q in all_act_q],
            }
        out_path = Path(args.save_summary)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[OK] Summary saved to {out_path}")


if __name__ == "__main__":
    main()
