#!/usr/bin/env python3
"""Probe whether signal actions actually affect SUMO metrics.

This is a diagnostic, not a benchmark. It runs a few forced policies on the
same config and reports whether action application and traffic metrics differ.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.phase1.marl_traffic_env import MARLTrafficEnv


def _scalar(info: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = info.get(key, default)
    try:
        return float(np.asarray(value).reshape(-1)[0])
    except Exception:
        return float(default)


def _policy_action(policy: str, step: int, num_agents: int) -> np.ndarray:
    if policy == "all_phase_0":
        return np.zeros(num_agents, dtype=np.int32)
    if policy == "all_phase_2":
        return np.full(num_agents, 2, dtype=np.int32)
    if policy == "fixed_cycle":
        phase = 0 if (step // 30) % 2 == 0 else 2
        return np.full(num_agents, phase, dtype=np.int32)
    if policy == "random":
        rng = np.random.default_rng(12345 + step)
        return rng.integers(0, 4, size=num_agents, dtype=np.int32)
    raise ValueError(f"Unknown policy: {policy}")


def _run_policy(config: dict[str, Any], policy: str, steps: int) -> dict[str, Any]:
    env = MARLTrafficEnv(config)
    try:
        env.reset()
        base_env = env.env
        num_agents = getattr(base_env, "num_agents", env.num_envs)
        total_reward = 0.0
        total_arrived = 0.0
        total_wait = 0.0
        total_queue = 0.0
        applied_counts: list[int] = []
        skipped_reasons: list[str | None] = []
        completed_steps = 0

        for step in range(steps):
            action = _policy_action(policy, step, num_agents)
            step_out = env.step(action)
            obs, reward, terminated, truncated, info = step_out
            info_dict = info[0] if isinstance(info, list) and info else info
            if not isinstance(info_dict, dict):
                info_dict = {}
            total_reward += float(np.asarray(reward).reshape(-1)[0])
            total_arrived += _scalar(info_dict, "step_arrived_vehicles")
            total_wait += _scalar(info_dict, "step_total_waiting_time")
            total_queue += _scalar(info_dict, "step_total_queue_length")
            applied_counts.append(int(_scalar(info_dict, "actions_applied_count")))
            skipped_reasons.append(info_dict.get("actions_skipped_reason"))
            completed_steps += 1
            if np.any(terminated) or np.any(truncated):
                break

        return {
            "policy": policy,
            "steps": completed_steps,
            "mean_reward": total_reward / max(1, completed_steps),
            "throughput": total_arrived,
            "mean_waiting_time": total_wait / max(1, completed_steps),
            "mean_queue_length": total_queue / max(1, completed_steps),
            "min_actions_applied": min(applied_counts) if applied_counts else 0,
            "max_actions_applied": max(applied_counts) if applied_counts else 0,
            "skipped_reasons": sorted({str(reason) for reason in skipped_reasons if reason is not None}),
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose whether SUMO responds to forced signal actions.")
    parser.add_argument("--config", default="configs/phase1_colab.yaml")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output", default="results/action_effect_probe.json")
    args = parser.parse_args()

    with (ROOT / args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.setdefault("sumo", {})["simulation_steps"] = args.steps

    policies = ["all_phase_0", "all_phase_2", "fixed_cycle", "random"]
    results = [_run_policy(config, policy, args.steps) for policy in policies]

    report = {
        "artifact_type": "action_effect_probe",
        "config": args.config,
        "steps": args.steps,
        "results": results,
        "honesty_note": "This diagnostic checks action application and metric sensitivity. It is not model evidence.",
    }
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
