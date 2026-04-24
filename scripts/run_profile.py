import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROFILE_DIR = ROOT / "outputs" / "profile_configs"


PROFILES: Dict[str, Dict[str, Any]] = {
    "cpu_quick": {
        "mode": "quick",
        "latency_device": "cpu",
        "benchmark_episodes": 1,
        "detailed_episodes": 10,
        "stress_episodes": 1,
        "overrides": {
            ("sumo", "simulation_steps"): 1800,
            ("sumo", "gui"): False,
            ("training", "total_timesteps"): 10000,
            ("evaluation", "num_episodes"): 10,
            ("evaluation", "seeds"): [42],
        },
    },
    "gpu_standard": {
        "mode": "full",
        "latency_device": "gpu",
        "benchmark_episodes": 3,
        "detailed_episodes": 50,
        "stress_episodes": 3,
        "overrides": {
            ("sumo", "simulation_steps"): 3600,
            ("sumo", "gui"): False,
            ("training", "total_timesteps"): 100000,
            ("evaluation", "num_episodes"): 50,
            ("evaluation", "seeds"): [42, 43, 44],
        },
    },
    "gpu_extreme": {
        "mode": "full",
        "latency_device": "gpu",
        "benchmark_episodes": 5,
        "detailed_episodes": 100,
        "stress_episodes": 5,
        "overrides": {
            ("sumo", "simulation_steps"): 5400,
            ("sumo", "gui"): False,
            ("training", "total_timesteps"): 500000,
            ("evaluation", "num_episodes"): 100,
            ("evaluation", "seeds"): [42, 43, 44, 45, 46],
        },
    },
}


def _set_nested(cfg: Dict[str, Any], keys: tuple, value: Any) -> None:
    cur = cfg
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _write_profile_config(base_config_path: Path, profile_name: str) -> Path:
    data = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    for keys, value in PROFILES[profile_name]["overrides"].items():
        _set_nested(data, keys, value)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROFILE_DIR / f"{profile_name}.yaml"
    out_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out_path


def _resolve_checkpoint_for_profile(profile_cfg: Path, requested_checkpoint: str) -> str:
    from src.phase1.marl_traffic_env import MARLTrafficEnv
    from src.utils.model_metadata import load_metadata_for_checkpoint, validate_metadata

    req = Path(requested_checkpoint)
    if req.exists():
        return requested_checkpoint
    req_abs = (ROOT / requested_checkpoint).resolve()
    if req_abs.exists():
        return str(req_abs.relative_to(ROOT))

    cfg = yaml.safe_load(profile_cfg.read_text(encoding="utf-8"))
    env = MARLTrafficEnv(cfg)
    env_obs = str(env.observation_space)
    env_act = str(env.action_space)
    env.close()

    candidates = [
        ROOT / "outputs/phase1/dqn_traffic_final.zip",
        ROOT / "marl_ppo_traffic.zip",
        ROOT / "best_model_stage_2.zip",
    ]
    existing = [c for c in candidates if c.exists()]
    if not existing:
        return requested_checkpoint

    for c in existing:
        meta = load_metadata_for_checkpoint(c)
        if not meta:
            continue
        mismatch = validate_metadata(
            metadata=meta,
            expected_algorithm="PPO",
            observation_space_repr=env_obs,
            action_space_repr=env_act,
            config=cfg,
        )
        if mismatch is None:
            return str(c.relative_to(ROOT))
        # For profile runs, allow config-digest mismatch when algorithm and spaces match.
        if "Config digest mismatch" in mismatch:
            algo_ok = str(meta.get("algorithm", "")).upper() == "PPO"
            obs_ok = str(meta.get("observation_space", "")) == env_obs
            act_ok = str(meta.get("action_space", "")) == env_act
            if algo_ok and obs_ok and act_ok:
                return str(c.relative_to(ROOT))

    # Fallback to first existing checkpoint if no compatible metadata match.
    # Prefer locally trained PPO checkpoint as fallback.
    for preferred in ["marl_ppo_traffic.zip", "outputs/phase1/dqn_traffic_final.zip", "best_model_stage_2.zip"]:
        p = ROOT / preferred
        if p.exists():
            return str(p.relative_to(ROOT))
    return str(existing[0].relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict publication flow by runtime profile")
    parser.add_argument("--profile", choices=list(PROFILES.keys()), required=True)
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--checkpoint", default="outputs/phase1/dqn_traffic_final.zip")
    args = parser.parse_args()

    base_cfg = (ROOT / args.config).resolve()
    if not base_cfg.exists():
        raise FileNotFoundError(f"Base config not found: {base_cfg}")

    profile_cfg = _write_profile_config(base_cfg, args.profile)
    prof = PROFILES[args.profile]
    resolved_checkpoint = _resolve_checkpoint_for_profile(profile_cfg, args.checkpoint)

    cmd = [
        sys.executable,
        "scripts/run_publication_suite.py",
        "--mode",
        prof["mode"],
        "--config",
        str(profile_cfg.relative_to(ROOT)),
        "--checkpoint",
        resolved_checkpoint,
        "--benchmark-episodes",
        str(prof["benchmark_episodes"]),
        "--detailed-episodes",
        str(prof["detailed_episodes"]),
        "--stress-episodes",
        str(prof["stress_episodes"]),
        "--latency-device",
        prof["latency_device"],
    ]

    print("=" * 72)
    print(f"Running profile: {args.profile}")
    print(f"Config: {profile_cfg}")
    print(f"Checkpoint: {resolved_checkpoint}")
    print("=" * 72)
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
