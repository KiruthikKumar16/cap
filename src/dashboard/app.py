import copy
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "configs" / "phase1.yaml"
DEFAULT_RESULTS = ROOT / "outputs" / "benchmark_results.json"
DEFAULT_EVAL_SUMMARY = ROOT / "outputs" / "phase1" / "evaluation_summary.json"
DEFAULT_STRESS_SUMMARY = ROOT / "outputs" / "phase3" / "adversarial_benchmark.json"
RUNS_DIR = ROOT / "outputs" / "dashboard_runs"
BEST_CHECKPOINT = "outputs/phase1/dqn_traffic_final.zip"
FIXED_NET_FILE = "data/raw/grid_5x5.net.xml"
FIXED_ROUTE_FILE = "data/raw/grid_5x5_medium.rou.xml"
FIXED_SUMOCFG_FILE = "data/raw/grid_5x5.sumocfg"

METRICS_META: Dict[str, Dict[str, Any]] = {
    "mean_reward": {"label": "Reward", "higher_is_better": True, "unit": "score"},
    "mean_throughput": {"label": "Throughput", "higher_is_better": True, "unit": "veh/h"},
    "mean_travel_time": {"label": "Travel Time", "higher_is_better": False, "unit": "s"},
    "mean_waiting_time": {"label": "Waiting Time", "higher_is_better": False, "unit": "s"},
    "mean_queue_length": {"label": "Queue Length", "higher_is_better": False, "unit": "vehicles"},
}


def _safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_nested(cfg: Dict[str, Any], keys: Tuple[str, ...], value: Any) -> None:
    cur = cfg
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _write_temp_config(base_cfg: Dict[str, Any], overrides: Dict[Tuple[str, ...], Any]) -> Path:
    cfg = copy.deepcopy(base_cfg)
    for keys, value in overrides.items():
        _set_nested(cfg, keys, value)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = RUNS_DIR / f"dashboard_config_{ts}.yaml"
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return target


def _run_command(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        raise RuntimeError(output.strip() or "Command failed.")
    return output.strip()


def _run_command_stream(
    cmd: List[str],
    log_slot,
    progress_bar,
    start: float,
    end: float,
    title: str,
) -> str:
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: List[str] = []
    if proc.stdout is None:
        raise RuntimeError(f"{title}: unable to read process output.")
    log_slot.code(f"[{title}] started...")
    for line in proc.stdout:
        lines.append(line.rstrip())
        tail = "\n".join(lines[-120:])
        log_slot.code(tail if tail else f"[{title}] running...")
        progress_bar.progress(min(end, start + (end - start) * 0.85))
    return_code = proc.wait()
    progress_bar.progress(end)
    output = "\n".join(lines).strip()
    if return_code != 0:
        raise RuntimeError(output or f"{title} failed.")
    return output


def _resolve_checkpoint() -> Path:
    candidates = [
        ROOT / BEST_CHECKPOINT,
        ROOT / "best_model_stage_2.zip",
        ROOT / "marl_ppo_traffic.zip",
        ROOT / "outputs/phase1/dqn_traffic_final.zip",
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path()


def _run_benchmark(config: Path, checkpoint: str, episodes: int) -> str:
    return _run_command(
        [
            sys.executable,
            "scripts/run_benchmarks.py",
            "--config",
            _safe_rel(config),
            "--checkpoint",
            checkpoint,
            "--episodes",
            str(episodes),
        ]
    )


def _run_detailed_eval(config: Path, checkpoint: str, episodes: int, include_random: bool, include_fixed: bool) -> str:
    cmd = [
        sys.executable,
        "src/phase1/evaluate.py",
        "--config",
        _safe_rel(config),
        "--checkpoint",
        checkpoint,
        "--episodes",
        str(episodes),
        "--save-summary",
        _safe_rel(DEFAULT_EVAL_SUMMARY),
    ]
    if include_fixed:
        cmd.append("--fixed-time")
    if include_random:
        cmd.append("--random")
    return _run_command(cmd)


def _run_stress_eval(config: Path, checkpoint: str, episodes: int, sensor_noise_rate: float) -> str:
    return _run_command(
        [
            sys.executable,
            "scripts/accident_injection.py",
            "--config",
            _safe_rel(config),
            "--checkpoint",
            checkpoint,
            "--episodes",
            str(episodes),
            "--sensor-noise-rate",
            str(sensor_noise_rate),
        ]
    )


def _precheck_checkpoint_compatibility(config: Path, checkpoint: Path) -> Tuple[bool, str]:
    try:
        from stable_baselines3 import PPO
        from src.phase1.marl_traffic_env import MARLTrafficEnv
    except Exception as exc:
        return False, (
            f"Precheck import failed: {exc}. "
            "Ensure dashboard is launched from project environment and dependencies are installed."
        )

    try:
        cfg = _load_yaml(config)
        env = MARLTrafficEnv(cfg)
        model = PPO.load(str(checkpoint))
        model_obs = getattr(model, "observation_space", None)
        env_obs = getattr(env, "observation_space", None)
        model_act = getattr(model, "action_space", None)
        env_act = getattr(env, "action_space", None)
        obs_ok = str(model_obs) == str(env_obs)
        act_ok = str(model_act) == str(env_act)
        env.close()
        if obs_ok and act_ok:
            return True, f"Compatible. obs={model_obs}, action={model_act}"
        return (
            False,
            "Mismatch detected. "
            f"checkpoint obs={model_obs}, env obs={env_obs}, checkpoint action={model_act}, env action={env_act}",
        )
    except Exception as exc:
        msg = str(exc)
        if "unexpected keyword argument 'use_sde'" in msg or "DQNPolicy.__init__" in msg:
            return (
                False,
                "Checkpoint algorithm mismatch: selected checkpoint is not a PPO checkpoint "
                "(possible DQN/SB3 format mismatch). Please provide a PPO checkpoint matching this configuration.",
            )
        return False, f"Precheck failed: {exc}"


def _flatten_results(raw: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_name, metrics in raw.items():
        if not isinstance(metrics, dict) or "mean_reward" not in metrics:
            continue
        row = {"model": model_name}
        for metric in METRICS_META:
            row[metric] = metrics.get(metric)
        rows.append(row)
    return pd.DataFrame(rows)


def _latency_df(raw: Dict[str, Any]) -> pd.DataFrame:
    payload = raw.get("latency_ms_per_step", [])
    if isinstance(payload, list) and payload:
        return pd.DataFrame(payload)
    return pd.DataFrame()


def _score_models(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    total = pd.Series(0.0, index=scored.index)
    for metric, meta in METRICS_META.items():
        values = scored[metric].astype(float)
        span = values.max() - values.min()
        if span == 0:
            norm = pd.Series(0.5, index=scored.index)
        else:
            if meta["higher_is_better"]:
                norm = (values - values.min()) / span
            else:
                norm = (values.max() - values) / span
        scored[f"{metric}_norm"] = norm
        total += norm
    scored["overall_score"] = total / len(METRICS_META)
    return scored.sort_values("overall_score", ascending=False)


def _render_overview(df: pd.DataFrame, lat_df: pd.DataFrame) -> None:
    scored = _score_models(df)
    leader = scored.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Overall Model", leader["model"])
    c2.metric("Top Overall Score", f"{leader['overall_score']:.3f}")
    c3.metric("Models Compared", int(df.shape[0]))
    k1, k2, k3, k4, k5 = st.columns(5)
    for i, metric in enumerate(METRICS_META.keys()):
        best_row = df.sort_values(metric, ascending=not METRICS_META[metric]["higher_is_better"]).iloc[0]
        lbl = f"Best {METRICS_META[metric]['label']}"
        val = f"{best_row[metric]:.2f} {METRICS_META[metric]['unit']}"
        if i == 0:
            k1.metric(lbl, val, best_row["model"])
        elif i == 1:
            k2.metric(lbl, val, best_row["model"])
        elif i == 2:
            k3.metric(lbl, val, best_row["model"])
        elif i == 3:
            k4.metric(lbl, val, best_row["model"])
        else:
            k5.metric(lbl, val, best_row["model"])

    st.subheader("Overall Ranking (Normalized Multi-Metric Score)")
    st.dataframe(
        scored[["model", "overall_score"] + list(METRICS_META.keys())],
        use_container_width=True,
    )

    if not lat_df.empty and "mean_ms" in lat_df:
        st.subheader("Inference Latency")
        st.dataframe(lat_df, use_container_width=True)
        lat_chart = lat_df[["model", "mean_ms"]].dropna().set_index("model")
        st.bar_chart(lat_chart)


def _render_metrics(df: pd.DataFrame) -> None:
    st.subheader("Model Comparison Table")
    st.dataframe(df.sort_values("mean_reward", ascending=False), use_container_width=True)

    st.subheader("Metric Charts")
    for metric, meta in METRICS_META.items():
        st.markdown(f"**{meta['label']} ({meta['unit']})**")
        chart_df = df[["model", metric]].dropna()
        chart_df = chart_df.sort_values(metric, ascending=not meta["higher_is_better"])
        st.bar_chart(chart_df.set_index("model"))

    st.subheader("Cross-Metric View")
    selected_models = st.multiselect("Models to plot", options=df["model"].tolist(), default=df["model"].tolist())
    selected_metrics = st.multiselect(
        "Metrics to plot",
        options=list(METRICS_META.keys()),
        default=list(METRICS_META.keys()),
    )
    if selected_models and selected_metrics:
        pivot_df = df[df["model"].isin(selected_models)][["model"] + selected_metrics].set_index("model")
        st.line_chart(pivot_df.T)


def _render_episode_analysis(eval_summary_path: Path) -> None:
    st.subheader("Per-Episode Analysis")
    if not eval_summary_path.exists():
        st.info("Detailed evaluation summary not found yet. Run with 'Also run detailed episode evaluation'.")
        return

    with eval_summary_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    available = [k for k in ["dqn", "fixed_time", "actuated", "random"] if k in raw]
    if not available:
        st.warning("No per-episode model blocks found in evaluation summary.")
        return

    labels = {
        "dqn": "MAPPO-STGNN",
        "fixed_time": "Fixed-Time",
        "actuated": "Actuated",
        "random": "Random",
    }
    metric_map = {
        "rewards": "Reward",
        "throughputs": "Throughput",
        "travel_times": "Travel Time",
        "waiting_times": "Waiting Time",
        "queue_lengths": "Queue Length",
    }

    choice = st.selectbox("Episode metric", options=list(metric_map.keys()), format_func=lambda x: metric_map[x])
    episode_rows: List[Dict[str, Any]] = []
    for model_key in available:
        values = raw.get(model_key, {}).get(choice, [])
        for idx, value in enumerate(values):
            episode_rows.append({"episode": idx + 1, "model": labels.get(model_key, model_key), "value": value})

    if not episode_rows:
        st.info("No episode-level values available for selected metric.")
        return

    episode_df = pd.DataFrame(episode_rows)
    st.line_chart(episode_df.pivot(index="episode", columns="model", values="value"))
    st.dataframe(episode_df, use_container_width=True)


def _render_stress_analysis(stress_summary_path: Path) -> None:
    st.subheader("Adversarial Stress Analysis")
    if not stress_summary_path.exists():
        st.info("Stress summary not found yet. Enable stress test to generate this.")
        return

    with stress_summary_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    deg = raw.get("degradation_limits_pct", {})
    if not isinstance(deg, dict) or not deg:
        st.warning("No degradation metrics found in stress summary.")
        return

    rows: List[Dict[str, Any]] = []
    for model_name, metrics in deg.items():
        rows.append(
            {
                "model": model_name.upper(),
                "throughput_drop_pct": metrics.get("throughput_drop_pct"),
                "waiting_time_increase_pct": metrics.get("waiting_time_increase_pct"),
                "queue_length_increase_pct": metrics.get("queue_length_increase_pct"),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.set_index("model"))


def _render_data_sanity_warnings(df: pd.DataFrame, eval_summary_path: Path) -> None:
    issues: List[str] = []
    checks: List[str] = []
    for metric in METRICS_META.keys():
        vals = df[metric].dropna()
        if vals.empty:
            issues.append(f"`{metric}` has no values.")
            continue
        if vals.nunique() <= 1:
            issues.append(f"`{metric}` is constant across models (possible degenerate run).")
        if (vals == 0).all():
            issues.append(f"`{metric}` is all zero.")
        if vals.nunique() > 1 and not (vals == 0).all():
            checks.append(f"`{metric}` variability check passed.")
    if eval_summary_path.exists():
        try:
            raw = json.loads(eval_summary_path.read_text(encoding="utf-8"))
            dqn_rewards = raw.get("dqn", {}).get("rewards", [])
            if dqn_rewards and len(set([round(float(x), 6) for x in dqn_rewards])) <= 1:
                issues.append("Per-episode rewards are constant; verify randomness/seeds and SUMO dynamics.")
            else:
                checks.append("Per-episode reward variability check passed.")
            dqn_wait = raw.get("dqn", {}).get("waiting_times", [])
            if dqn_wait and len(set([round(float(x), 6) for x in dqn_wait])) <= 1:
                issues.append("Per-episode waiting time is constant; check scenario dynamics.")
            elif dqn_wait:
                checks.append("Per-episode waiting time variability check passed.")
        except Exception:
            issues.append("Could not parse evaluation summary for sanity checks.")
    if issues:
        st.warning("Data sanity warnings:\n- " + "\n- ".join(issues))
    else:
        st.success("Data sanity checks passed.")
    if checks:
        st.info("Checks performed:\n- " + "\n- ".join(checks))


def _build_demo_report(
    df: pd.DataFrame,
    lat_df: pd.DataFrame,
    config_path: Path,
    checkpoint_path: Path,
    run_stress: bool,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Adaptive Traffic Control Evaluation Report",
        "",
        f"- Generated: {now}",
        f"- Config: `{_safe_rel(config_path)}`",
        f"- Checkpoint: `{_safe_rel(checkpoint_path)}`",
        f"- Stress test enabled: `{run_stress}`",
        "",
        "## Benchmark Metrics",
        "",
        df.to_markdown(index=False),
        "",
    ]
    if not lat_df.empty:
        lines.extend(["## Latency Metrics", "", lat_df.to_markdown(index=False), ""])
    return "\n".join(lines)


def main() -> None:
    st.set_page_config(page_title="Adaptive Traffic Control Evaluation Suite", layout="wide")
    st.title("Adaptive Traffic Control Evaluation Suite")
    st.caption(
        "Evaluate MAPPO-STGNN against CoLight and NSTLight under normal and stress conditions."
    )

    with st.sidebar:
        st.header("Experiment Setup")
        preset = st.selectbox("Scenario preset", options=["Normal", "High Demand", "Stress Demo"], index=0)
        preset_map = {
            "Normal": {"demand": "medium", "noise": 0.10, "stress": True, "steps": 3600},
            "High Demand": {"demand": "high", "noise": 0.10, "stress": True, "steps": 3600},
            "Stress Demo": {"demand": "high", "noise": 0.20, "stress": True, "steps": 4200},
        }
        preset_values = preset_map[preset]
        episodes = st.number_input(
            "Benchmark episodes [count]",
            min_value=1,
            max_value=500,
            value=3 if preset != "Stress Demo" else 2,
            step=1,
            help="Number of full simulation episodes used for aggregate benchmark metrics.",
        )
        detailed_required = st.checkbox("Detailed episode evaluation (required)", value=True, disabled=True)
        detailed_episodes = st.number_input(
            "Detailed eval episodes [count]",
            min_value=1,
            max_value=500,
            value=25,
            step=1,
            help="Per-episode analysis run count used for trend plots and reviewer tables.",
        )
        run_stress = st.checkbox("Run adversarial stress test", value=True)
        stress_noise = st.slider(
            "Sensor noise level (stress) [ratio]",
            min_value=0.0,
            max_value=0.5,
            value=float(preset_values["noise"]),
            step=0.01,
            help="0.10 means 10% effective sensor corruption in stress mode.",
        )

        st.markdown("### SUMO Runtime Controls")
        simulation_steps = st.number_input(
            "SUMO simulation steps [steps]",
            min_value=100,
            max_value=20000,
            value=int(preset_values["steps"]),
            step=100,
            help="Total simulation horizon per episode. 3600 steps ~= 1 hour at 1-second step length.",
        )
        step_length = st.number_input(
            "SUMO step length [seconds/step]",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Simulation time represented by each SUMO step.",
        )
        gui = st.checkbox("SUMO GUI (mandatory)", value=True, disabled=True)
        st.caption("Algorithm is fixed to PPO (best model). Baseline comparison includes CoLight and NSTLight.")

        with st.expander("Traffic Demand + Reward Weights"):
            demand_level = st.selectbox(
                "Traffic demand level",
                options=["low", "medium", "high"],
                index=["low", "medium", "high"].index(preset_values["demand"]),
                help="Route demand profile used during simulation.",
            )
            waiting_w = st.number_input(
                "Waiting time weight [reward units / second]",
                value=0.1,
                step=0.01,
                format="%.4f",
                help="Penalty multiplier applied to cumulative waiting time.",
            )
            queue_w = st.number_input(
                "Queue length weight [reward units / vehicle]",
                value=0.05,
                step=0.01,
                format="%.4f",
                help="Penalty multiplier applied to queue length.",
            )
            throughput_w = st.number_input(
                "Throughput weight [reward units / arrived vehicle]",
                value=0.0,
                step=0.01,
                format="%.4f",
                help="Reward bonus per arrived vehicle (flow incentive).",
            )
            pressure_w = st.number_input(
                "Pressure weight [reward units / pressure score]",
                value=0.0002,
                step=0.0001,
                format="%.4f",
                help="Weight for pressure-based control signal.",
            )
            speed_w = st.number_input(
                "Speed reward weight [reward units / normalized speed]",
                value=0.5,
                step=0.1,
                format="%.4f",
                help="Bonus for higher average speed to reduce stop-and-go behavior.",
            )

        run_now = st.button("Run Evaluation Suite", use_container_width=True)
        precheck_now = st.button("Checkpoint Compatibility Precheck", use_container_width=True)
        load_latest = st.button("Load Latest Results", use_container_width=True)

    run_logs = ""
    detail_logs = ""
    stress_logs = ""
    active_config_path = DEFAULT_CONFIG
    active_checkpoint_path = Path()

    if precheck_now:
        checkpoint_path = _resolve_checkpoint()
        if not checkpoint_path:
            st.error("No checkpoint found for compatibility precheck.")
            return
        cfg_path = DEFAULT_CONFIG
        base_cfg = _load_yaml(cfg_path)
        overrides = {
            ("sumo", "simulation_steps"): int(simulation_steps),
            ("sumo", "step_length"): float(step_length),
            ("sumo", "gui"): bool(gui),
            ("sumo", "net_file"): FIXED_NET_FILE,
            ("sumo", "route_file"): FIXED_ROUTE_FILE,
            ("sumo", "config_file"): FIXED_SUMOCFG_FILE,
            ("rl", "algorithm"): "PPO",
        }
        temp_cfg = _write_temp_config(base_cfg, overrides)
        ok, msg = _precheck_checkpoint_compatibility(temp_cfg, checkpoint_path)
        if ok:
            st.success(f"Compatibility precheck passed. {msg}")
        else:
            st.error(f"Compatibility precheck failed. {msg}")
        return

    if run_now:
        try:
            progress = st.progress(0.01)
            stage_slot = st.empty()
            stage_slot.info("Stage 0/3: Running compatibility precheck...")
            checkpoint_path = _resolve_checkpoint()
            if not checkpoint_path:
                st.error(
                    "No PPO checkpoint found. Expected one of: "
                    "`outputs/phase1/dqn_traffic_final.zip`, `best_model_stage_2.zip`, or `marl_ppo_traffic.zip`. "
                    "Train Phase 1 first, then rerun."
                )
                return
            cfg_path = DEFAULT_CONFIG
            if not cfg_path.exists():
                st.error(f"Config not found: {cfg_path}")
                return
            base_cfg = _load_yaml(cfg_path)
            route_map = {
                "low": "data/raw/grid_5x5_low.rou.xml",
                "medium": FIXED_ROUTE_FILE,
                "high": "data/raw/grid_5x5_high.rou.xml",
            }
            overrides = {
                ("sumo", "simulation_steps"): int(simulation_steps),
                ("sumo", "step_length"): float(step_length),
                ("sumo", "gui"): bool(gui),
                ("sumo", "net_file"): FIXED_NET_FILE,
                ("sumo", "route_file"): route_map[demand_level],
                ("sumo", "config_file"): FIXED_SUMOCFG_FILE,
                ("rl", "algorithm"): "PPO",
                ("evaluation", "num_episodes"): int(episodes),
                ("reward", "waiting_time_weight"): float(waiting_w),
                ("reward", "queue_length_weight"): float(queue_w),
                ("reward", "throughput_weight"): float(throughput_w),
                ("reward", "pressure_weight"): float(pressure_w),
                ("reward", "speed_reward_weight"): float(speed_w),
            }
            temp_cfg = _write_temp_config(base_cfg, overrides)
            active_config_path = temp_cfg
            active_checkpoint_path = checkpoint_path

            # Mandatory compatibility gate before long benchmark jobs.
            pre_ok, pre_msg = _precheck_checkpoint_compatibility(temp_cfg, checkpoint_path)
            if not pre_ok:
                progress.progress(1.0)
                st.error(
                    "Compatibility check failed before execution. "
                    "Execution has been stopped because the current checkpoint is incompatible with the selected configuration.\n\n"
                    f"Details: {pre_msg}"
                )
                st.info("Use a PPO checkpoint trained with the same scenario and configuration.")
                return
            progress.progress(0.05)
            st.success(f"Compatibility check passed. {pre_msg}")

            log_slot = st.empty()
            stage_slot.info("Stage 1/3: Running benchmark comparison...")
            run_logs = _run_command_stream(
                [
                    sys.executable,
                    "scripts/run_benchmarks.py",
                    "--config",
                    _safe_rel(temp_cfg),
                    "--checkpoint",
                    _safe_rel(checkpoint_path),
                    "--episodes",
                    str(int(episodes)),
                ],
                log_slot=log_slot,
                progress_bar=progress,
                start=0.05,
                end=0.35,
                title="Benchmark",
            )
            st.success(f"Benchmark completed with config `{_safe_rel(temp_cfg)}`.")

            stage_slot.info("Stage 2/3: Running detailed episode evaluation...")
            detail_logs = _run_command_stream(
                [
                    sys.executable,
                    "src/phase1/evaluate.py",
                    "--config",
                    _safe_rel(temp_cfg),
                    "--checkpoint",
                    _safe_rel(checkpoint_path),
                    "--episodes",
                    str(int(detailed_episodes)),
                    "--save-summary",
                    _safe_rel(DEFAULT_EVAL_SUMMARY),
                    "--fixed-time",
                    "--random",
                ],
                log_slot=log_slot,
                progress_bar=progress,
                start=0.35,
                end=0.75,
                title="Detailed Evaluation",
            )
            st.success("Detailed episode evaluation completed.")

            if run_stress:
                stage_slot.info("Stage 3/3: Running adversarial stress test...")
                stress_logs = _run_command_stream(
                    [
                        sys.executable,
                        "scripts/accident_injection.py",
                        "--config",
                        _safe_rel(temp_cfg),
                        "--checkpoint",
                        _safe_rel(checkpoint_path),
                        "--episodes",
                        str(int(episodes)),
                        "--sensor-noise-rate",
                        str(float(stress_noise)),
                    ],
                    log_slot=log_slot,
                    progress_bar=progress,
                    start=0.75,
                    end=1.0,
                    title="Stress Test",
                )
                st.success("Adversarial stress test completed.")
            else:
                progress.progress(1.0)
                stage_slot.info("Stage 3/3: Stress test skipped by user. Run complete.")
            if run_stress:
                stage_slot.success("All stages completed successfully.")
        except Exception as exc:
            try:
                progress.progress(1.0)
            except Exception:
                pass
            st.error(f"Execution failed: {exc}")
            return

    if run_logs:
        with st.expander("Benchmark Logs", expanded=False):
            st.code(run_logs)
    if detail_logs:
        with st.expander("Detailed Eval Logs", expanded=False):
            st.code(detail_logs)
    if stress_logs:
        with st.expander("Stress Test Logs", expanded=False):
            st.code(stress_logs)

    if not (load_latest or run_now):
        st.info(
            "Choose scenario settings in the left panel, then click `Run Evaluation Suite` "
            "or `Load Latest Results` to review previously generated outputs."
        )
        st.markdown(
            "- Fixed setup: PPO (`best_model_stage_2.zip`) on `grid_5x5` SUMO scenario\n"
            "- Includes benchmark, mandatory detailed episode evaluation, and optional stress test"
        )
        return

    if not DEFAULT_RESULTS.exists():
        st.warning(f"Results file not found: {DEFAULT_RESULTS}")
        return

    with DEFAULT_RESULTS.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    df = _flatten_results(raw)
    if df.empty:
        st.warning("No model metrics found in benchmark results.")
        return

    lat_df = _latency_df(raw)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Executive Overview", "All Metrics", "Episode Trends", "Stress Test", "Export"])

    with tab1:
        _render_overview(df, lat_df)
        _render_data_sanity_warnings(df, DEFAULT_EVAL_SUMMARY)

    with tab2:
        _render_metrics(df)

    with tab3:
        _render_episode_analysis(DEFAULT_EVAL_SUMMARY)

    with tab4:
        _render_stress_analysis(DEFAULT_STRESS_SUMMARY)

    with tab5:
        st.subheader("Export Artifacts")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download benchmark table (CSV)", data=csv_bytes, file_name="benchmark_table.csv", mime="text/csv")
        st.download_button("Download raw benchmark JSON", data=json.dumps(raw, indent=2), file_name="benchmark_results.json", mime="application/json")
        if DEFAULT_EVAL_SUMMARY.exists():
            st.download_button(
                "Download episode summary JSON",
                data=DEFAULT_EVAL_SUMMARY.read_text(encoding="utf-8"),
                file_name="evaluation_summary.json",
                mime="application/json",
            )
        report_md = _build_demo_report(
            df=df,
            lat_df=lat_df,
            config_path=active_config_path if run_now else DEFAULT_CONFIG,
            checkpoint_path=active_checkpoint_path if active_checkpoint_path else _resolve_checkpoint(),
            run_stress=run_stress,
        )
        st.download_button(
            "Download demo report (Markdown)",
            data=report_md,
            file_name="demo_report.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()

