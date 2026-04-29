import copy
import base64
import json
import subprocess
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "configs" / "phase1.yaml"
DEFAULT_RESULTS = ROOT / "outputs" / "benchmark_results.json"
DEFAULT_MEDIA = ROOT / "outputs" / "dashboard_media.json"
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

FEATURED_MODELS = ["CoLight", "NSTLight", "MAPPO-STGNN"]
CHART_COLORS = ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#7c3aed", "#64748b", "#14b8a6"]


def _streamlit_fragment(func):
    fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
    if fragment is None:
        return func
    return fragment(func)


def _safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _resolve_user_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


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


def _align_config_with_checkpoint_metadata(base_cfg: Dict[str, Any], checkpoint_path: Path) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    try:
        from src.utils.model_metadata import load_metadata_for_checkpoint
    except Exception:
        return cfg

    metadata = load_metadata_for_checkpoint(checkpoint_path)
    if not metadata:
        return cfg

    for key in ["model", "sumo", "reward"]:
        block = metadata.get(key)
        if isinstance(block, dict) and block:
            cfg[key] = copy.deepcopy(block)

    eval_block = metadata.get("evaluation")
    if isinstance(eval_block, dict) and eval_block:
        cfg.setdefault("evaluation", {})
        cfg["evaluation"].update(copy.deepcopy(eval_block))

    cfg.setdefault("rl", {})
    cfg["rl"]["algorithm"] = str(metadata.get("algorithm", cfg["rl"].get("algorithm", "PPO")))
    return cfg


def _run_command(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        raise RuntimeError(output.strip() or "Command failed.")
    return output.strip()


def _probe_python_runtime(python_exe: Path) -> Dict[str, Any]:
    if not python_exe.exists():
        return {
            "python": str(python_exe),
            "exists": False,
            "cuda_available": False,
            "torch_import_ok": False,
        }
    code = (
        "import json,sys\n"
        "payload={'python':sys.executable,'exists':True,'cuda_available':False,'torch_import_ok':False}\n"
        "try:\n"
        " import torch\n"
        " payload['torch_import_ok']=True\n"
        " payload['torch_version']=torch.__version__\n"
        " payload['cuda_available']=bool(torch.cuda.is_available())\n"
        " payload['device_count']=int(torch.cuda.device_count()) if torch.cuda.is_available() else 0\n"
        " payload['device_name']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''\n"
        "except Exception as exc:\n"
        " payload['torch_error']=str(exc)\n"
        "print(json.dumps(payload))\n"
    )
    proc = subprocess.run(
        [str(python_exe), "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return {
            "python": str(python_exe),
            "exists": True,
            "cuda_available": False,
            "torch_import_ok": False,
            "probe_error": (proc.stderr or proc.stdout or "").strip(),
        }
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {
            "python": str(python_exe),
            "exists": True,
            "cuda_available": False,
            "torch_import_ok": False,
            "probe_error": proc.stdout.strip(),
        }


@lru_cache(maxsize=1)
def _preferred_python_runtime() -> Dict[str, Any]:
    current_runtime = _probe_python_runtime(Path(sys.executable))
    gpu_runtime = _probe_python_runtime(ROOT / "venv_gpu" / "Scripts" / "python.exe")
    if gpu_runtime.get("cuda_available"):
        gpu_runtime["source"] = "venv_gpu"
        return gpu_runtime
    current_runtime["source"] = "current"
    return current_runtime


def _preferred_python_executable() -> str:
    return str(_preferred_python_runtime().get("python", sys.executable))


def _gpu_snapshot() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = dict(_preferred_python_runtime())
    snapshot.setdefault("cuda_available", False)

    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            first = proc.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if len(parts) >= 5:
                snapshot["device_name"] = parts[0]
                snapshot["utilization_pct"] = parts[1]
                snapshot["memory_used_mb"] = parts[2]
                snapshot["memory_total_mb"] = parts[3]
                snapshot["temperature_c"] = parts[4]
    except Exception:
        pass
    return snapshot


def _render_gpu_status(slot) -> None:
    gpu = _gpu_snapshot()
    with slot.container():
        st.subheader("GPU Runtime")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CUDA", "Available" if gpu.get("cuda_available") else "Unavailable")
        c2.metric("GPU Util", f"{gpu.get('utilization_pct', 'N/A')}%")
        mem_used = gpu.get("memory_used_mb", gpu.get("torch_memory_used_mb", "N/A"))
        mem_total = gpu.get("memory_total_mb", gpu.get("torch_memory_total_mb", "N/A"))
        c3.metric("Memory", f"{mem_used} / {mem_total} MB")
        c4.metric("Temp", f"{gpu.get('temperature_c', 'N/A')} C")
        st.caption(gpu.get("device_name", "GPU not detected"))
        st.caption(f"Execution Python: {gpu.get('python', sys.executable)}")


def _model_display_label(model_name: str) -> str:
    if model_name == "MAPPO-STGNN":
        return "MAPPO-STGNN"
    return model_name


def _render_local_gif(path: Path) -> None:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    st.markdown(
        f'<img src="data:image/gif;base64,{data}" style="width:100%; border-radius:8px; border:1px solid #e2e8f0;" />',
        unsafe_allow_html=True,
    )


def _render_model_showcase(raw: Dict[str, Any], featured_models: List[str] = FEATURED_MODELS) -> None:
    media = raw.get("dashboard_media", {}) if isinstance(raw, dict) else {}
    metrics_blocks = {k: v for k, v in raw.items() if isinstance(v, dict)} if isinstance(raw, dict) else {}
    st.subheader("Evaluation Playback")
    cols = st.columns(len(featured_models))
    for col, model_name in zip(cols, featured_models):
        metric_block = metrics_blocks.get(model_name, {})
        media_block = media.get(model_name, {}) if isinstance(media, dict) else {}
        with col:
            st.markdown(f"**{_model_display_label(model_name)}**")
            gif_path = media_block.get("gif_path")
            poster_path = media_block.get("poster_path")
            gif_file = _resolve_user_path(gif_path) if gif_path else None
            poster_file = _resolve_user_path(poster_path) if poster_path else None
            if gif_file and gif_file.exists():
                _render_local_gif(gif_file)
            elif poster_file and poster_file.exists():
                st.image(str(poster_file.resolve()))
                st.caption("GIF pending")
            else:
                st.info("Run pending")

            if metric_block and "mean_reward" in metric_block:
                st.metric("Reward", f"{metric_block.get('mean_reward', 0.0):.2f}")
                st.metric("Waiting", f"{metric_block.get('mean_waiting_time', 0.0):.2f} s")
                st.metric("Queue", f"{metric_block.get('mean_queue_length', 0.0):.2f}")


def _init_live_model_panels(featured_models: List[str] = FEATURED_MODELS) -> Dict[str, Dict[str, Any]]:
    st.subheader("Live Evaluation Panels")
    cols = st.columns(len(featured_models))
    panels: Dict[str, Dict[str, Any]] = {}
    for col, model_name in zip(cols, featured_models):
        with col:
            st.markdown(f"**{_model_display_label(model_name)}**")
            status_slot = st.empty()
            image_slot = st.empty()
            meta_slot = st.empty()
            status_slot.info("Queued")
            image_slot.info("Waiting for rollout")
            meta_slot.caption("No artifact yet")
            panels[model_name] = {
                "status": status_slot,
                "image": image_slot,
                "meta": meta_slot,
            }
    return panels


def _update_live_model_panel(
    panels: Dict[str, Dict[str, Any]],
    model_name: str,
    *,
    status: str,
    gif_path: str = "",
) -> None:
    panel = panels.get(model_name)
    if not panel:
        return
    if status == "running":
        panel["status"].warning("Running")
        panel["meta"].caption("Evaluation in progress")
    elif status == "done":
        panel["status"].success("Complete")
        if gif_path and Path(gif_path).exists():
            panel["image"].image(str(Path(gif_path).resolve()))
            panel["meta"].caption(Path(gif_path).name)
        else:
            panel["image"].info("GIF artifact not found")
    else:
        panel["status"].info(status)


def _parse_benchmark_progress_line(
    line: str,
    panels: Dict[str, Dict[str, Any]],
) -> None:
    if line.startswith("[MODEL_START] "):
        model_name = line.split("]", 1)[1].strip()
        _update_live_model_panel(panels, model_name, status="running")
    elif line.startswith("[VISUAL_READY] "):
        payload = line.split("]", 1)[1].strip()
        try:
            model_name, gif_path = payload.split(" ", 1)
        except ValueError:
            return
        _update_live_model_panel(panels, model_name, status="done", gif_path=gif_path.strip())


def _run_command_stream(
    cmd: List[str],
    log_slot,
    progress_bar,
    start: float,
    end: float,
    title: str,
    line_callback=None,
    gpu_slot=None,
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
        if line_callback is not None:
            try:
                line_callback(line.rstrip())
            except Exception:
                pass
        if gpu_slot is not None:
            try:
                _render_gpu_status(gpu_slot)
            except Exception:
                pass
        progress_bar.progress(min(end, start + (end - start) * 0.85))
    return_code = proc.wait()
    progress_bar.progress(end)
    output = "\n".join(lines).strip()
    if return_code != 0:
        raise RuntimeError(output or f"{title} failed.")
    return output


def _resolve_checkpoint() -> Path:
    existing = _checkpoint_candidates()
    if not existing:
        return Path()

    for c in existing:
        metadata_path = c.with_suffix("").with_suffix(".metadata.json")
        if metadata_path.exists():
            return c

    for c in existing:
        if c.name == "marl_ppo_traffic.zip":
            return c

    for c in existing:
        return c
    return Path()


def _checkpoint_candidates() -> List[Path]:
    candidates = [
        ROOT / BEST_CHECKPOINT,
        ROOT / "best_model_stage_2.zip",
        ROOT / "marl_ppo_traffic.zip",
        ROOT / "outputs/phase1/dqn_traffic_final.zip",
    ]
    existing: List[Path] = []
    seen = set()
    for candidate in candidates:
        if candidate.exists() and candidate not in seen:
            existing.append(candidate)
            seen.add(candidate)
    return existing


def _run_benchmark(config: Path, checkpoint: str, episodes: int) -> str:
    return _run_command(
        [
            _preferred_python_executable(),
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
        _preferred_python_executable(),
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
            _preferred_python_executable(),
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
        from src.phase1.marl_traffic_env import MARLTrafficEnv
        from src.utils.model_metadata import (
            is_digest_only_mismatch,
            load_metadata_for_checkpoint,
            validate_metadata,
        )
    except Exception as exc:
        return False, (
            f"Precheck import failed: {exc}. "
            "Ensure dashboard is launched from project environment and dependencies are installed."
        )

    env = None
    try:
        cfg = _load_yaml(config)
        env = MARLTrafficEnv(cfg)
        env_obs = str(getattr(env, "observation_space", None))
        env_act = str(getattr(env, "action_space", None))
        metadata = load_metadata_for_checkpoint(checkpoint)
        if not metadata:
            metadata_note = "Checkpoint metadata file not found; validated by loading checkpoint binary."
        else:
            mismatch = validate_metadata(
                metadata=metadata,
                expected_algorithm="PPO",
                observation_space_repr=env_obs,
                action_space_repr=env_act,
                config=cfg,
            )
            if mismatch:
                if is_digest_only_mismatch(mismatch, metadata, "PPO", env_obs, env_act):
                    metadata_note = (
                        "Compatible with warning: checkpoint config digest differs, "
                        "but metadata spaces match the selected evaluation setup."
                    )
                else:
                    return False, mismatch
            else:
                metadata_note = f"Metadata compatible. obs={env_obs}, action={env_act}"

        load_ok, load_msg = _precheck_checkpoint_binary_load(config, checkpoint)
        if not load_ok:
            return False, load_msg

        return True, f"{metadata_note}; checkpoint load passed."
    except Exception as exc:
        msg = str(exc)
        if "unexpected keyword argument 'use_sde'" in msg or "DQNPolicy.__init__" in msg:
            return (
                False,
                "Checkpoint algorithm mismatch: selected checkpoint is not a PPO checkpoint "
                "(possible DQN/SB3 format mismatch). Please provide a PPO checkpoint matching this configuration.",
            )
        return False, f"Precheck failed: {exc}"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _precheck_checkpoint_binary_load(config: Path, checkpoint: Path) -> Tuple[bool, str]:
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "import yaml\n"
        "from stable_baselines3 import PPO\n"
        "from src.phase1.marl_traffic_env import MARLTrafficEnv\n"
        "config_path=Path(sys.argv[1])\n"
        "checkpoint_path=Path(sys.argv[2])\n"
        "with config_path.open('r', encoding='utf-8') as f:\n"
        " cfg=yaml.safe_load(f)\n"
        "env=MARLTrafficEnv(cfg)\n"
        "try:\n"
        " PPO.load(str(checkpoint_path), env=env)\n"
        " print('OK')\n"
        "finally:\n"
        " env.close()\n"
    )
    proc = subprocess.run(
        [_preferred_python_executable(), "-c", code, str(config), str(checkpoint)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode == 0:
        return True, "Checkpoint binary load passed."
    output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    if "Observation spaces do not match" in output or "Action spaces do not match" in output:
        return (
            False,
            "Checkpoint binary does not match the selected evaluation environment. "
            f"{output}. Metadata may be stale for `{_safe_rel(checkpoint)}`.",
        )
    return False, f"Checkpoint load failed during compatibility precheck: {output}"


def _resolve_compatible_checkpoint(config: Path) -> Tuple[Path, str]:
    failures: List[str] = []
    for candidate in _checkpoint_candidates():
        ok, msg = _precheck_checkpoint_compatibility(config, candidate)
        if ok:
            return candidate, msg
        failures.append(f"- `{_safe_rel(candidate)}`: {msg}")
    return Path(), "\n".join(failures) if failures else "No checkpoint files were found."


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


def _is_valid_model_row(row: pd.Series) -> bool:
    core = ["mean_throughput", "mean_travel_time", "mean_waiting_time", "mean_queue_length"]
    vals = [float(row.get(c, 0.0) or 0.0) for c in core]
    return any(v != 0.0 for v in vals)


def _latency_df(raw: Dict[str, Any]) -> pd.DataFrame:
    payload = raw.get("latency_ms_per_step", [])
    if isinstance(payload, list) and payload:
        return pd.DataFrame(payload)
    return pd.DataFrame()


def _chart_bar(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    color: str,
    title: str,
    y_title: str,
    higher_is_better: bool = True,
    height: int = 320,
) -> alt.Chart:
    order = "descending" if higher_is_better else "ascending"
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X(f"{x}:N", sort=alt.EncodingSortField(field=y, order=order), title=None, axis=alt.Axis(labelAngle=-20)),
            y=alt.Y(f"{y}:Q", title=y_title),
            color=alt.Color(f"{color}:N", scale=alt.Scale(range=CHART_COLORS), legend=None),
            tooltip=[
                alt.Tooltip(f"{x}:N", title=x.replace("_", " ").title()),
                alt.Tooltip(f"{y}:Q", title=y_title, format=",.2f"),
            ],
        )
        .properties(title=title, height=height)
    )


def _action_diagnostics_df(raw: Dict[str, Any]) -> pd.DataFrame:
    payload = raw.get("action_diagnostics", {})
    if not isinstance(payload, dict) or not payload:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for model_name, diag in payload.items():
        if not isinstance(diag, dict):
            continue
        rows.append(
            {
                "model": model_name,
                "trace_steps": diag.get("trace_steps"),
                "unique_action_vectors": diag.get("unique_action_vectors"),
                "dominant_vector_fraction": diag.get("dominant_vector_fraction"),
                "vector_change_rate": diag.get("vector_change_rate"),
                "mean_unique_phases_per_step": diag.get("mean_unique_phases_per_step"),
                "weights_loaded": diag.get("weights_loaded"),
            }
        )
    return pd.DataFrame(rows)


def _score_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    scored = df.copy()
    total = pd.Series(0.0, index=scored.index)
    informative_metrics: List[str] = []
    for metric, meta in METRICS_META.items():
        values = scored[metric].astype(float)
        span = values.max() - values.min()
        if span <= 1e-12:
            continue
        informative_metrics.append(metric)
        if span == 0:
            norm = pd.Series(0.5, index=scored.index)
        else:
            if meta["higher_is_better"]:
                norm = (values - values.min()) / span
            else:
                norm = (values.max() - values) / span
        scored[f"{metric}_norm"] = norm
        total += norm
    if informative_metrics:
        scored["overall_score"] = total / len(informative_metrics)
    else:
        scored["overall_score"] = 0.0
    return scored.sort_values("overall_score", ascending=False), informative_metrics


def _render_overview(df: pd.DataFrame, lat_df: pd.DataFrame) -> None:
    valid_df = df[df.apply(_is_valid_model_row, axis=1)].copy()
    if valid_df.empty:
        st.warning("No valid model rows available for executive ranking. Run quality may be incomplete.")
        return
    scored, informative_metrics = _score_models(valid_df)
    leader = scored.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Overall Model", leader["model"])
    c2.metric("Top Overall Score", f"{leader['overall_score']:.3f}")
    c3.metric("Models Compared", int(valid_df.shape[0]))
    if len(informative_metrics) < len(METRICS_META):
        excluded = [METRICS_META[m]["label"] for m in METRICS_META if m not in informative_metrics]
        st.info("Overall ranking excludes non-informative metrics: " + ", ".join(excluded))
    k1, k2, k3, k4, k5 = st.columns(5)
    for i, metric in enumerate(METRICS_META.keys()):
        best_row = valid_df.sort_values(metric, ascending=not METRICS_META[metric]["higher_is_better"]).iloc[0]
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
    rank_chart_df = scored[["model", "overall_score"]].copy()
    rank_chart = (
        alt.Chart(rank_chart_df)
        .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
        .encode(
            y=alt.Y("model:N", sort=alt.EncodingSortField(field="overall_score", order="descending"), title=None),
            x=alt.X("overall_score:Q", title="Normalized score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("model:N", scale=alt.Scale(range=CHART_COLORS), legend=None),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("overall_score:Q", title="Score", format=".3f"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(rank_chart, use_container_width=True)
    st.dataframe(
        scored[["model", "overall_score"] + list(METRICS_META.keys())],
        use_container_width=True,
    )

    if not lat_df.empty and "mean_ms" in lat_df:
        st.subheader("Inference Latency")
        lat_chart_df = lat_df[["model", "mean_ms"]].dropna()
        st.altair_chart(
            _chart_bar(
                lat_chart_df,
                x="model",
                y="mean_ms",
                color="model",
                title="Mean Inference Time per Control Step",
                y_title="Milliseconds",
                higher_is_better=False,
                height=260,
            ),
            use_container_width=True,
        )
        st.dataframe(lat_df, use_container_width=True)


def _render_metrics(df: pd.DataFrame) -> None:
    st.subheader("Model Comparison Table")
    st.dataframe(df.sort_values("mean_reward", ascending=False), use_container_width=True)

    st.subheader("Metric Charts")
    chart_cols = st.columns(2)
    for idx, (metric, meta) in enumerate(METRICS_META.items()):
        chart_df = df[["model", metric]].dropna()
        chart_df = chart_df.sort_values(metric, ascending=not meta["higher_is_better"])
        with chart_cols[idx % 2]:
            st.altair_chart(
                _chart_bar(
                    chart_df,
                    x="model",
                    y=metric,
                    color="model",
                    title=f"{meta['label']} ({meta['unit']})",
                    y_title=meta["unit"],
                    higher_is_better=bool(meta["higher_is_better"]),
                    height=270,
                ),
                use_container_width=True,
            )

    st.subheader("Cross-Metric View")
    selected_models = st.multiselect("Models to plot", options=df["model"].tolist(), default=df["model"].tolist())
    selected_metrics = st.multiselect(
        "Metrics to plot",
        options=list(METRICS_META.keys()),
        default=list(METRICS_META.keys()),
    )
    if selected_models and selected_metrics:
        plot_df = df[df["model"].isin(selected_models)][["model"] + selected_metrics].copy()
        normalized_rows: List[Dict[str, Any]] = []
        for metric in selected_metrics:
            values = plot_df[metric].astype(float)
            span = values.max() - values.min()
            if span <= 1e-12:
                norm = pd.Series(0.5, index=plot_df.index)
            elif METRICS_META[metric]["higher_is_better"]:
                norm = (values - values.min()) / span
            else:
                norm = (values.max() - values) / span
            for row_idx, score in norm.items():
                normalized_rows.append(
                    {
                        "model": plot_df.loc[row_idx, "model"],
                        "metric": METRICS_META[metric]["label"],
                        "normalized_score": float(score),
                    }
                )
        norm_df = pd.DataFrame(normalized_rows)
        line = (
            alt.Chart(norm_df)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X("metric:N", title=None),
                y=alt.Y("normalized_score:Q", title="Normalized score", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("model:N", scale=alt.Scale(range=CHART_COLORS)),
                tooltip=[
                    alt.Tooltip("model:N", title="Model"),
                    alt.Tooltip("metric:N", title="Metric"),
                    alt.Tooltip("normalized_score:Q", title="Score", format=".3f"),
                ],
            )
            .properties(height=330)
        )
        st.altair_chart(line, use_container_width=True)


def _render_action_diagnostics(raw: Dict[str, Any]) -> None:
    diag_df = _action_diagnostics_df(raw)
    if diag_df.empty:
        st.info("Action diagnostics not available yet.")
        return

    st.subheader("Action Diagnostics")
    st.dataframe(diag_df, use_container_width=True)

    warnings: List[str] = []
    missing_weights = diag_df[diag_df["weights_loaded"] == False]["model"].tolist()
    if missing_weights:
        warnings.append("These learned baselines are running without trained weights: " + ", ".join(missing_weights))
    highly_static = diag_df[
        diag_df["dominant_vector_fraction"].fillna(0) >= 0.95
    ]["model"].tolist()
    if highly_static:
        warnings.append("These controllers are nearly constant over the sampled trace: " + ", ".join(highly_static))

    similarity = raw.get("action_similarity", {})
    if isinstance(similarity, dict):
        for ref_name, mapping in similarity.items():
            if not isinstance(mapping, dict):
                continue
            same_as_ref = [
                model_name
                for model_name, score in mapping.items()
                if model_name != ref_name and score is not None and score >= 0.95
            ]
            if same_as_ref:
                warnings.append(
                    f"Action trace is almost identical to {ref_name} for: " + ", ".join(same_as_ref)
                )

    if warnings:
        st.warning("Controller behavior warnings:\n- " + "\n- ".join(warnings))

    similarity = raw.get("action_similarity", {})
    if isinstance(similarity, dict) and similarity:
        st.subheader("Action Similarity")
        sim_rows: List[Dict[str, Any]] = []
        for ref_name, mapping in similarity.items():
            if not isinstance(mapping, dict):
                continue
            for model_name, score in mapping.items():
                sim_rows.append({"reference": ref_name, "model": model_name, "same_step_fraction": score})
        if sim_rows:
            st.dataframe(pd.DataFrame(sim_rows), use_container_width=True)


def _chart_domain(values: pd.Series) -> Optional[Tuple[float, float]]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None

    low = float(numeric.min())
    high = float(numeric.max())
    if low == high:
        pad = max(abs(low) * 0.05, 1.0)
    else:
        pad = (high - low) * 0.12
    return low - pad, high + pad


@_streamlit_fragment
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

    choice = st.selectbox(
        "Episode metric",
        options=list(metric_map.keys()),
        format_func=lambda x: metric_map[x],
        key="episode_analysis_metric",
    )
    episode_rows: List[Dict[str, Any]] = []
    for model_key in available:
        values = raw.get(model_key, {}).get(choice, [])
        for idx, value in enumerate(values):
            episode_rows.append({"episode": idx + 1, "model": labels.get(model_key, model_key), "value": value})

    if not episode_rows:
        st.info("No episode-level values available for selected metric.")
        return

    episode_df = pd.DataFrame(episode_rows)
    y_domain = _chart_domain(episode_df["value"])
    y_scale = alt.Scale(domain=list(y_domain), zero=False) if y_domain else alt.Scale(zero=False)
    episode_chart = (
        alt.Chart(episode_df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("episode:O", title="Episode"),
            y=alt.Y(
                "value:Q",
                title=metric_map[choice],
                scale=y_scale,
            ),
            color=alt.Color("model:N", scale=alt.Scale(range=CHART_COLORS)),
            tooltip=[
                alt.Tooltip("episode:O", title="Episode"),
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("value:Q", title=metric_map[choice], format=",.2f"),
            ],
        )
        .properties(height=330)
    )
    st.altair_chart(episode_chart, use_container_width=True)
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
    stress_long = df.melt(id_vars="model", var_name="metric", value_name="percent")
    stress_long["metric"] = stress_long["metric"].map(
        {
            "throughput_drop_pct": "Throughput Drop",
            "waiting_time_increase_pct": "Waiting Increase",
            "queue_length_increase_pct": "Queue Increase",
        }
    )
    stress_chart = (
        alt.Chart(stress_long)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("metric:N", title=None),
            y=alt.Y("percent:Q", title="Percent change under stress"),
            xOffset="model:N",
            color=alt.Color("model:N", scale=alt.Scale(range=CHART_COLORS)),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("percent:Q", title="Percent", format=".2f"),
            ],
        )
        .properties(height=330)
    )
    st.altair_chart(stress_chart, use_container_width=True)
    st.dataframe(df, use_container_width=True)


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
    def _table_text(table_df: pd.DataFrame) -> str:
        try:
            return table_df.to_markdown(index=False)
        except Exception:
            return table_df.to_string(index=False)

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
        _table_text(df),
        "",
    ]
    if not lat_df.empty:
        lines.extend(["## Latency Metrics", "", _table_text(lat_df), ""])
    return "\n".join(lines)


def _set_last_run(mode: str, command: str, status: str) -> None:
    st.session_state["last_run_mode"] = mode
    st.session_state["last_run_command"] = command
    st.session_state["last_run_status"] = status
    st.session_state["last_run_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _render_run_banner(current_mode: str) -> None:
    c1, c2, c3, c4 = st.columns([1, 1, 2, 2])
    c1.metric("Mode", current_mode)
    c2.metric("Last Status", st.session_state.get("last_run_status", "Not run"))
    c3.metric("Last Run", st.session_state.get("last_run_time", "N/A"))
    c4.metric("Last Command", st.session_state.get("last_run_command", "N/A"))


def main() -> None:
    st.set_page_config(page_title="Adaptive Traffic Control Evaluation Suite", layout="wide")
    st.title("Adaptive Traffic Control Evaluation Suite")
    st.caption(
        "Evaluate MAPPO-STGNN against CoLight and NSTLight under normal and stress conditions."
    )

    # Shared state defaults for both dashboard modes.
    run_now = False
    precheck_now = False
    load_latest = False
    run_stress = True
    run_logs = ""
    detail_logs = ""
    stress_logs = ""
    dev_logs = ""
    partial_run = False
    active_config_path = DEFAULT_CONFIG
    active_checkpoint_path = Path()
    run_cpu_quick = False
    run_gpu_standard = False
    run_gpu_extreme = False
    run_manual = False

    with st.sidebar:
        st.header("Experiment Setup")
        dashboard_mode = st.selectbox("Dashboard mode", options=["Observation", "Development"], index=0)

        if dashboard_mode == "Observation":
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
            gui = st.checkbox("External SUMO GUI popup", value=True)
            st.caption("Algorithm is fixed to PPO (best model). Baseline comparison includes CoLight and NSTLight. Dashboard playback is rendered in-page from evaluation rollouts.")

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

            run_now = st.button("Run Observation Evaluation Suite", use_container_width=True)
            precheck_now = st.button("Checkpoint Compatibility Precheck", use_container_width=True)
            load_latest = st.button("Load Latest Results", use_container_width=True)
        else:
            st.subheader("Development Flow")
            st.caption("Run wrapped publication profiles or execute a fully manual strict run.")
            run_cpu_quick = st.button("Run wrapped profile: cpu_quick", use_container_width=True)
            run_gpu_standard = st.button("Run wrapped profile: gpu_standard", use_container_width=True)
            run_gpu_extreme = st.button("Run wrapped profile: gpu_extreme", use_container_width=True)

            st.markdown("### Manual Strict Run")
            manual_config = st.text_input("Config path", value="configs/phase1.yaml")
            manual_checkpoint = st.text_input("Checkpoint path", value="outputs/phase1/dqn_traffic_final.zip")
            manual_mode = st.selectbox("Mode", options=["quick", "full"], index=1)
            manual_bench = st.number_input("Benchmark episodes", min_value=1, max_value=500, value=3, step=1)
            manual_detail = st.number_input("Detailed episodes", min_value=1, max_value=500, value=50, step=1)
            manual_stress = st.number_input("Stress episodes", min_value=1, max_value=500, value=3, step=1)
            manual_latency = st.selectbox("Latency device", options=["gpu", "cpu"], index=0)
            run_manual = st.button("Run manual strict flow", use_container_width=True)
            load_latest = st.button("Load Latest Results", use_container_width=True)

    execution_requested = (
        run_now
        or precheck_now
        or run_cpu_quick
        or run_gpu_standard
        or run_gpu_extreme
        or run_manual
    )
    if execution_requested:
        _render_run_banner(dashboard_mode)
        gpu_slot = st.empty()
        _render_gpu_status(gpu_slot)
        live_panels = _init_live_model_panels() if dashboard_mode == "Observation" and run_now else {}
    else:
        gpu_slot = st.empty()
        live_panels = {}

    if precheck_now:
        checkpoint_path = _resolve_checkpoint()
        if not checkpoint_path:
            st.error("No checkpoint found for compatibility precheck.")
            return
        cfg_path = DEFAULT_CONFIG
        base_cfg = _align_config_with_checkpoint_metadata(_load_yaml(cfg_path), checkpoint_path)
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
        compatible_checkpoint, msg = _resolve_compatible_checkpoint(temp_cfg)
        ok = bool(compatible_checkpoint)
        if ok:
            st.success(
                "Compatibility precheck passed. "
                f"Using `{_safe_rel(compatible_checkpoint)}`. {msg}"
            )
            _set_last_run(
                dashboard_mode,
                "Checkpoint Compatibility Precheck",
                "Success",
            )
        else:
            st.error(f"Compatibility precheck failed for all known checkpoints:\n{msg}")
            _set_last_run(
                dashboard_mode,
                "Checkpoint Compatibility Precheck",
                "Failed",
            )
        return

    if run_now:
        try:
            _set_last_run(
                dashboard_mode,
                "Observation Evaluation Suite",
                "Running",
            )
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
            base_cfg = _align_config_with_checkpoint_metadata(_load_yaml(cfg_path), checkpoint_path)
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
            compatible_checkpoint, compatible_msg = _resolve_compatible_checkpoint(temp_cfg)
            if not compatible_checkpoint:
                progress.progress(1.0)
                st.error(
                    "No compatible PPO checkpoint was found for the selected dashboard configuration.\n\n"
                    f"{compatible_msg}\n\n"
                    "This usually means the checkpoint was trained with a different SUMO scenario "
                    "or a stale `.metadata.json` is next to the checkpoint."
                )
                _set_last_run(
                    dashboard_mode,
                    "Observation Evaluation Suite",
                    "Failed",
                )
                return
            checkpoint_path = compatible_checkpoint
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
                _set_last_run(
                    dashboard_mode,
                    "Observation Evaluation Suite",
                    "Failed",
                )
                return
            progress.progress(0.05)
            st.success(f"Compatibility check passed. Using `{_safe_rel(checkpoint_path)}`. {pre_msg}")

            log_slot = st.empty()
            stage_slot.info("Stage 1/3: Running benchmark comparison...")
            run_logs = _run_command_stream(
                [
                    _preferred_python_executable(),
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
                line_callback=lambda line: _parse_benchmark_progress_line(line, live_panels),
                gpu_slot=gpu_slot,
            )
            st.success(f"Benchmark completed with config `{_safe_rel(temp_cfg)}`.")

            stage_slot.info("Stage 2/3: Running detailed episode evaluation...")
            detail_logs = _run_command_stream(
                [
                    _preferred_python_executable(),
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
                gpu_slot=gpu_slot,
            )
            st.success("Detailed episode evaluation completed.")

            if run_stress:
                stage_slot.info("Stage 3/3: Running adversarial stress test...")
                stress_logs = _run_command_stream(
                    [
                        _preferred_python_executable(),
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
                    gpu_slot=gpu_slot,
                )
                st.success("Adversarial stress test completed.")
            else:
                progress.progress(1.0)
                stage_slot.info("Stage 3/3: Stress test skipped by user. Run complete.")
            if run_stress:
                stage_slot.success("All stages completed successfully.")
            _set_last_run(
                dashboard_mode,
                "Observation Evaluation Suite",
                "Success",
            )
        except Exception as exc:
            try:
                progress.progress(1.0)
            except Exception:
                pass
            st.error(f"Execution failed: {exc}")
            _set_last_run(
                dashboard_mode,
                "Observation Evaluation Suite",
                "Failed",
            )
            return

    if dashboard_mode == "Development":
        try:
            run_profile = None
            if run_cpu_quick:
                run_profile = "cpu_quick"
            elif run_gpu_standard:
                run_profile = "gpu_standard"
            elif run_gpu_extreme:
                run_profile = "gpu_extreme"

            if run_profile is not None:
                _set_last_run(
                    dashboard_mode,
                    f"python scripts/run_profile.py --profile {run_profile}",
                    "Running",
                )
                progress = st.progress(0.01)
                stage_slot = st.empty()
                log_slot = st.empty()
                stage_slot.info(f"Running wrapped profile `{run_profile}`...")
                dev_logs = _run_command_stream(
                    [
                        _preferred_python_executable(),
                        "scripts/run_profile.py",
                        "--profile",
                        run_profile,
                    ],
                    log_slot=log_slot,
                    progress_bar=progress,
                    start=0.05,
                    end=1.0,
                    title=f"Profile {run_profile}",
                    gpu_slot=gpu_slot,
                )
                stage_slot.success(f"Wrapped profile `{run_profile}` completed.")
                run_now = True
                active_config_path = DEFAULT_CONFIG
                active_checkpoint_path = _resolve_checkpoint()
                partial_run = "FAILED (allowed)" in dev_logs
                _set_last_run(
                    dashboard_mode,
                    f"python scripts/run_profile.py --profile {run_profile}",
                    "Success",
                )

            if run_manual:
                _set_last_run(
                    dashboard_mode,
                    (
                        "python scripts/run_publication_suite.py "
                        f"--mode {manual_mode} --config {manual_config} --checkpoint {manual_checkpoint} "
                        f"--benchmark-episodes {int(manual_bench)} --detailed-episodes {int(manual_detail)} "
                        f"--stress-episodes {int(manual_stress)} --latency-device {manual_latency}"
                    ),
                    "Running",
                )
                progress = st.progress(0.01)
                stage_slot = st.empty()
                log_slot = st.empty()
                stage_slot.info("Running manual strict publication flow...")
                dev_logs = _run_command_stream(
                    [
                        _preferred_python_executable(),
                        "scripts/run_publication_suite.py",
                        "--mode",
                        manual_mode,
                        "--config",
                        manual_config,
                        "--checkpoint",
                        manual_checkpoint,
                        "--benchmark-episodes",
                        str(int(manual_bench)),
                        "--detailed-episodes",
                        str(int(manual_detail)),
                        "--stress-episodes",
                        str(int(manual_stress)),
                        "--latency-device",
                        manual_latency,
                    ],
                    log_slot=log_slot,
                    progress_bar=progress,
                    start=0.05,
                    end=1.0,
                    title="Manual strict flow",
                    gpu_slot=gpu_slot,
                )
                stage_slot.success("Manual strict publication flow completed.")
                run_now = True
                active_config_path = _resolve_user_path(manual_config)
                active_checkpoint_path = _resolve_user_path(manual_checkpoint)
                partial_run = "FAILED (allowed)" in dev_logs
                _set_last_run(
                    dashboard_mode,
                    (
                        "python scripts/run_publication_suite.py "
                        f"--mode {manual_mode} --config {manual_config} --checkpoint {manual_checkpoint} "
                        f"--benchmark-episodes {int(manual_bench)} --detailed-episodes {int(manual_detail)} "
                        f"--stress-episodes {int(manual_stress)} --latency-device {manual_latency}"
                    ),
                    "Success",
                )
        except Exception as exc:
            st.error(f"Development flow execution failed: {exc}")
            _set_last_run(
                dashboard_mode,
                "Development flow",
                "Failed",
            )
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
    if dev_logs:
        with st.expander("Development Flow Logs", expanded=False):
            st.code(dev_logs)

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
    if "dashboard_media" not in raw and DEFAULT_MEDIA.exists():
        try:
            raw["dashboard_media"] = json.loads(DEFAULT_MEDIA.read_text(encoding="utf-8"))
        except Exception:
            pass
    artifact_meta = raw.get("artifact_metadata", {}) if isinstance(raw, dict) else {}
    #if artifact_meta.get("artifact_type") == "presentation_demo":
        #st.warning(
         #   "Presentation demo artifacts loaded. These values are synthetic/sample outputs for UI demonstration, "
          #  "not benchmark evidence. Use a real evaluation run for final reported metrics."
        #)

    df = _flatten_results(raw)
    if df.empty:
        st.warning("No model metrics found in benchmark results.")
        return

    lat_df = _latency_df(raw)
    _render_model_showcase(raw)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Executive Overview", "All Metrics", "Episode Trends", "Stress Test", "Export"])

    with tab1:
        if partial_run:
            st.warning("Partial run detected: one or more stages failed. Rankings exclude invalid all-zero rows.")
        _render_overview(df, lat_df)
        _render_data_sanity_warnings(df, DEFAULT_EVAL_SUMMARY)

    with tab2:
        _render_metrics(df)
        _render_action_diagnostics(raw)

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
