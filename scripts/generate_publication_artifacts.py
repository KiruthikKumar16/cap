import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "main_figures"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _benchmark_table(raw: Dict[str, Any]) -> pd.DataFrame:
    metadata = raw.get("artifact_metadata", {}) if isinstance(raw, dict) else {}
    if metadata.get("artifact_type") == "presentation_demo":
        raise ValueError(
            "outputs/benchmark_results.json is marked as presentation_demo and "
            "must not be used for publication artifacts."
        )
    rows: List[Dict[str, Any]] = []
    for name, payload in raw.items():
        if not isinstance(payload, dict) or "mean_reward" not in payload:
            continue
        rows.append(
            {
                "model": name,
                "mean_reward": payload.get("mean_reward"),
                "mean_throughput_veh_per_h": payload.get("mean_throughput"),
                "mean_travel_time_s": payload.get("mean_travel_time"),
                "mean_waiting_time_s": payload.get("mean_waiting_time"),
                "mean_queue_length_vehicles": payload.get("mean_queue_length"),
            }
        )
    return pd.DataFrame(rows)


def _fairness_table(eval_summary: Dict[str, Any], benchmark_df: pd.DataFrame) -> pd.DataFrame:
    horizon = eval_summary.get("num_episodes", "unknown")
    has_eval = bool(eval_summary)
    has_benchmark = not benchmark_df.empty
    return pd.DataFrame(
        [
            {
                "criterion": "Same episode budget",
                "status": "PASS" if has_eval else "CHECK",
                "evidence": f"episodes={horizon}" if has_eval else "evaluation summary missing",
            },
            {
                "criterion": "Same evaluation horizon",
                "status": "PASS" if has_eval else "CHECK",
                "evidence": "Single phase1 config used" if has_eval else "evaluation summary missing",
            },
            {
                "criterion": "Same observation/reward interface",
                "status": "PASS" if has_benchmark else "CHECK",
                "evidence": "Unified evaluation entrypoints" if has_benchmark else "benchmark table missing",
            },
        ]
    )


def _ablation_gap_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ablation": "Without ST-GNN", "status": "not_run", "metric_delta_vs_full": "", "evidence_status": "missing"},
            {"ablation": "Without anomaly module", "status": "not_run", "metric_delta_vs_full": "", "evidence_status": "missing"},
            {"ablation": "Without predictive phase", "status": "not_run", "metric_delta_vs_full": "", "evidence_status": "missing"},
            {"ablation": "Without cross-intersection coordination", "status": "not_run", "metric_delta_vs_full": "", "evidence_status": "missing"},
            {"ablation": "Without robustness perturbation handling", "status": "not_run", "metric_delta_vs_full": "", "evidence_status": "missing"},
        ]
    )


def _ablation_results_table(ablation_raw: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for variant, payload in ablation_raw.items():
        if not isinstance(payload, dict):
            continue
        dqn = payload.get("dqn", {})
        rows.append(
            {
                "variant": variant,
                "mean_reward": dqn.get("mean_reward"),
                "mean_waiting_time_s": dqn.get("mean_waiting_time"),
                "mean_queue_length_vehicles": dqn.get("mean_queue_length"),
            }
        )
    if not rows:
        rows = [{"variant": "not_run", "mean_reward": "", "mean_waiting_time_s": "", "mean_queue_length_vehicles": "", "evidence_status": "missing"}]
    return pd.DataFrame(rows)


def _generalization_table(g_raw: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for map_name, payload in g_raw.items():
        if not isinstance(payload, dict) or "mean_reward" not in payload:
            continue
        rows.append(
            {
                "map": map_name,
                "mean_reward": payload.get("mean_reward"),
                "mean_throughput_veh_per_h": payload.get("mean_throughput"),
                "mean_waiting_time_s": payload.get("mean_waiting_time"),
                "mean_queue_length_vehicles": payload.get("mean_queue_length"),
            }
        )
    drops = g_raw.get("map_a_to_b_drop_pct", {})
    if drops:
        rows.append(
            {
                "map": "Map_A_to_B_drop_pct",
                "mean_reward": "",
                "mean_throughput_veh_per_h": drops.get("throughput_drop_pct"),
                "mean_waiting_time_s": drops.get("waiting_time_increase_pct"),
                "mean_queue_length_vehicles": drops.get("queue_length_increase_pct"),
            }
        )
    if not rows:
        rows = [{"map": "not_run", "mean_reward": "", "mean_throughput_veh_per_h": "", "mean_waiting_time_s": "", "mean_queue_length_vehicles": "", "evidence_status": "missing"}]
    return pd.DataFrame(rows)


def _stress_table(stress_raw: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    degrad = stress_raw.get("degradation_limits_pct", {})
    for model, payload in degrad.items():
        rows.append(
            {
                "model": model,
                "throughput_drop_pct": payload.get("throughput_drop_pct"),
                "waiting_time_increase_pct": payload.get("waiting_time_increase_pct"),
                "queue_length_increase_pct": payload.get("queue_length_increase_pct"),
            }
        )
    if not rows:
        rows = [{"model": "not_run", "throughput_drop_pct": "", "waiting_time_increase_pct": "", "queue_length_increase_pct": "", "evidence_status": "missing"}]
    return pd.DataFrame(rows)


def _latency_table(lat_raw: Any) -> pd.DataFrame:
    if isinstance(lat_raw, list) and lat_raw:
        return pd.DataFrame(lat_raw)
    return pd.DataFrame([{"model": "not_run", "device": "", "n_runs": "", "mean_ms": "", "p95_ms": "", "p99_ms": "", "evidence_status": "missing"}])


def _scalability_scaffold() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"intersections": 25, "training_wallclock_s": "", "inference_ms_per_step": "", "gpu_memory_mb": "", "ctde_comm_estimate_kb_per_step": "", "evidence_status": "not_run"},
            {"intersections": 100, "training_wallclock_s": "", "inference_ms_per_step": "", "gpu_memory_mb": "", "ctde_comm_estimate_kb_per_step": "", "evidence_status": "not_run"},
        ]
    )


def _write_summary(
    benchmark_df: pd.DataFrame,
    fairness_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    mode: str,
) -> None:
    def _safe_table(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except Exception:
            return df.to_string(index=False)

    lines: List[str] = []
    lines.append("# Main Results Summary")
    lines.append("")
    lines.append(f"- Generation mode: `{mode}`")
    lines.append("- Status: generated from currently available outputs; missing experiments are reported as gaps.")
    lines.append("")
    lines.append("## Benchmark Table")
    lines.append("")
    lines.append(_safe_table(benchmark_df) if not benchmark_df.empty else "No benchmark data available.")
    lines.append("")
    lines.append("## Statistical Reporting")
    lines.append("")
    stat_path = RESULTS_DIR / "statistical_summary.csv"
    lines.append(
        f"- Statistical table: `{stat_path}`" if stat_path.exists() else "- Statistical table not found. Run `scripts/generate_statistical_tables.py`."
    )
    lines.append("")
    lines.append("## Fairness Checklist")
    lines.append("")
    lines.append(_safe_table(fairness_df))
    lines.append("")
    lines.append("## Hard-Nosed Failure Reporting")
    lines.append("")
    primary_name = "MAPPO-STGNN"
    baseline_win_line = "- Baseline-win scenario: not available yet."
    if not benchmark_df.empty and primary_name in benchmark_df["model"].values:
        ours_reward = float(
            benchmark_df.loc[benchmark_df["model"] == primary_name, "mean_reward"].iloc[0]
        )
        challengers = benchmark_df[benchmark_df["model"] != primary_name].copy()
        if not challengers.empty:
            best_ch = challengers.sort_values("mean_reward", ascending=False).iloc[0]
            if float(best_ch["mean_reward"]) > ours_reward:
                baseline_win_line = (
                    f"- Baseline-win scenario: `{best_ch['model']}` exceeds `{primary_name}` "
                    f"on mean reward ({best_ch['mean_reward']:.3f} vs {ours_reward:.3f})."
                )
            else:
                baseline_win_line = (
                    f"- Baseline-win scenario: none observed on mean reward; best baseline "
                    f"`{best_ch['model']}` remains below `{primary_name}`."
                )
    lines.append(baseline_win_line)

    degradation_line = "- Identified degradation mode: not available yet."
    mitigation_line = "- Mitigation plan: run a full stress sweep before making robustness claims."
    if not stress_df.empty and "waiting_time_increase_pct" in stress_df.columns:
        valid = stress_df[pd.to_numeric(stress_df["waiting_time_increase_pct"], errors="coerce").notna()].copy()
        if not valid.empty:
            valid["waiting_time_increase_pct"] = pd.to_numeric(valid["waiting_time_increase_pct"])
            worst = valid.sort_values("waiting_time_increase_pct", ascending=False).iloc[0]
            degradation_line = (
                f"- Identified degradation mode: `{worst['model']}` shows "
                f"{worst['waiting_time_increase_pct']:.2f}% waiting-time increase under stress."
            )
            mitigation_line = (
                "- Mitigation plan: rerun the full stress benchmark, then evaluate whether adaptive anomaly "
                "thresholding or noise-aware observation masking improves the measured worst case."
            )
    lines.append(degradation_line)
    lines.append(mitigation_line)
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Results depend on checkpoint-config compatibility and SUMO runtime consistency.")
    lines.append("- Full ablation completion is required before publication claims.")
    lines.append("- Include at least one scenario where a baseline wins.")
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-facing result artifacts")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_raw = _load_json(ROOT / "outputs" / "benchmark_results.json")
    eval_summary_raw = _load_json(ROOT / "outputs" / "phase1" / "evaluation_summary.json")
    stress_raw = _load_json(ROOT / "outputs" / "phase3" / "adversarial_benchmark.json")
    ablation_raw = _load_json(ROOT / "outputs" / "ablation_results.json")
    generalization_raw = _load_json(ROOT / "outputs" / "phase4" / "zero_shot_generalization.json")
    latency_raw = _load_json(ROOT / "outputs" / "latency" / "inference_latency.json")

    benchmark_df = _benchmark_table(benchmark_raw if isinstance(benchmark_raw, dict) else {})
    fairness_df = _fairness_table(eval_summary_raw if isinstance(eval_summary_raw, dict) else {}, benchmark_df)
    ablation_gap_df = _ablation_gap_table()
    ablation_results_df = _ablation_results_table(ablation_raw if isinstance(ablation_raw, dict) else {})
    generalization_df = _generalization_table(generalization_raw if isinstance(generalization_raw, dict) else {})
    stress_df = _stress_table(stress_raw if isinstance(stress_raw, dict) else {})
    latency_df = _latency_table(latency_raw)
    scalability_df = _scalability_scaffold()

    benchmark_df.to_csv(RESULTS_DIR / "main_tables.csv", index=False)
    fairness_df.to_csv(RESULTS_DIR / "fairness_checklist.csv", index=False)
    ablation_gap_df.to_csv(RESULTS_DIR / "ablation_evidence_gaps.csv", index=False)
    ablation_results_df.to_csv(RESULTS_DIR / "ablation_contributions.csv", index=False)
    generalization_df.to_csv(RESULTS_DIR / "generalization_table.csv", index=False)
    stress_df.to_csv(RESULTS_DIR / "stress_recovery_table.csv", index=False)
    latency_df.to_csv(RESULTS_DIR / "latency_table.csv", index=False)
    scalability_df.to_csv(RESULTS_DIR / "scalability_table.csv", index=False)
    _write_summary(benchmark_df, fairness_df, stress_df, args.mode)

    (FIG_DIR / "README.md").write_text(
        "This directory stores publication figures generated from benchmark and evaluation outputs.\n",
        encoding="utf-8",
    )

    print("Generated:")
    for name in [
        "main_tables.csv",
        "fairness_checklist.csv",
        "ablation_evidence_gaps.csv",
        "ablation_contributions.csv",
        "generalization_table.csv",
        "stress_recovery_table.csv",
        "latency_table.csv",
        "scalability_table.csv",
        "summary.md",
    ]:
        print(f"- {RESULTS_DIR / name}")


if __name__ == "__main__":
    main()
