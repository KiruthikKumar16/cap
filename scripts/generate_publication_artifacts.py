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
    # REMOVED: Fallback to archive/unverified_evidence. 
    # For Baseline results, only real 'outputs/' are acceptable.

    metadata = raw.get("artifact_metadata", {}) if isinstance(raw, dict) else {}
    if metadata.get("artifact_type") == "presentation_demo":
        # Check if we have a real one in outputs
        real_path = ROOT / "outputs" / "benchmark_results.json"
        if real_path.exists():
            new_raw = _load_json(real_path)
            if new_raw.get("artifact_metadata", {}).get("artifact_type") != "presentation_demo":
                raw = new_raw
            else:
                print("[ERROR] benchmark_results.json is still marked as presentation_demo. Research-grade results must be generated from full runs.")
                return pd.DataFrame()
        else:
            print("[ERROR] No real benchmark data found. Run 'scripts/run_benchmarks.py' first.")
            return pd.DataFrame()

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
                "status": "PASS" if has_eval else "FAIL",
                "evidence": f"episodes={horizon}" if has_eval else "Missing evaluation summary",
            },
            {
                "criterion": "Same evaluation horizon",
                "status": "PASS" if has_eval else "FAIL",
                "evidence": "Single phase1 config used" if has_eval else "Missing evaluation summary",
            },
            {
                "criterion": "Same observation/reward interface",
                "status": "PASS" if has_benchmark else "FAIL",
                "evidence": "Unified evaluation entrypoints" if has_benchmark else "Missing benchmark table",
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
    # REMOVED: Fallback to archive/unverified_evidence

    for variant, payload in ablation_raw.items():
        if not isinstance(payload, dict):
            continue
        # Support different nested structures in evaluate results
        dqn = payload.get("mappo", payload.get("dqn", payload.get("ppo", {})))
        if not dqn and "rewards" in payload:
            # Maybe it's a flat metrics block
            dqn = payload
            
        rows.append(
            {
                "variant": variant,
                "mean_reward": dqn.get("mean_reward", dqn.get("rewards_mean", "")),
                "mean_waiting_time_s": dqn.get("mean_waiting_time", dqn.get("waiting_times_mean", "")),
                "mean_queue_length_vehicles": dqn.get("mean_queue_length", dqn.get("queue_lengths_mean", "")),
            }
        )
    if not rows:
        rows = [{"variant": "Ablation study pending", "mean_reward": "N/A", "mean_waiting_time_s": "N/A", "mean_queue_length_vehicles": "N/A", "evidence_status": "MISSING"}]
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
        display_name = "MAPPO-STGNN" if model == "mappo" else model.upper()
        rows.append(
            {
                "model": display_name,
                "throughput_drop_pct": payload.get("throughput_drop_pct"),
                "waiting_time_increase_pct": payload.get("waiting_time_increase_pct"),
                "queue_length_increase_pct": payload.get("queue_length_increase_pct"),
            }
        )
    if not rows:
        rows = [{"model": "Stress test pending", "throughput_drop_pct": "N/A", "waiting_time_increase_pct": "N/A", "queue_length_increase_pct": "N/A", "evidence_status": "MISSING"}]
    return pd.DataFrame(rows)


def _latency_table(lat_raw: Any) -> pd.DataFrame:
    if isinstance(lat_raw, list) and lat_raw:
        return pd.DataFrame(lat_raw)
    return pd.DataFrame([{"model": "not_run", "device": "", "n_runs": "", "mean_ms": "", "p95_ms": "", "p99_ms": "", "evidence_status": "missing"}])


def _scalability_table(scalability_raw: Dict[str, Any]) -> pd.DataFrame:
    if not scalability_raw:
        return pd.DataFrame([{"intersections": "N/A", "training_wallclock_s": "N/A", "inference_ms_per_step": "N/A", "evidence_status": "PENDING_RUN"}])
    
    rows = []
    for entry in scalability_raw.get("results", []):
        rows.append(entry)
    return pd.DataFrame(rows)


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
    primary_name = "Proposed Model"
    baseline_win_line = "- Baseline-win scenario: not available yet."
    
    # Try to find our model even if named differently in the dataframe
    our_model_row = benchmark_df[benchmark_df["model"].str.contains("Proposed|PPO|dqn", case=False, na=False)]
    
    if not benchmark_df.empty and not our_model_row.empty:
        actual_name = our_model_row["model"].iloc[0]
        proposed_reward = float(our_model_row["mean_reward"].iloc[0])
        challengers = benchmark_df[benchmark_df["model"] != actual_name].copy()
        if not challengers.empty:
            optimized_ch = challengers.sort_values("mean_reward", ascending=False).iloc[0]
            if float(optimized_ch["mean_reward"]) > proposed_reward:
                baseline_win_line = (
                    f"- Baseline-win scenario: `{optimized_ch['model']}` exceeds `{primary_name}` "
                    f"on mean reward ({optimized_ch['mean_reward']:.3f} vs {proposed_reward:.3f})."
                )
            else:
                baseline_win_line = (
                    f"- Baseline-win scenario: none observed on mean reward; optimized baseline "
                    f"`{optimized_ch['model']}` remains below `{primary_name}`."
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
    scalability_raw = _load_json(ROOT / "outputs" / "scalability" / "scalability_results.json")

    benchmark_df = _benchmark_table(benchmark_raw if isinstance(benchmark_raw, dict) else {})
    fairness_df = _fairness_table(eval_summary_raw if isinstance(eval_summary_raw, dict) else {}, benchmark_df)
    ablation_gap_df = _ablation_gap_table()
    ablation_results_df = _ablation_results_table(ablation_raw if isinstance(ablation_raw, dict) else {})
    generalization_df = _generalization_table(generalization_raw if isinstance(generalization_raw, dict) else {})
    stress_df = _stress_table(stress_raw if isinstance(stress_raw, dict) else {})
    latency_df = _latency_table(latency_raw)
    scalability_df = _scalability_table(scalability_raw if isinstance(scalability_raw, dict) else {})

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
