import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


ROOT = Path(__file__).resolve().parent.parent
EVAL_SUMMARY = ROOT / "outputs" / "phase1" / "evaluation_summary.json"
OUT_TABLE = ROOT / "results" / "statistical_summary.csv"


def _ci95(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    s = pd.Series(values, dtype=float)
    return 1.96 * float(s.std(ddof=1)) / (len(values) ** 0.5)


def _mean_std_ci(values: List[float]) -> Dict[str, float]:
    s = pd.Series(values, dtype=float) if values else pd.Series([0.0], dtype=float)
    return {"mean": float(s.mean()), "std": float(s.std(ddof=0)), "ci95": _ci95(values)}


def main() -> None:
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    if not EVAL_SUMMARY.exists():
        pd.DataFrame(
            [{"note": "evaluation_summary.json not found; run evaluation first"}]
        ).to_csv(OUT_TABLE, index=False)
        print(f"Saved placeholder stats table to {OUT_TABLE}")
        return

    raw = json.loads(EVAL_SUMMARY.read_text(encoding="utf-8"))
    model_keys = [k for k in ["dqn", "fixed_time", "actuated", "random"] if k in raw]
    metric_keys = ["rewards", "throughputs", "travel_times", "waiting_times", "queue_lengths"]
    rows = []
    for mk in model_keys:
        for metric in metric_keys:
            vals = raw.get(mk, {}).get(metric, [])
            stats = _mean_std_ci(vals)
            rows.append(
                {
                    "model": mk,
                    "metric": metric,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "ci95": stats["ci95"],
                }
            )

    # Pairwise significance (dqn vs fixed_time where available).
    if scipy_stats and "dqn" in model_keys and "fixed_time" in model_keys:
        for metric in metric_keys:
            a = raw.get("dqn", {}).get(metric, [])
            b = raw.get("fixed_time", {}).get(metric, [])
            if len(a) > 1 and len(b) > 1:
                _, p_val = scipy_stats.ttest_ind(a, b, equal_var=False)
                rows.append(
                    {
                        "model": "dqn_vs_fixed_time",
                        "metric": metric,
                        "mean": "",
                        "std": "",
                        "ci95": "",
                        "p_value": float(p_val),
                    }
                )

    pd.DataFrame(rows).to_csv(OUT_TABLE, index=False)
    print(f"Saved statistical summary to {OUT_TABLE}")


if __name__ == "__main__":
    main()
