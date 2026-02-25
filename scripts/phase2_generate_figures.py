"""
Generate Phase 2 anomaly detection figures.

Reads outputs/phase2/anomaly_eval_summary.json and creates:
1) A single-method metrics bar chart (ours).
2) A SOTA comparison chart across methods (ours vs baselines) when available.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    summary_path = project_root / "outputs" / "phase2" / "anomaly_eval_summary.json"
    out_dir = project_root / "outputs" / "phase2" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase2_anomaly_metrics.png"
    sota_path = out_dir / "phase2_anomaly_sota_comparison.png"

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    metrics = summary.get("metrics", {})
    labels = ["Precision", "Recall", "F1", "ROC-AUC", "False Alarm Rate"]
    values = [
        float(metrics.get("precision", 0.0)),
        float(metrics.get("recall", 0.0)),
        float(metrics.get("f1", 0.0)),
        float(metrics.get("roc_auc", 0.0)),
        float(metrics.get("false_alarm_rate", 0.0)),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Phase 2 Anomaly Detection Metrics")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(val + 0.02, 1.02),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved: {out_path}")

    # SOTA comparison chart if multiple methods are available
    methods = summary.get("methods")
    if methods:
        method_keys = list(methods.keys())
        method_labels = [methods[k].get("label", k) for k in method_keys]
        metric_keys = ["precision", "recall", "f1", "roc_auc", "false_alarm_rate"]
        metric_names = ["Precision", "Recall", "F1", "ROC-AUC", "False Alarm Rate"]

        # Build matrix: [methods x metrics]
        data = []
        for k in method_keys:
            m = methods[k].get("metrics", {})
            data.append([float(m.get(mk, 0.0)) for mk in metric_keys])

        x = range(len(metric_keys))
        width = 0.8 / max(1, len(method_keys))
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, row in enumerate(data):
            ax.bar([v + i * width for v in x], row, width=width, label=method_labels[i])

        ax.set_xticks([v + width * (len(method_keys) - 1) / 2 for v in x])
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Phase 2 SOTA Comparison (Ours vs Baselines)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(sota_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[OK] Saved: {sota_path}")


if __name__ == "__main__":
    main()
