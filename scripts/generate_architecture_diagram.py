from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "main_figures"


PALETTE = {
    "navy": "#1F3A5F",
    "teal": "#2C7A7B",
    "green": "#2F855A",
    "amber": "#B7791F",
    "red": "#C53030",
    "slate": "#4A5568",
    "light": "#F7FAFC",
    "line": "#2D3748",
    "muted": "#718096",
}


def _rounded_box(ax, x, y, w, h, title, body, color):
    title_lines = title.count("\n") + 1
    title_font = 12 if len(title) < 24 and title_lines == 1 else 10.5
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        facecolor=color,
        edgecolor="white",
        linewidth=1.6,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=title_font,
        fontweight="bold",
        color="white",
    )
    ax.text(
        x + w / 2,
        y + h * 0.33,
        body,
        ha="center",
        va="center",
        fontsize=9.3,
        color="white",
        linespacing=1.3,
    )


def _arrow(ax, x1, y1, x2, y2, label=None, curve=0.0, linestyle="-", color=None, label_dx=0.0, label_dy=0.0):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=2.0,
            color=color or PALETTE["line"],
            linestyle=linestyle,
            connectionstyle=f"arc3,rad={curve}",
            shrinkA=4,
            shrinkB=4,
        ),
    )
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2 + (0.03 if curve >= 0 else -0.03)
        ax.text(mx + label_dx, my + label_dy, label, fontsize=8.6, color=PALETTE["line"], ha="center", va="center")


def _section_label(ax, x, y, text):
    ax.text(x, y, text, fontsize=10, fontweight="bold", color=PALETTE["muted"], ha="left", va="center")


def generate_framework_architecture():
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(PALETTE["light"])

    _section_label(ax, 0.04, 0.93, "Research Inputs")
    _section_label(ax, 0.29, 0.93, "Perception and Prediction")
    _section_label(ax, 0.54, 0.93, "Decision and Adaptation")
    _section_label(ax, 0.79, 0.93, "Evaluation and Publication")

    _rounded_box(
        ax, 0.04, 0.60, 0.18, 0.23,
        "Scenario and Config Layer",
        "YAML experiment settings\nSUMO net/route files\nseed protocol and runtime profiles",
        PALETTE["navy"],
    )
    _rounded_box(
        ax, 0.04, 0.28, 0.18, 0.23,
        "Traffic Environment Layer",
        "SUMO + TraCI execution\nfeature extraction\ntraffic graph construction\nmulti-intersection state history",
        PALETTE["slate"],
    )

    _rounded_box(
        ax, 0.29, 0.60, 0.18, 0.23,
        "ST-GNN Forecasting Layer",
        "SpatialTemporalAutoencoder\nGAT-based graph reasoning\nGRU temporal encoder\nmulti-step traffic forecasting",
        PALETTE["teal"],
    )
    _rounded_box(
        ax, 0.29, 0.28, 0.18, 0.23,
        "Graph Representation Layer",
        "TrafficGNNEncoder\nnode and global embeddings\nneighbor-aware observations\nforecast-conditioned features",
        PALETTE["green"],
    )

    _rounded_box(
        ax, 0.54, 0.60, 0.18, 0.23,
        "Control Learning Layer",
        "MAPPO / PPO policy\ncentralized training\nshared critic and local actors\nsignal phase action selection",
        PALETTE["red"],
    )
    _rounded_box(
        ax, 0.54, 0.28, 0.18, 0.23,
        "Robustness Layer",
        "Phase 2 anomaly detector\nPhase 3 anomaly-aware integration\nrisk-aware reward shaping\nadaptive thresholds and penalties",
        PALETTE["amber"],
    )

    _rounded_box(
        ax, 0.79, 0.60, 0.18, 0.23,
        "Benchmarking Layer",
        "fixed-time, random,\nactuated, CoLight,\nPressLight, NSTLight baselines\nstress and generalization tests",
        PALETTE["navy"],
    )
    _rounded_box(
        ax, 0.79, 0.28, 0.18, 0.23,
        "Artifact Layer",
        "tables, figures, summaries\nlatency and fairness reports\nStreamlit dashboard\nLaTeX-ready publication outputs",
        PALETTE["slate"],
    )

    _arrow(ax, 0.22, 0.72, 0.29, 0.72)
    _arrow(ax, 0.22, 0.39, 0.29, 0.39)
    _arrow(ax, 0.38, 0.60, 0.38, 0.51)
    _arrow(ax, 0.47, 0.72, 0.54, 0.72)
    _arrow(ax, 0.47, 0.39, 0.54, 0.39)
    _arrow(ax, 0.63, 0.60, 0.63, 0.51)
    _arrow(ax, 0.72, 0.72, 0.79, 0.72)
    _arrow(ax, 0.72, 0.39, 0.79, 0.39)
    _arrow(ax, 0.88, 0.60, 0.13, 0.51, "closed-loop control", curve=0.22, linestyle="--", label_dy=0.04)
    _arrow(ax, 0.13, 0.51, 0.88, 0.28, "runtime telemetry and artifacts", curve=-0.18, linestyle="--", label_dy=-0.03)

    ax.text(
        0.5,
        0.08,
        "Closed-loop operation: scenario configuration initializes SUMO, the ST-GNN produces forecast-aware embeddings, "
        "MAPPO selects decentralized signal actions, and the evaluation layer converts runtime outcomes into reproducible publication artifacts.",
        ha="center",
        va="center",
        fontsize=10,
        color=PALETTE["line"],
        wrap=True,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "Whole-Framework Architecture for Robust Multi-Agent Traffic Control",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    return fig


def generate_system_workflow():
    fig, ax = plt.subplots(figsize=(16, 8.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(PALETTE["light"])

    stages = [
        (0.05, "Data and Scenario\nPreparation", "network generation\nSUMO routes\ntraining traces\nconfig selection", PALETTE["navy"]),
        (0.24, "Phase 1 Controller\nTraining", "SUMOTrafficEnv\nPredictiveGNNRL\nMAPPO/PPO updates\nforecast-loss callback", PALETTE["teal"]),
        (0.43, "Phase 2 Anomaly\nModeling", "synthetic or SUMO sequences\nSpatialTemporalAutoencoder\nthreshold calibration\nanomaly scoring", PALETTE["amber"]),
        (0.62, "Phase 3 Robust\nIntegration", "anomaly-aware controller\nrisk model\nadaptive reward shaping\nstress scenario handling", PALETTE["red"]),
        (0.81, "Evaluation and\nDissemination", "benchmarks and ablations\ngeneralization and latency\nCSV/JSON tables\nfigures and dashboard", PALETTE["green"]),
    ]

    for x, title, body, color in stages:
        _rounded_box(ax, x, 0.47, 0.14, 0.28, title, body, color)

    _arrow(ax, 0.19, 0.61, 0.24, 0.61)
    _arrow(ax, 0.38, 0.61, 0.43, 0.61)
    _arrow(ax, 0.57, 0.61, 0.62, 0.61)
    _arrow(ax, 0.76, 0.61, 0.81, 0.61)

    _rounded_box(
        ax, 0.24, 0.12, 0.18, 0.18,
        "Baselines",
        "Fixed-time\nRandom\nActuated\nCoLight / PressLight / NSTLight",
        PALETTE["slate"],
    )
    _rounded_box(
        ax, 0.46, 0.12, 0.18, 0.18,
        "Validation Protocol",
        "seeded evaluation\nstress injection\ngeneralization maps\nstatistical summaries",
        PALETTE["slate"],
    )
    _rounded_box(
        ax, 0.68, 0.12, 0.18, 0.18,
        "Publication Outputs",
        "main tables\nfigure panels\nsummary markdown\nreport integration",
        PALETTE["slate"],
    )

    _arrow(ax, 0.33, 0.47, 0.33, 0.30, "compare against")
    _arrow(ax, 0.52, 0.47, 0.55, 0.30, "validated with")
    _arrow(ax, 0.71, 0.47, 0.77, 0.30, "rendered as")
    _arrow(ax, 0.41, 0.21, 0.46, 0.21, "feeds")
    _arrow(ax, 0.64, 0.21, 0.68, 0.21, "materializes")

    ax.text(
        0.5,
        0.87,
        "System view spanning offline preparation, controller learning, anomaly modeling, robust integration, and final publication artifact generation.",
        ha="center",
        va="center",
        fontsize=10.5,
        color=PALETTE["line"],
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "End-to-End System Workflow and Evaluation Pipeline",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    return fig


def _save(fig, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main():
    framework_paths = _save(generate_framework_architecture(), "framework_architecture_publication")
    workflow_paths = _save(generate_system_workflow(), "system_workflow_publication")
    print(f"Created {framework_paths[0]}")
    print(f"Created {framework_paths[1]}")
    print(f"Created {workflow_paths[0]}")
    print(f"Created {workflow_paths[1]}")


if __name__ == "__main__":
    main()
