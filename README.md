# Robust Multi-Agent Traffic Control Under Non-Stationarity With Anomaly-Aware Proactive Adaptation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red.svg)
![SUMO](https://img.shields.io/badge/Simulator-SUMO-green.svg)

This repository presents a unified research pipeline for robust urban traffic signal control under non-stationary conditions.  
The central contribution is the integration of MAPPO-based multi-agent control, spatio-temporal representation learning, and anomaly-aware proactive adaptation in one reproducible evaluation framework.

## Core Contribution Statement

The project introduces a robust multi-agent traffic control framework that:
- learns coordinated signal policies with MAPPO under CTDE,
- incorporates spatio-temporal structure for network-level state representation,
- adapts to non-stationary disturbances (sensor failures, accidents, demand shifts),
- and evaluates performance using standardized, seed-based, publication-oriented protocols.

## One-Command Reproducibility

Run the complete pipeline using:

```bash
python scripts/reproduce_main_results.py --mode quick
```

or

```bash
python scripts/reproduce_main_results.py --mode full
```

The pipeline executes training/evaluation stages and produces standardized artifacts in `results/`.

For full publication scaffolding (ablation + stress + generalization + latency + stats + artifacts):

```bash
python scripts/run_publication_suite.py --mode full
```

## Reproducibility and Protocol

- Experiment protocol: `docs/EXPERIMENT_PROTOCOL.md`
- Reproducibility guide: `docs/REPRODUCIBILITY.md`
- Limitations and failure analysis: `docs/LIMITATIONS.md`
- Fixed seed list for publication tables: `configs/phase1.yaml` and `docs/APPENDIX_SEEDS.md`

## Results Artifacts (Auto-Generated)

- Main tables: `results/main_tables.csv`
- Statistical summary: `results/statistical_summary.csv`
- Ablation contributions: `results/ablation_contributions.csv`
- Generalization table: `results/generalization_table.csv`
- Stress recovery table: `results/stress_recovery_table.csv`
- Latency table: `results/latency_table.csv`
- Scalability scaffold: `results/scalability_table.csv`
- Main figures: `results/main_figures/`
- Report-ready summary: `results/summary.md`
- Fairness checklist: `results/fairness_checklist.csv`

## Dashboard

Launch the interactive evaluation dashboard:

```bash
streamlit run src/dashboard/app.py
```

## Repository Layout

- `configs/` experiment and seed configuration
- `src/` model, environment, training, evaluation, dashboard
- `scripts/` orchestration, reproducibility, reporting utilities
- `docs/` protocol, reproducibility, appendix, limitations
- `results/` generated tables, figures, and summaries
- `archive/` non-core drafts and legacy artifacts
