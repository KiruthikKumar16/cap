# Experiment Protocol

## Core Story

Robust multi-agent traffic control under non-stationarity with anomaly-aware proactive adaptation.

## Primary Evaluation Design

- **Primary model:** MAPPO-STGNN policy used in `src/phase1/`.
- **Baselines:** CoLight, NSTLight, MaxPressure, FixedTime, Random.
- **Environment:** SUMO with shared phase-1 configuration and identical episode horizon.
- **Evaluation horizon:** fixed per run using the same `sumo.simulation_steps`.
- **Seed protocol:** fixed seed list from `configs/phase1.yaml`.

## Required Statistical Reporting

For each main metric:
- Mean +- standard deviation across seeds.
- 95% confidence interval or bootstrap confidence interval.
- Statistical significance against primary baselines (p-value).

Main metrics:
- Reward
- Throughput
- Travel time
- Waiting time
- Queue length

## Baseline Fairness Checklist

All comparisons must satisfy:
- Same episode budget.
- Same observation/reward interface.
- Same evaluation horizon and scenario.

Fairness output is standardized in `results/fairness_checklist.csv`.

## Required Ablations

- Without ST-GNN.
- Without anomaly module.
- Without predictive phase.
- Without cross-intersection coordination.
- Without perturbation-robustness hooks.

Template output: `results/ablation_table_template.csv`.

## Stress-Test Suite

- Accident injection severity levels.
- Sensor noise/blackout/dropout.
- Topology transfer to unseen map.
- Demand-shock scenarios.

Report recovery/stability indicators, not only average metrics.

## Compute and Scalability Reporting

Include:
- Training wall-clock.
- Inference latency (ms/step).
- Memory/GPU usage.
- Scaling trend by intersection count.
- CTDE communication overhead estimate.
