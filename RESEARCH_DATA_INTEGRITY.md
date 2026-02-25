# Research and Patent Data Integrity

This document states how the project ensures **no false, placeholder, or synthetic data** is used in any results or figures intended for **patent or research publication**.

## Principles

1. **Placeholder mode is for development only.** When SUMO is not installed or not running, the environment can still step (using a placeholder reward and features) so that code paths and training pipelines can be tested. **Throughput and travel time in that mode are always 0** and are never reported as simulation results.

2. **All reported metrics come from real simulation.** For patent or research:
   - **Throughput** (departed vehicles) and **travel time** are only taken from SUMO via TraCI when `sumo_running` is true.
   - The evaluation summary includes a flag **`used_sumo`**. When `used_sumo` is false, comparison figures for throughput and travel time show a clear notice that real SUMO is required; no synthetic curves or fake numbers are plotted.

3. **Figures use only real or explicitly non-data content.**
   - **Reward per episode (Fig 7.1):** Plotted only from real evaluation data (`evaluations.npz` from SUMO runs with sufficient variation). Otherwise the figure shows: *"Reward per episode requires real evaluation data. Not shown for patent/research integrity."*
   - **Queue length (Fig 7.2) and Waiting time (Fig 7.3):** Require real SUMO simulation and environment logging. Until such data exists, figures show: *"Requires real SUMO simulation and environment logging. Not shown for patent/research integrity."*
   - **Comparison charts (throughput, travel time):** Only plot real data when `used_sumo` is true in `evaluation_summary.json`. Otherwise they show a notice that real SUMO is required.
   - **Improvement % chart:** Throughput improvement is included only when evaluation was run with SUMO (`used_sumo` true and mean throughput &gt; 0).

## Implementation Summary

| Component | Behavior |
|----------|----------|
| `traffic_env._get_info()` | In placeholder mode: `departed=0`, `travel_time=0`, `placeholder_mode=True`. No synthetic formulae. |
| `evaluate.py` | Detects `placeholder_mode` from env info; sets `used_sumo` in saved summary. Prints note when placeholder mode is used. |
| `phase1_generate_figures.py` | Reward curve only from real eval data; queue/waiting show notice if no real data; throughput/travel time comparison only when `used_sumo`. |

## For Publication

- **Run training and evaluation with SUMO** installed and configured so that `used_sumo` is true and all metrics (reward, throughput, travel time) are from simulation.
- **Do not use** evaluation summaries or figures generated in placeholder mode as evidence in a patent or paper; use them only for development and demos.
- **Cite this document** (or equivalent) in your methods to state that reported results are from real simulation only.
