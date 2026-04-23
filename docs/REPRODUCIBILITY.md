# Reproducibility Guide

## One-Command Pipeline

Quick mode:

```bash
python scripts/reproduce_main_results.py --mode quick
```

Full mode:

```bash
python scripts/reproduce_main_results.py --mode full
```

Strict publication suite:

```bash
python scripts/run_publication_suite.py --mode full
```

## What the Pipeline Runs

- Environment verification.
- Phase-1 training entrypoint.
- Baseline benchmark comparison.
- Detailed per-episode evaluation summary.
- Adversarial stress benchmark.
- Publication artifact generation.

The strict suite additionally runs ablation, generalization, stress, and latency benchmarks in one sequence.

## Output Artifacts

Generated under `results/`:
- `main_tables.csv`
- `fairness_checklist.csv`
- `ablation_table_template.csv`
- `summary.md`
- `main_figures/`

Intermediate outputs remain in `outputs/`.

## Seed Reproducibility

Seed list is defined in `configs/phase1.yaml` and mirrored in `docs/APPENDIX_SEEDS.md`.
Use the same list for all publication tables.

## Known Reproducibility Constraints

- Checkpoint and environment configuration must match.
- SUMO binaries and scenario files must be available.
- Cross-machine checkpoint portability requires consistent model/env dimensions.
