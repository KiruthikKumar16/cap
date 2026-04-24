# Limitations and Failure Cases

## Current Limitations

- Metadata compatibility is now enforced, but legacy checkpoints without metadata files require retraining or metadata backfill.
- Full publication claims still depend on executing all ablation/stress/generalization/scalability experiments and filling generated result tables with measured values.
- Cross-city and topology-transfer evidence is still limited to currently available scenarios and maps.

## Hard-Nosed Failure Requirements

Publication version must include:
- At least one scenario where a baseline outperforms the primary method.
- One identified degradation mode with quantified impact.
- One mitigation plan tied to measurable follow-up experiment.

## Practical Failure Modes to Report

- Performance drop under severe sensor corruption.
- Stability loss under sudden demand shocks.
- Communication overhead growth in larger CTDE setups.
- Generalization degradation on unseen topology or map distribution shift.
