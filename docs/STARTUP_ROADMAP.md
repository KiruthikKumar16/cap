# Startup Roadmap

This roadmap keeps the project honest while moving it toward a hardware-ready traffic operations product. A stage is not complete until its exit criteria are met and committed.

## Stage 1: Evidence and CI

Goal: make false or stale claims hard to reintroduce.

Work:
- Keep demo, forged, and invalid outputs quarantined under `archive/unverified_evidence/`.
- Run compile, setup smoke, and evidence validation checks in CI.
- Maintain `docs/EVIDENCE_STATUS.md` as the source of truth for what is verified.
- Keep generated tables explicit about `missing`, `not_run`, and synthetic-only evidence.

Exit criteria:
- CI passes on `main`.
- No current result table implies completed benchmark, ablation, scalability, or real anomaly evidence unless the backing run exists.

## Stage 2: Benchmark Harness

Goal: produce reproducible simulation evidence before product claims.

Work:
- Build seeded benchmark runs for fixed-time, random, MaxPressure, actuated, and learned controllers.
- Store scenario, seed, controller, checkpoint, and runtime metadata with every result.
- Report confidence intervals across seeds.
- Fail runs when prerequisites are missing instead of silently continuing.

Exit criteria:
- One command can run a small benchmark suite locally.
- Results can be regenerated from committed configs and documented artifacts.

## Stage 3: First Customer and Use Case

Goal: narrow the product wedge.

Work:
- Choose one initial market: campus, industrial park, private road network, or small municipal corridor.
- Define the buyer, operator, measurable pain, and deployment constraints.
- Convert that into product requirements and pilot success metrics.

Exit criteria:
- A one-page product requirements document exists.
- The target deployment mode is advisory or shadow mode, not unsupervised field control.

## Stage 4: Real Data Ingestion

Goal: move beyond synthetic-only evidence.

Work:
- Define schemas for signal phase timing, detector counts, queue estimates, incidents, and travel-time observations.
- Add importers for CSV logs first, then live adapters.
- Preserve raw data separately from cleaned features.

Exit criteria:
- A real or partner-provided corridor dataset can be replayed through the system.
- Data provenance is recorded with every experiment.

## Stage 5: Safety Action Validator

Goal: ensure ML recommendations cannot violate traffic-signal constraints.

Work:
- Add a rule-based validation layer for minimum green, yellow/all-red clearance, pedestrian phases, max cycle limits, coordination constraints, and fallback behavior.
- Log accepted, rejected, and modified recommendations.
- Treat the model as advisory until this layer is mature.

Exit criteria:
- Invalid phase transitions are rejected by tests.
- Every proposed action has a validation record.

## Stage 6: Operator Dashboard

Goal: make the system understandable to traffic engineers and operators.

Work:
- Show live or replayed corridor state, queue estimates, controller status, recommendations, and reasons.
- Separate evidence-backed metrics from demo visuals.
- Add exports for before/after reports.

Exit criteria:
- An operator can review recommendations without reading code.
- The UI clearly distinguishes advisory, shadow, and controlled modes.

## Stage 7: Hardware-in-the-Loop Lab

Goal: prove reliability before field exposure.

Work:
- Run the controller stack on an edge device.
- Connect to simulator or controller test equipment through a constrained adapter.
- Test watchdog, fallback, restart, power-loss, and network-loss behavior.

Exit criteria:
- A 72-hour hardware-in-the-loop run completes with full logs.
- Fault injection confirms fallback behavior.

## Stage 8: Shadow Pilot

Goal: validate against real operations without controlling signals.

Work:
- Consume live or near-live field data.
- Generate recommendations without applying them.
- Compare recommendations against current plans and observed outcomes.

Exit criteria:
- A pilot report documents uptime, recommendation quality, operator feedback, and measured before/after opportunity.

## Stage 9: Controlled Field Actuation

Goal: move from advisory to limited supervised control only after prior stages pass.

Work:
- Restrict control to approved time windows and intersections.
- Require operator override and automatic fallback.
- Monitor safety, operations, and cybersecurity continuously.

Exit criteria:
- Written approval, safety case, rollback plan, and measured performance report exist.
