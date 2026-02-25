# Phase 1 Evaluation — Run from scratch

Use the **clean evaluation** script (core concepts only, no legacy logic).

## 1. Run evaluation (clean)

```bash
python -m src.phase1.evaluate_clean --config configs/phase1.yaml --episodes 10 --seeds 3 --save-summary outputs/phase1/evaluation_summary.json
```

- **Fixed-time**: action = `(step // 30) % 4` per intersection (cycle every 30 steps).
- **DQN**: action = `model.predict(obs)[0]` on the same wrapped env.
- First 5 actions per policy are printed so you can verify they differ.
- Summary is written to `outputs/phase1/evaluation_summary.json`.

## 2. Generate figures

```bash
python scripts/phase1_generate_figures.py
```

Figures read from `outputs/phase1/evaluation_summary.json` (same format as before).

## 3. If DQN and Fixed-time rewards are still identical

The current DQN was trained **without** the pressure term (or with different reward). So it never learned to reduce vehicle count on lanes.

**Re-train** with the current config (pressure_weight > 0 in `configs/phase1.yaml`), then run evaluation again:

```bash
python -m src.phase1.train_rl --config configs/phase1.yaml
python -m src.phase1.evaluate_clean --episodes 10 --seeds 3 --save-summary outputs/phase1/evaluation_summary.json
python scripts/phase1_generate_figures.py
```

## Core concepts (clean script)

1. **Fixed-time**: raw env, same seeds, action = cycle per intersection.
2. **DQN**: wrapped env (MultiDiscreteToDiscrete + GNNObservation), same seeds, action = model.predict(obs).
3. **Metrics**: reward, departed (throughput), travel_time, waiting_time, queue_length from env `info` each step; summed/averaged per episode.
4. **Seeds**: one seed per episode (round-robin) so you get variance and std > 0.
5. **No placeholders**: all values come from SUMO TraCI.
