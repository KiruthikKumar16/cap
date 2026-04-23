# Phase 1 — Immediate Priority: Steps to Finish All 12 SOTA Items

All 12 items are **immediate priority**. Below are concrete steps to finish each. Code changes for items 2–5, 7–8, 11 are implemented or started; items 1, 9, 10, 12 need your environment or scripts.

---

## 1. SUMO runs

**Goal:** Install SUMO and run training/evaluation with real simulation (not placeholder).

**Steps:**

1. **Install SUMO**
   - Windows: Download installer from https://eclipse.dev/sumo/ or `choco install sumo`
   - Add SUMO to PATH (e.g. `C:\Program Files (x86)\Eclipse SUMO\bin` or where `sumo.exe` lives)
   - Verify: open a terminal and run `sumo --version`

2. **Create SUMO network files** (if not present)
   - Run: `python scripts/create_sumo_network.py` (if it generates 2×2) or use existing files in `data/raw/`
   - Ensure `data/raw/` contains: `grid_2x2.net.xml`, `grid_2x2.rou.xml`, `grid_2x2.sumocfg` (paths must be valid; use relative paths from project root)

3. **Train with SUMO**
   - From project root:  
     `python -m src.phase1.train_rl --config configs/phase1.yaml`
   - If SUMO is on PATH, the env will use it; check logs for "Using placeholder mode" (should disappear)

4. **Evaluate with SUMO**
   - `python -m src.phase1.evaluate --config configs/phase1.yaml --episodes 10`
   - You should see throughput (departed/episode) and, when implemented, travel time

---

## 2. Actuated baseline

**Goal:** Compare DQN vs fixed-time vs actuated in evaluation.

**Steps:**

1. **Code:** `evaluate_actuated()` is added in `src/phase1/evaluate.py`. Actuated logic: cycle phases 0→1→2→3 with a fixed phase duration (same as fixed-time for now); you can later replace with detector-based logic (switch when queue on current phase is low or max green reached).

2. **Run evaluation with actuated**
   - `python -m src.phase1.evaluate --config configs/phase1.yaml --episodes 10`
   - The script now evaluates DQN, fixed-time, and actuated and prints a comparison table.

3. **Optional (real actuated):** With SUMO, use detector subscriptions and switch phase when gap-out or max green; implement in `evaluate_actuated()` using `traci.trafficlight.getPhaseDuration` and lane detectors.

---

## 3. Multiple seeds

**Goal:** Run training/eval with 3–5 seeds; report mean ± std.

**Steps:**

1. **Config:** In `configs/phase1.yaml`, under `evaluation`, set `seeds: [42, 43, 44, 45, 46]` (or 3 seeds if time is limited).

2. **Evaluation:** Use `--seeds 5` (or a list) so the eval script runs 5 seeds and aggregates:
   - `python -m src.phase1.evaluate --config configs/phase1.yaml --episodes 10 --seeds 5`
   - Script reports mean ± std for reward, length, throughput, travel time per method (DQN, fixed-time, actuated).

3. **Training with multiple seeds:** Run training once per seed and save checkpoints separately:
   - For seed 42: `python -m src.phase1.train_rl --config configs/phase1.yaml` (uses experiment.seed from config)
   - For seed 43: temporarily set `experiment.seed: 43` and `output.final_model_path: outputs/phase1/dqn_traffic_final_s43.zip`, then run again. Repeat for 44, 45, 46.
   - Then run evaluation with `--seeds 5` and point to the 5 checkpoints (or one checkpoint and 5 eval seeds; see step 2).

4. **Report:** In your report, show a table: Method | Mean reward ± Std | Mean throughput ± Std | (and travel time when SUMO is used).

---

## 4. Double DQN

**Goal:** Enable Double DQN in config and document.

**Steps:**

1. **Config:** In `configs/phase1.yaml`, under `rl`, add:
   - `use_double_dqn: true`

2. **Code:** `create_dqn_agent()` in `src/phase1/dqn_agent.py` now reads `use_double_dqn` from config and passes it to DQN.

3. **Train:** Run training as usual; the agent will use Double DQN.

4. **Document:** In `PHASE1_HYPERPARAMETERS.md` (or report), note "Double DQN: enabled" in the RL section.

---

## 5. Dueling DQN

**Goal:** Add config option and enable Dueling architecture.

**Steps:**

1. **Config:** In `configs/phase1.yaml`, under `rl`, add:
   - `dueling: true`

2. **Code:** `create_dqn_agent()` now builds `policy_kwargs={"dueling": True}` when `dueling: true` and passes it to DQN.

3. **Train:** Run training; the agent will use Dueling DQN.

4. **Document:** In the hyperparameter table, note "Dueling: enabled".

---

## 6. Pressure (PressLight-style)

**Goal:** Add optional pressure term to reward (or features).

**Steps:**

1. **Config:** In `configs/phase1.yaml`, under `reward`, add:
   - `pressure_weight: 0.0`  # set to e.g. 0.02 to enable
   - (Optional) `use_pressure_in_reward: true`

2. **Code:** `RewardCalculator` in `src/phase1/reward_calculator.py` now accepts `pressure_weight` and, when SUMO is used, can add a pressure term (e.g. negative of sum of |incoming_queue − outgoing_queue| per intersection). Placeholder: pressure = 0.

3. **With SUMO:** In `calculate_from_sumo()`, compute per-intersection pressure from controlled lanes (incoming vs outgoing queue lengths) and add `- pressure_weight * pressure` to the reward.

4. **Document:** In the report, state "Optional PressLight-style pressure term in reward (config: pressure_weight)."

---

## 7. Travel time metric

**Goal:** Log and report average travel time when SUMO is used.

**Steps:**

1. **Env info:** `_get_info()` in `src/phase1/traffic_env.py` now includes `travel_time` (0 in placeholder mode). With SUMO, you can subscribe to vehicle depart/arrive and compute travel time per vehicle, then expose sum or count in `info["travel_time"]` and/or `info["travel_time_count"]`.

2. **Evaluation:** `evaluate.py` now collects `info.get("travel_time", 0)` per step and aggregates per episode; it reports "Mean travel time (per episode)" when available.

3. **SUMO implementation:** In `traffic_env.py`, when SUMO is running, subscribe to `traci.vehicle.subscribe(veh_id, [traci.constants.VAR_DEPARTED, ...])` and track depart time; on arrival, compute travel time and add to a running sum; put in `info["travel_time_sum"]` and `info["travel_time_count"]` so eval can compute average.

---

## 8. Statistical test

**Goal:** Report p-value (e.g. t-test) for DQN vs baseline(s).

**Steps:**

1. **Code:** `evaluate.py` now runs a two-sample t-test (DQN vs fixed-time, DQN vs actuated) when `scipy` is available and prints p-value. If p < 0.05, report "DQN is significantly better (p < 0.05)."

2. **Run:** Use multiple episodes (e.g. 30–100) and, if possible, multiple seeds so the test has power:
   - `python -m src.phase1.evaluate --config configs/phase1.yaml --episodes 50`

3. **Report:** In the results table, add a column "p-value (vs DQN)" or state in text: "Improvement over fixed-time is statistically significant (p = 0.02)."

---

## 9. CoLight / PressLight comparison

**Goal:** Implement simplified baselines or cite their numbers with same metrics.

**Steps:**

1. **Option A — Cite numbers:** In your report, add a subsection "Comparison with CoLight and PressLight." From their papers (CoLight: Wei et al., CIKM 2019; PressLight: Li et al., KDD 2021), copy their reported metrics (e.g. average delay or travel time improvement over fixed-time) and state: "On similar settings (multi-intersection, SUMO), CoLight reports X% improvement; PressLight reports Y%. Our GNN-DQN achieves Z%." Use the same metric names (e.g. average travel time) where possible.

2. **Option B — Implement simplified baselines:**
   - **CoLight-style:** Use graph attention over neighbors and a shared policy; implement a small script that uses the same graph/features but a CoLight-like agent (e.g. each intersection has a local Q-net that takes neighbor embeddings) and run eval.
   - **PressLight-style:** Use max-pressure policy: at each step, choose the phase that maximizes pressure (incoming − outgoing queue). Implement in `evaluate.py` as `evaluate_presslight()` using current queue state from the env (or from SUMO lanes) and compare.

3. **Table:** Add a row in the results table: Method | Mean reward | Throughput | Travel time | CoLight (cited) | … | PressLight (cited) | …

---

## 10. Larger networks (4×4, 6×6)

**Goal:** Add 4×4 (and optionally 6×6) grids; report scalability.

**Steps:**

1. **Create 4×4 SUMO network:** Add or use a script that generates `grid_4x4.net.xml`, `grid_4x4.rou.xml`, `grid_4x4.sumocfg` (e.g. extend `scripts/create_sumo_network.py` to support `--grid 4x4`).

2. **Config:** Add a second config file `configs/phase1_4x4.yaml` that overrides `sumo.net_file`, `sumo.route_file`, `sumo.config_file` to the 4×4 files, and optionally `training.total_timesteps` (e.g. 200k for larger net).

3. **Train:** Run training with 4×4:  
   `python -m src.phase1.train_rl --config configs/phase1_4x4.yaml`

4. **Evaluate:** Run evaluation with 4×4 config and report: "On 4×4 grid, DQN achieves … vs fixed-time …; on 2×2, …" to show scalability.

5. **Optional 6×6:** Same idea with `grid_6x6.*` and a dedicated config.

---

## 11. Hyperparameter table

**Goal:** One table in report/README with all Phase 1 hyperparameters.

**Steps:**

1. **File:** `PHASE1_HYPERPARAMETERS.md` is added in the project root with a full table (model, RL, reward, SUMO, training, evaluation). Copy it into your report or README.

2. **Report:** In the "Experimental setup" or "Implementation details" section, paste the table and refer to `configs/phase1.yaml` for reproducibility.

---

## 12. Real learning curves (Fig 7.1–7.3)

**Goal:** Use logged eval data for Fig 7.1–7.3 when available.

**Steps:**

1. **Logging:** Training already runs EvalCallback and writes to `outputs/phase1/logs/` (evaluations.npz). Ensure `eval_freq` and `eval_episodes` in config give enough points (e.g. eval every 5000 steps, 10 episodes per eval).

2. **Figure script:** `scripts/phase1_generate_figures.py` already uses `evaluations.npz` for the reward curve when the file exists and has sufficient variation (reward range > 50). For queue and waiting time, the script still uses synthetic curves unless you add env logging.

3. **Env logging (optional):** In `traffic_env.py`, optionally append to lists (e.g. per-step queue sum, waiting sum) and write them to a log file at the end of each eval episode; then in the figure script, load these logs and plot real queue/waiting curves for Fig 7.2 and 7.3.

4. **Regenerate figures:** After training with SUMO and enough evals, run:  
   `python scripts/phase1_generate_figures.py`  
   so Fig 7.1 (and 7.2/7.3 if you added logging) use real data.

---

## Quick reference: order of execution

1. **Config & code (one-time):** Items 2, 4, 5, 6, 7, 8, 11 — config and code are in place; run eval to see actuated, seeds, travel time placeholder, and p-value.
2. **SUMO (your machine):** Item 1 — install SUMO, create networks, run train and eval.
3. **Seeds:** Item 3 — run eval with `--seeds 5`; optionally train 5 seeds and eval each.
4. **Travel time with SUMO:** Item 7 — implement SUMO travel-time subscription and fill `info["travel_time"]`.
5. **Pressure with SUMO:** Item 6 — implement pressure in `calculate_from_sumo()`.
6. **Report:** Items 9, 11 — add CoLight/PressLight comparison (cite or implement) and paste hyperparameter table.
7. **Scalability:** Item 10 — add 4×4 (and 6×6) networks and configs; train and report.
8. **Figures:** Item 12 — after real runs, regenerate figures so 7.1–7.3 use real curves where possible.
