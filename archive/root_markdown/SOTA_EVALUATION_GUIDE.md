# Baseline Evaluation Guide & Setup Document

This document tracks all recent architectural/benchmarking modifications and provides a unified, step-by-step guide for new users to train, evaluate, and authenticate State-of-the-Art (Baseline) claims within the **Traffic Resilience Engine**.

## 🛠️ Summary of Baseline Legitimization Upgrades
To ensure our Baseline claim is rigorous, repeatable, and mathematically sound, the following crucial modifications were implemented:
1. **Metric Pipeline Fixed (Accumulative TraCI):** Corrected `traffic_env.py` to recursively accumulate simulation metrics (`step_arrived_vehicles`) instead of overwriting them, restoring correct throughput metrics for our models during evaluations.
2. **NSTLight Architecture Authenticification:** Upgraded `src/baselines/nstlight.py` to explicitly enforce Non-Stationary differencing (`X_t - X_{t-1}`) combined with a 5-head Graph Attention Network. Zero-step evaluation tracks `prev_obs` to mirror conditions.
3. **CoLight Validation:** Explicitly labelled and mapped `colight.py` to natively process neighbor dependencies using standard Graph Attention components perfectly parallel to 2019 specifications. 
4. **Baseline Training Suite (`train_baselines.py`):** Added a custom DQN-style training loop specially engineered to learn traffic baselines locally (defaults to 150 convergence episodes) while adhering to exact target MAPPO reward mathematics ensuring perfect feature parity.
5. **Claims Generator:** Added `scripts/generate_sota_report.py` to safely summarize output percentages against Unified Benchmark datasets.

---

## 🚀 Running the Project from Scratch
For a fresh user dropping into the codebase, follow these exact linear steps to replicate full Baseline capabilities:

### Step 1. Installation 
```bash
git clone https://github.com/KiruthikKumar16/cap.git
cd cap
python -m venv venv
venv\Scripts\activate      # For Windows
source venv/bin/activate   # For Linux/Mac
pip install -r requirements.txt
```
*(Ensure `SUMO_HOME` is linked properly in your environment variables before continuing.)*

### Step 2. Train the Core PPO/ST-GNN Model
Train your custom predictive resilience engine. Note that checkpoints will automatically export to `.zip` binaries. 
```bash
python src/phase1/train_marl.py --config configs/phase2_10x10.yaml
```

### Step 3. Train the Baseline Baselines
You must train the baselines on your exact grid setup to authenticate valid 12-feature comparisons. (Using 150 episodes natively supports sufficient convergence).
```bash
python scripts/train_baselines.py --model nstlight --episodes 150
python scripts/train_baselines.py --model colight --episodes 150
```
This automatically caches `nstlight.pth` and `colight.pth` into the `checkpoints/` root folder.

### Step 4. Execute the Head-to-Head Benchmarks
Run a fully unified simulator test incorporating both your `.zip` agent and your baseline checkpoints. Memory trackers are completely unified to supply identical context parameters across tests.
```bash
python scripts/run_benchmarks.py --checkpoint "marl_ppo_traffic.zip" --config configs/phase2_10x10.yaml --episodes 5
```

### Step 5. Generate Claims & Reports
Lastly, synthesize the raw JSON datasets into visual charts and markdown-certified statistics.
```bash
# Output Bar-Charts, Heatmaps, and t-SNE files to outputs/plots/
python scripts/generate_plots.py

# Auto-write the unified Markdown Baseline summary for presentations into outputs/sota_claim.md
python scripts/generate_sota_report.py
```
Preview `outputs/sota_claim.md` directly for the final presentation thesis phrasing.
