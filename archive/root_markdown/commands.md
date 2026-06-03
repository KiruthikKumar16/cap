# Research Pipeline Commands

Standard execution flow for producing research-grade results on Linux.

## 1) Environment Setup

```bash
source venv/bin/activate
pip install -r requirements.txt
python scripts/test_setup.py
```

## 2) Phase 1: Foundation & Baselines

Generate topologies and train the competitive landscape.

```bash
# Topology Diversity
python scripts/generate_random_maps.py --count 10 --output_dir data/raw/procedural/

# Baseline: CoLight
python scripts/train_baselines.py --model colight --config configs/phase1.yaml --episodes 150

# Baseline: NSTLight
python scripts/train_baselines.py --model nstlight --config configs/phase1.yaml --episodes 150
```

## 3) Phase 2: Proposed Model Training

Train the MAPPO-STGNN agent with regional hierarchical coordination.

```bash
python src/phase1/train_marl.py \
    --config configs/phase1.yaml \
    --total-timesteps 100000 \
    --use_regional_critics True
```

## 4) Phase 3: Anomaly Detection & Risk Sensing

Train the ST-GNN detector to identify accidents and non-stationary events.

```bash
# Data Collection
python scripts/generate_anomaly_data.py --checkpoint marl_ppo_traffic.zip --episodes 10
# Detector Training
python src/phase2/anomaly_trainer.py --epochs 30
```

## 5) Comprehensive Scientific Evaluation

Run the full suite of benchmarks, ablations, and generalization tests.

```bash
# Baseline Comparison (multi-seed)
python scripts/run_benchmarks.py --config configs/phase1.yaml --checkpoint marl_ppo_traffic.zip --episodes 5 --seeds 3

# Ablation Study (Component Attribution)
python scripts/run_ablation_study.py

# Zero-Shot Generalization (Real-world map)
python scripts/evaluate_generalization.py --config configs/bengaluru_city.yaml --checkpoint marl_ppo_traffic.zip

# Latency Benchmark (Efficiency)
python scripts/latency_benchmark.py --cpu
```

## 6) Publication Artifacts

Generate the final LaTeX tables and statistical summaries.

```bash
# Statistical Significance (P-values/CI)
python scripts/generate_statistical_tables.py

# Final PDF/CSV Artifacts
python scripts/generate_publication_artifacts.py --mode full
```

## 7) Integrated Demo

Launch the research dashboard for interactive visualization.

```bash
streamlit run src/dashboard/app.py --server.port 8505
```
