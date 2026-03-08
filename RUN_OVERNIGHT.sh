#!/bin/bash
# Phase 1: Run 100k training overnight for publication quality

# Step 1: Update config to 100k
sed -i 's/total_timesteps: 20000/total_timesteps: 100000/g' configs/phase1.yaml

# Step 2: Train (4-5 hours)
python -m src.phase1.train_rl --config configs/phase1.yaml

# Step 3: Evaluate (30 min)
python -m src.phase1.evaluate_clean --episodes 10 --seeds 3 --save-summary outputs/phase1/evaluation_summary.json

# Step 4: Generate figures
python scripts/phase1_generate_figures.py

echo "Done! Check outputs/phase1/figures/ for results."
