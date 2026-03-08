@echo off
REM Phase 1: Run 100k training overnight for publication quality

echo Step 1: Updating config to 100k...
powershell -Command "(gc configs/phase1.yaml) -replace 'total_timesteps: 20000', 'total_timesteps: 100000' | Out-File -encoding ASCII configs/phase1.yaml"

echo Step 2: Training 100k steps (4-5 hours)...
python -m src.phase1.train_rl --config configs/phase1.yaml

echo Step 3: Evaluating (30 min)...
python -m src.phase1.evaluate_clean --episodes 10 --seeds 3 --save-summary outputs/phase1/evaluation_summary.json

echo Step 4: Generating figures...
python scripts/phase1_generate_figures.py

echo Done! Check outputs/phase1/figures/ for results.
pause
