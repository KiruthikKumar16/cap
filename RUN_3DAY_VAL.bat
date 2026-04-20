@echo off
setlocal

:: 3-Day Fast-Track Mega Validation Suite (Capstone Edition)
:: Runs 5x5 Grid training for SOTA baselines and MAPPO-STGNN.
:: Generates a full suite of professional research plots (Heatmaps, Clusters, Bars).

echo ============================================================
echo   STARTING CAPSTONE FAST-TRACK VALIDATION (5x5 GRID)
echo ============================================================

echo ============================================================
echo   LEGIT START: 80 episodes, 300k timesteps, Medium Traffic
echo   Estimated duration: 12-15 hours.
echo ============================================================

set VENV_PATH=.\venv_gpu\Scripts\python.exe
set CONFIG=configs/fast_validate.yaml

:: Clean up old fast-track results
echo [0/4] Cleaning previous results...
if exist "FAST_VAL_RESULTS" (
    del /q "FAST_VAL_RESULTS\*.csv" >nul 2>&1
    del /q "FAST_VAL_RESULTS\*.zip" >nul 2>&1
    if exist "FAST_VAL_RESULTS\plots" del /q "FAST_VAL_RESULTS\plots\*.png" >nul 2>&1
) else (
    mkdir "FAST_VAL_RESULTS"
)

:: Step 1: NSTLight
if exist episode_metrics.csv del episode_metrics.csv
echo [1/4] Training NSTLight Baseline (80 episodes)...
%VENV_PATH% scripts/train_baselines_fast.py --config %CONFIG% --model nstlight --episodes 80
if exist episode_metrics.csv (
    move /y episode_metrics.csv FAST_VAL_RESULTS\metrics_nstlight.csv
)

:: Step 2: CoLight
if exist episode_metrics.csv del episode_metrics.csv
echo [2/4] Training CoLight Baseline (80 episodes)...
%VENV_PATH% scripts/train_baselines_fast.py --config %CONFIG% --model colight --episodes 80
if exist episode_metrics.csv (
    move /y episode_metrics.csv FAST_VAL_RESULTS\metrics_colight.csv
)

:: Step 3: MAPPO-STGNN
if exist episode_metrics.csv del episode_metrics.csv
echo [3/4] Training Main MARL Model (MAPPO-STGNN) on 5x5...
%VENV_PATH% src/phase1/train_marl.py --config %CONFIG%
if exist episode_metrics.csv (
    move /y episode_metrics.csv FAST_VAL_RESULTS\metrics_mappo.csv
)
if exist marl_ppo_traffic.zip (
    move /y marl_ppo_traffic.zip FAST_VAL_RESULTS\mappo_5x5_model.zip
)

:: Step 4: Mega Visualization Suite
echo [4/4] Generating Professional Visualization Suite...
%VENV_PATH% scripts/generate_fast_plots.py

echo ============================================================
echo   CAPSTONE VALIDATION COMPLETE
echo ============================================================
echo Final metrics: FAST_VAL_RESULTS/
echo Mega Plots:    FAST_VAL_RESULTS/plots/
echo ============================================================
echo   Check FAST_VAL_RESULTS/plots/ for:
echo   - Congestion Heatmaps (per model)
echo   - Latent Cluster Map (t-SNE)
echo   - Efficiency Pareto Frontiers
echo   - Multi-metric Convergence Linecharts
echo   - Summary Performance Barcharts
echo ============================================================
pause
