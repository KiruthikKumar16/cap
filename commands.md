# Commands

GPU-first command list for this project, in proper order.  
Run from repo root: `C:\Users\Kiruthik Kumar M\cap-1`

**CPU-only dev:** Section 0 exits with code `2` when CUDA is missing (by design). For a quick training smoke test, cancel after you see the run start, or use  
`python src/phase1/train_marl.py --config configs/phase1.yaml --total-timesteps 2048`  
(needs at least `n_steps` from the config, typically 2048). Latency: `python scripts/latency_benchmark.py --gpu` falls back to CPU if CUDA is unavailable.

## 0) GPU preflight (required)

```powershell
$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES="0"
python -c "import torch,sys; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); sys.exit(0 if torch.cuda.is_available() else 2)"
```

## 1) Setup

```powershell
python -m pip install -r requirements.txt
python scripts/setup_environment.py
python scripts/check_sumo.py
```

## 2) Data / scenario generation (if needed)

```powershell
python scripts/create_sumo_network.py
python scripts/create_sumo_scenario.py
python scripts/generate_anomaly_data.py
```

## 3) Training

`configs/phase1.yaml` uses **PPO / MAPPO** (`rl.algorithm: PPO`). Use the MARL trainer (writes `marl_ppo_traffic.zip` in the repo root):

```powershell
python src/phase1/train_marl.py --config configs/phase1.yaml
```

For a **DQN-only** config (`rl.algorithm` not `PPO`), use:

```powershell
python src/phase1/train_rl.py --config configs/phase1.yaml
```

## 4) Core evaluations (1 episode where supported)

**Time:** `run_benchmarks` / `accident_injection` each drive full SUMO episodes (`simulation_steps` in the YAML, often 3600). `evaluate_generalization` on large nets is slow. `run_generalization_test` runs a **full** 5x5 train then eval unless you edit the script. `run_ablation_study` runs **four** train+eval cycles. On a dev machine without GPU, use Ctrl+C after a successful start, or shorten timesteps / configs locally.

After training, the default MAPPO checkpoint is `marl_ppo_traffic.zip` (repo root). Pass that path (or your own `.zip`) to evaluation scripts:

```powershell
python scripts/run_benchmarks.py --config configs/phase1.yaml --checkpoint marl_ppo_traffic.zip --episodes 1
python scripts/accident_injection.py --config configs/phase1.yaml --checkpoint marl_ppo_traffic.zip --episodes 1 --sensor-noise-rate 0.10
python scripts/evaluate_generalization.py
python scripts/run_generalization_test.py
python scripts/real_sumo_evaluation.py
python scripts/run_ablation_study.py
```

## 5) Latency (explicit GPU)

```powershell
python scripts/latency_benchmark.py --gpu
```

## 6) Visualizations / plots / figures

```powershell
python scripts/generate_plots.py
python scripts/generate_heatmap.py
python scripts/sota_visualizations.py
python scripts/phase1_generate_figures.py
python scripts/phase2_generate_figures.py
```

## 7) Tests

```powershell
python scripts/test_setup.py
python scripts/test_phase1.py
python scripts/test_phase2.py
python scripts/test_phase3.py
python scripts/test_sota_integration.py
python scripts/test_phase3_integration.py
```

## 8) Demo

```powershell
python scripts/run_phase1_demo.py
python scripts/run_phase1_demo.py --quick
```

`--quick` runs a short MAPPO smoke train (`--total-timesteps 2048`) and a 1-episode / 1-seed eval so the full pipeline finishes in reasonable time on CPU.

## Optional one-shot runner (GPU-enforced)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_end_to_end_gpu.ps1 -Config "configs/phase1.yaml" -Checkpoint "marl_ppo_traffic.zip" -Episodes 1
```
