# Commands

GPU-first command list for this project, in proper order.  
Run from repo root: `C:\Users\Kiruthik Kumar M\cap-1`

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

```powershell
python src/phase1/train_rl.py --config configs/phase1.yaml
```

## 4) Core evaluations (1 episode where supported)

```powershell
python scripts/run_benchmarks.py --config configs/phase1.yaml --checkpoint best_model_stage_2.zip --episodes 1
python scripts/accident_injection.py --config configs/phase1.yaml --checkpoint best_model_stage_2.zip --episodes 1 --sensor-noise-rate 0.10
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
```

## Optional one-shot runner (GPU-enforced)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_end_to_end_gpu.ps1 -Config "configs/phase1.yaml" -Checkpoint "best_model_stage_2.zip" -Episodes 1
```
