param(
    [string]$Config = "configs/phase1.yaml",
    [string]$Checkpoint = "best_model_stage_2.zip",
    [int]$TrainTimesteps = 0,
    [int]$Episodes = 1
)

$ErrorActionPreference = "Stop"

function Run-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )
    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor Cyan
    Write-Host ("[STEP] " + $Name) -ForegroundColor Cyan
    Write-Host ("=" * 72) -ForegroundColor Cyan
    & $Action
}

function Assert-GPU {
    Write-Host "Checking CUDA / GPU availability..." -ForegroundColor Yellow
    python -c "import torch,sys; print('torch_version=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count()); sys.exit(0 if torch.cuda.is_available() else 2)"
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA GPU is not available in current Python env. Activate your CUDA-enabled venv and retry."
    }
}

function Ensure-Checkpoint {
    param([string]$Path)
    if (!(Test-Path $Path)) {
        throw "Checkpoint not found at '$Path'. Train first or pass -Checkpoint with a valid .zip path."
    }
}

Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

Run-Step "GPU preflight" {
    Assert-GPU
    $env:CUDA_VISIBLE_DEVICES = "0"
    Write-Host "CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES"
}

if ($TrainTimesteps -gt 0) {
    Run-Step "Phase 1 training (GPU expected by torch)" {
        python "src/phase1/train_rl.py" --config $Config
        if ($LASTEXITCODE -ne 0) { throw "Training failed." }
    }
}

Run-Step "Checkpoint validation" {
    Ensure-Checkpoint -Path $Checkpoint
}

Run-Step "Latency benchmark (CUDA forced)" {
    python "scripts/latency_benchmark.py" --gpu
    if ($LASTEXITCODE -ne 0) { throw "Latency benchmark failed." }
}

Run-Step "MAPPO vs NSTLight benchmark" {
    python "scripts/run_benchmarks.py" --config $Config --checkpoint $Checkpoint --episodes $Episodes
    if ($LASTEXITCODE -ne 0) { throw "Benchmark run failed." }
}

Run-Step "Adversarial accident + sensor failure stress test" {
    python "scripts/accident_injection.py" --config $Config --checkpoint $Checkpoint --episodes $Episodes --sensor-noise-rate 0.10
    if ($LASTEXITCODE -ne 0) { throw "Adversarial stress run failed." }
}

Run-Step "Zero-shot Bengaluru generalization" {
    python "scripts/evaluate_generalization.py"
    if ($LASTEXITCODE -ne 0) { throw "Generalization evaluation failed." }
}

Run-Step "Plot generation (modern visuals)" {
    python "scripts/generate_plots.py"
    if ($LASTEXITCODE -ne 0) { throw "Plot generation failed." }
}

Write-Host ""
Write-Host ("=" * 72) -ForegroundColor Green
Write-Host "[DONE] End-to-end GPU pipeline complete." -ForegroundColor Green
Write-Host "Artifacts:"
Write-Host "  - outputs/latency/inference_latency.json"
Write-Host "  - outputs/benchmark_results.json"
Write-Host "  - outputs/phase3/adversarial_benchmark.json"
Write-Host "  - outputs/phase4/zero_shot_generalization.json"
Write-Host "  - outputs/plots/*.png"
Write-Host ("=" * 72) -ForegroundColor Green
