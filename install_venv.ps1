# Install packages in virtual environment
# Run this script from the project root directory

Write-Host "Installing packages in virtual environment..." -ForegroundColor Green

# Check if venv exists
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Error: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Create venv first: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

# Use venv's pip to install packages
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install stable-baselines3[extra] gymnasium torch torch-geometric pyyaml numpy pandas scikit-learn

Write-Host "`nVerifying installation..." -ForegroundColor Green
.venv\Scripts\python.exe -c "import stable_baselines3; import gymnasium; print('[OK] Packages installed successfully')"

Write-Host "`nDone! You can now run:" -ForegroundColor Green
Write-Host "  python -m src.phase1.train_rl --config configs/phase1.yaml" -ForegroundColor Cyan
