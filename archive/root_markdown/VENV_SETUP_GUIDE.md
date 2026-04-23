# Virtual Environment Setup Guide

## Issue
Packages installing to user site-packages instead of venv.

## Quick Fix

### Step 1: Verify Venv is Activated
```bash
# Check Python path (should show .venv path)
python -c "import sys; print(sys.executable)"
```

If it shows `.venv\Scripts\python.exe`, venv is active ✅  
If it shows `C:\Python313\...`, venv is NOT active ❌

### Step 2: Activate Venv Properly
```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Or if that fails:
.venv\Scripts\activate.bat
```

### Step 3: Install Packages in Venv
```bash
# Use venv's pip directly
.venv\Scripts\python.exe -m pip install stable-baselines3[extra] gymnasium torch torch-geometric pyyaml

# Or if venv is activated:
pip install --force-reinstall stable-baselines3[extra] gymnasium
```

### Step 4: Verify Installation
```bash
python -c "import stable_baselines3; print('OK')"
python -m src.phase1.train_rl --config configs/phase1.yaml
```

---

## Alternative: Use System Python (Current Setup)

If packages are in user site-packages and working, you can continue using them:

```bash
# Just run the training script
python -m src.phase1.train_rl --config configs/phase1.yaml
```

The script should work as long as packages are importable.

---

## Check Current Setup

Run this to see where packages are:
```bash
python -c "import stable_baselines3; import sys; print('Python:', sys.executable); print('SB3:', stable_baselines3.__file__)"
```

If it works, you're good to go! ✅
