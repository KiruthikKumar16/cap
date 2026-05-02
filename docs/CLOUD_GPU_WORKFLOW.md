# Cloud GPU Workflow

This project can be trained or evaluated from a free cloud notebook, but free GPU access is not guaranteed. Treat every cloud run as opportunistic compute, not production infrastructure.

## Recommended Free Options

1. Google Colab
   - Best first choice for notebook-driven GPU experiments.
   - Free tier can provide NVIDIA GPU/TPU access, but availability and session length vary.
   - Use Drive for checkpoint persistence.

2. Kaggle Notebooks
   - Good alternative when Colab has no GPU available.
   - Internet and accelerator settings must be enabled in notebook settings.
   - Quotas and available GPU types change, so check the notebook sidebar before planning a long run.

3. GitHub Codespaces
   - Useful for CPU-only development, documentation, CI debugging, and small smoke tests.
   - Not a practical free GPU training environment.

4. Lightning AI Studio
   - Useful if free credits are available on your account.
   - Treat it like a limited-credit option, not a permanent free GPU source.

## Honest Compute Strategy

Use your local CPU machine for:
- Editing code.
- Running `python scripts/ci_validate_evidence.py`.
- Running `python scripts/test_setup.py`.
- Reviewing generated artifacts.

Use cloud GPU for:
- Training or re-training heavier models.
- Multi-seed simulation experiments if local runtime is too slow.
- Generating proof artifacts before paper writing.

Do not use cloud GPU results as field evidence. They are still simulation evidence unless the input data is real and the experiment protocol says so.

## Colab Setup

Create a new Colab notebook, set runtime accelerator to GPU, then run:

```bash
!git clone https://github.com/KiruthikKumar16/cap.git
%cd cap
!python -m pip install --upgrade pip wheel setuptools
!python -m pip install -r requirements.txt
```

Verify the environment:

```bash
!python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
!python -m compileall -q src scripts
!python scripts/ci_validate_evidence.py
!python scripts/test_setup.py
```

If CUDA is false, you can still run CPU smoke checks, but do not call it a GPU run.

## SUMO On Colab

SUMO is required for meaningful Phase 1 traffic simulation. In a Colab Linux runtime, try:

```bash
!sudo apt-get update
!sudo apt-get install -y sumo sumo-tools sumo-doc
```

Then verify:

```bash
!sumo --version
!python scripts/check_sumo.py
```

If SUMO install fails with `ppa.launchpadcontent.net` or `ubuntugis` timeout errors, run the Colab recovery script:

```bash
!bash scripts/setup_colab_sumo.sh
```

The script disables unstable Launchpad PPA entries for the current runtime, installs SUMO from Ubuntu repositories, and runs `scripts/check_sumo.py`.

If SUMO still fails or `scripts/check_sumo.py` fails, stop and record that failure. Do not report benchmark numbers from a broken simulator setup.

## Checkpoint Persistence

Colab runtimes are temporary. Store heavy checkpoints outside the runtime.

Option A: Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Copy a checkpoint into the repo before running proof experiments:

```bash
!cp /content/drive/MyDrive/cap-checkpoints/marl_ppo_traffic.zip ./marl_ppo_traffic.zip
```

If you do not already have `marl_ppo_traffic.zip`, train one after SUMO is working:

```bash
!python src/phase1/train_marl.py --config configs/phase1.yaml --total-timesteps 100000
```

For a very short pipeline test only, you can use fewer timesteps:

```bash
!python src/phase1/train_marl.py --config configs/phase1.yaml --total-timesteps 10000
```

Honesty rule: a 10,000-step checkpoint is a smoke-test checkpoint, not a trained result you should use for paper claims.

Option B: GitHub release artifact or private storage
- Download the checkpoint at runtime.
- Record the download URL and checksum in your notes.
- Do not commit large checkpoints unless the repo policy changes.

## Proof Experiment Command

Run this only after SUMO and the checkpoint are available:

```bash
!python scripts/run_proof_experiment.py \
  --config configs/phase1.yaml \
  --checkpoint marl_ppo_traffic.zip \
  --episodes 3 \
  --seeds 3
```

Outputs:
- `outputs/benchmark_results.json`
- `results/main_tables.csv`
- `results/summary.md`
- `results/proof_manifest.json`

Honest interpretation:
- `episodes 3 --seeds 3` is an early proof run.
- It is not a publication-grade final benchmark.
- Use it to decide what to improve next.

## If Training Is Needed

Use the smallest run first:

```bash
!python src/phase2/anomaly_trainer.py --epochs 5 --output_dir outputs/phase2
!python scripts/test_phase2.py
```

For Phase 1 controller training, do not start a long training job until:
- SUMO works.
- The config is fixed.
- The checkpoint/output path is clear.
- You know where the checkpoint will be persisted.

## Download Results

Zip only the lightweight evidence files:

```bash
!zip -r proof_results.zip results outputs/phase1/evaluation_summary.json outputs/phase2/anomaly_eval_summary.json outputs/phase3/adversarial_benchmark.json
```

Then download `proof_results.zip` from the notebook UI or copy it to Drive.

## Minimum Run Log To Keep

For every cloud run, record:
- Platform: Colab, Kaggle, Codespaces, or Lightning.
- GPU model or CPU-only.
- Git commit.
- Config path.
- Checkpoint path and checksum.
- Episodes and seeds.
- Whether SUMO passed.
- Exact command used.
- Result files generated.

If any of those are missing, the run is not auditable.
