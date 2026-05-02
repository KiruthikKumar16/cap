#!/usr/bin/env python3
"""
Test Phase 3 anomaly-aware reward integration.

This is a structural readiness test. It intentionally fails when required
artifacts, configuration, or integration hooks are missing.
"""

import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_anomaly_aware_training() -> bool:
    print("Testing Phase 3: Anomaly-Aware Reward Integration")
    print("=" * 60)

    anomaly_model_path = PROJECT_ROOT / "outputs" / "phase2" / "st_gnn_anomaly_detector.pt"
    _require(
        anomaly_model_path.exists(),
        f"Anomaly model not found: {anomaly_model_path}. Run Phase 2 training first.",
    )
    print(f"[OK] Found anomaly model: {anomaly_model_path}")

    config_path = PROJECT_ROOT / "configs" / "phase1_anomaly_aware.yaml"
    _require(config_path.exists(), f"Config not found: {config_path}")
    print(f"[OK] Found config: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    phase3_config = config.get("phase3", {})
    _require(phase3_config, "Config missing 'phase3' section")
    _require(
        phase3_config.get("enable_anomaly_awareness", False),
        "Anomaly awareness is not enabled in config",
    )
    _require(phase3_config.get("anomaly_weight", 0.0) > 0, "Anomaly weight must be positive")
    print("[OK] Config loaded and validated")

    integration_file = PROJECT_ROOT / "src" / "phase3" / "integration.py"
    _require(integration_file.exists(), f"Integration file not found: {integration_file}")
    integration_content = integration_file.read_text(encoding="utf-8")
    for name in ("init_anomaly_controller", "get_anomaly_controller", "AnomalyAwareTrafficController"):
        _require(name in integration_content, f"Required symbol '{name}' not found in integration.py")
    print("[OK] Integration module structure validated")

    env_file = PROJECT_ROOT / "src" / "phase1" / "traffic_env.py"
    _require(env_file.exists(), f"Environment file not found: {env_file}")
    _require(
        "enable_anomaly_awareness" in env_file.read_text(encoding="utf-8"),
        "Environment missing anomaly awareness support",
    )
    print("[OK] Environment module updated for anomaly awareness")

    train_file = PROJECT_ROOT / "src" / "phase1" / "train_rl.py"
    _require(train_file.exists(), f"Training file not found: {train_file}")
    _require(
        "init_anomaly_controller" in train_file.read_text(encoding="utf-8"),
        "Training script missing anomaly controller initialization",
    )
    print("[OK] Training script updated for anomaly awareness")

    print("=" * 60)
    print("Phase 3 Integration Setup Test PASSED")
    return True


if __name__ == "__main__":
    try:
        test_anomaly_aware_training()
    except Exception as exc:
        print(f"[FAIL] {exc}")
        sys.exit(1)
