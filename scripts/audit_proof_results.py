#!/usr/bin/env python3
"""Audit proof experiment outputs for evidence-quality warnings."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
CONTROLLER_KEYS = ["MAPPO-STGNN", "MaxPressure", "PressLight", "CoLight", "NSTLight", "FixedTime", "Random"]
METRIC_KEYS = [
    "mean_reward",
    "mean_throughput",
    "mean_travel_time",
    "mean_waiting_time",
    "mean_queue_length",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _same_metric_values(results: dict[str, Any], metric: str) -> list[float]:
    values: list[float] = []
    for controller in CONTROLLER_KEYS:
        payload = results.get(controller)
        if not isinstance(payload, dict):
            continue
        value = _numeric(payload.get(metric))
        if value is not None:
            values.append(value)
    return values


def _has_random_initialized_warning(payload: Any) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"weights_status", "checkpoint_status", "evidence_status"}:
                if isinstance(value, str) and any(term in value.lower() for term in ("random", "untrained", "missing")):
                    return True
            if _has_random_initialized_warning(value):
                return True
    elif isinstance(payload, list):
        return any(_has_random_initialized_warning(item) for item in payload)
    elif isinstance(payload, str):
        return any(term in payload.lower() for term in ("randomly initialized", "weights not found", "untrained"))
    return False


def audit(results: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    failures: list[str] = []

    metadata = results.get("artifact_metadata", {})
    if not isinstance(metadata, dict):
        failures.append("Missing artifact_metadata object.")
    else:
        if metadata.get("artifact_type") != "benchmark_run":
            failures.append("artifact_metadata.artifact_type is not benchmark_run.")
        evidence_status = metadata.get("evidence_status")
        if evidence_status not in {"simulation_smoke", "simulation_benchmark_candidate"}:
            warnings.append(f"Unexpected evidence_status: {evidence_status!r}.")
        if not metadata.get("checkpoint_sha256"):
            warnings.append("Checkpoint SHA-256 is missing; run provenance is incomplete.")

    present_controllers = [name for name in CONTROLLER_KEYS if isinstance(results.get(name), dict)]
    if len(present_controllers) < 3:
        failures.append("Fewer than three controllers were found in benchmark results.")

    for metric in METRIC_KEYS:
        values = _same_metric_values(results, metric)
        if len(values) >= 3 and max(values) - min(values) <= 1e-9:
            warnings.append(
                f"Metric {metric} is identical across {len(values)} controllers; "
                "this suggests actions may not be affecting the simulation or baselines are not distinct."
            )

    for controller in ("CoLight", "NSTLight", "PressLight"):
        payload = results.get(controller)
        if _has_random_initialized_warning(payload):
            warnings.append(f"{controller} appears to use missing or randomly initialized weights.")

    action_similarity = results.get("action_similarity")
    if isinstance(action_similarity, dict):
        for reference, comparisons in action_similarity.items():
            if not isinstance(comparisons, dict):
                continue
            identical = [name for name, score in comparisons.items() if name != reference and score == 1.0]
            if identical:
                warnings.append(
                    f"Action trace for {reference} is identical to {', '.join(identical)} in the sampled trace."
                )

    status = "fail" if failures else ("needs_review" if warnings else "pass")
    return {
        "artifact_type": "proof_audit",
        "status": status,
        "warnings": warnings,
        "failures": failures,
        "honesty_note": (
            "A pass means the proof artifact cleared basic consistency checks. "
            "It does not make the results field evidence or publication-grade evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit proof experiment outputs for evidence-quality warnings.")
    parser.add_argument("--results", default="outputs/benchmark_results.json", help="Benchmark JSON to audit.")
    parser.add_argument("--output", default="results/proof_audit.json", help="Audit JSON output path.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero for warnings as well as failures.")
    args = parser.parse_args()

    results_path = ROOT / args.results
    output_path = ROOT / args.output
    results = _load_json(results_path)
    report = audit(results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    if report["status"] == "fail":
        return 1
    if args.strict and report["status"] == "needs_review":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
