#!/usr/bin/env python3
"""CI guardrails for honest evidence reporting.

This script is intentionally conservative. It does not prove that the project is
publication-ready; it only blocks known-bad states from creeping back in.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _failures() -> list[str]:
    failures: list[str] = []

    if (ROOT / "FAST_VAL_RESULTS").exists():
        failures.append("FAST_VAL_RESULTS must stay quarantined under archive/unverified_evidence/.")

    benchmark_path = ROOT / "outputs" / "benchmark_results.json"
    if benchmark_path.exists():
        with benchmark_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metadata = payload.get("artifact_metadata", {}) if isinstance(payload, dict) else {}
        if metadata.get("artifact_type") == "presentation_demo":
            failures.append("outputs/benchmark_results.json is marked presentation_demo.")

    archive_path = ROOT / "archive" / "unverified_evidence"
    if not archive_path.exists():
        failures.append("archive/unverified_evidence/ is missing.")

    evidence_doc = ROOT / "docs" / "EVIDENCE_STATUS.md"
    if not evidence_doc.exists():
        failures.append("docs/EVIDENCE_STATUS.md is missing.")
    else:
        evidence_text = evidence_doc.read_text(encoding="utf-8").lower()
        for required in ("not yet publication evidence", "quarantined"):
            if required not in evidence_text:
                failures.append(f"docs/EVIDENCE_STATUS.md must include '{required}'.")

    gaps_path = ROOT / "results" / "ablation_evidence_gaps.csv"
    if not gaps_path.exists():
        failures.append("results/ablation_evidence_gaps.csv is missing.")
    else:
        with gaps_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            failures.append("results/ablation_evidence_gaps.csv must list missing ablations.")
        for row in rows:
            if row.get("evidence_status") != "missing":
                failures.append("ablation evidence gaps must remain marked as missing until rerun.")
                break

    legacy_template = ROOT / "results" / "ablation_table_template.csv"
    if legacy_template.exists():
        failures.append("results/ablation_table_template.csv should not be revived as evidence.")

    return failures


def main() -> int:
    failures = _failures()
    if failures:
        print("Evidence validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Evidence validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
