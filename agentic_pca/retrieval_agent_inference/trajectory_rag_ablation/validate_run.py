#!/usr/bin/env python3
"""Validate structural completeness and provenance of one ablation output."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--require-all-completed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.run_dir / "config.json"
    manifest_path = args.run_dir / "patient_manifest.json"
    if not config_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Run is missing config.json or patient_manifest.json")
    config = load_json(config_path)
    manifest = load_json(manifest_path)
    if not isinstance(config, dict) or not isinstance(manifest, dict):
        raise TypeError("Config and manifest must be JSON objects")
    selection = config.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Config selection metadata is missing")
    expected_files = (
        int(selection["selected_patients"])
        * int(selection["num_trajectories"])
    )
    experiment_id = config["inference_config"]["ablation_spec"]["id"]
    fingerprint = config["inference_config_fingerprint"]
    paths = sorted(
        (args.run_dir / "patients").glob("case_*/trajectory_*.json")
    )
    statuses: Counter[str] = Counter()
    trajectory_ids: set[str] = set()
    errors: list[str] = []
    for path in paths:
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: invalid JSON: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"{path}: top level is not an object")
            continue
        status = str(payload.get("status", "missing"))
        statuses[status] += 1
        trajectory_id = payload.get("trajectory_id")
        if not isinstance(trajectory_id, str) or trajectory_id in trajectory_ids:
            errors.append(f"{path}: invalid or duplicate trajectory_id")
        else:
            trajectory_ids.add(trajectory_id)
        if payload.get("provenance", {}).get(
            "inference_config_fingerprint"
        ) != fingerprint:
            errors.append(f"{path}: inference fingerprint differs")
        if payload.get("ablation", {}).get("experiment_id") != experiment_id:
            errors.append(f"{path}: experiment ID differs")
    if len(paths) != expected_files:
        errors.append(
            f"expected {expected_files} trajectory files, found {len(paths)}"
        )
    unexpected_statuses = sorted(set(statuses) - {"completed", "failed"})
    if unexpected_statuses:
        errors.append(f"unexpected statuses: {unexpected_statuses}")
    if len(manifest) != int(selection["selected_patients"]):
        errors.append(
            "patient_manifest size differs from selected patient count"
        )
    if args.require_all_completed and statuses["failed"]:
        errors.append(f"{statuses['failed']} trajectories failed")
    report = {
        "status": "valid" if not errors else "invalid",
        "run_dir": str(args.run_dir),
        "experiment_id": experiment_id,
        "expected_trajectory_files": expected_files,
        "actual_trajectory_files": len(paths),
        "status_counts": dict(statuses),
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
