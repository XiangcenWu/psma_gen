#!/usr/bin/env python3
"""Shared, dependency-light utilities for the PET cross-validation baseline."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_DATASET = (
    REPO_ROOT
    / "agentic_pca/agent_dataset/FDG&PSMA双探针_clean_en_with_report.json"
)
DEFAULT_IMAGE_ROOT = Path("/share/home/anyone/Data/RJPCa")
DEFAULT_CACHE_DIR = HERE / "cache/pet_128x128x384"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "agentic_pca/retrieval_agent_inference/outputs/"
    "dl_swin3d_fdg_psma_5fold"
)

SCHEMA_VERSION = "1.0"
LABEL_MAPPING_VERSION = "observed-management-v1"
PREPROCESSING_VERSION = "pet-independent-ct-body-normalization-v1"
DEFAULT_CASE_ID_SALT = "retrieval-agent-inference-v1"
DEFAULT_SEED = 20260727
CLASS_NAMES = (
    "radical_prostatectomy",
    "systemic_treatment",
    "local_treatment",
    "other_examination",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_stat_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def stable_case_id(
    patient_key: str,
    salt: str = DEFAULT_CASE_ID_SALT,
) -> str:
    """Use exactly the same pseudonymous ID rule as the Agent experiment."""
    digest = hashlib.sha256(f"{salt}:{patient_key}".encode("utf-8")).hexdigest()
    return f"case_{digest[:16]}"


def treatment_to_category(treatment: Any) -> str:
    """Exact copy of infer.py's ``observed-management-v1`` label mapping."""
    text = " ".join(str(treatment or "").lower().split())
    if not text:
        raise ValueError("Treatment is empty.")

    if re.search(r"\bradical prostatectomy\b|\brarp\b", text):
        return "radical_prostatectomy"

    systemic_terms = (
        "hormon",
        "androgen deprivation",
        "adt",
        "antiandrogen",
        "abiraterone",
        "enzalutamide",
        "apalutamide",
        "darolutamide",
        "rezvilutamide",
        "bicalutamide",
        "chemotherap",
        "immunotherap",
        "neoadjuvant therapy",
        "novel hormonal agent",
        "nha",
    )
    if any(term in text for term in systemic_terms):
        return "systemic_treatment"

    local_terms = (
        "radiotherap",
        "radiation",
        "ablation",
        "nanoknife",
        "irreversible electroporation",
        "surgical resection",
    )
    if any(term in text for term in local_terms):
        return "local_treatment"

    other_terms = (
        "follow-up",
        "follow up",
        "biopsy",
        "examination",
        "greenlight laser",
        "transurethral",
        "symptomatic treatment",
    )
    if any(term in text for term in other_terms):
        return "other_examination"
    raise ValueError(
        f"Treatment is not covered by {LABEL_MAPPING_VERSION}: {treatment!r}"
    )


def class_counts(
    records: Iterable[Mapping[str, Any]],
    label_key: str = "label",
) -> dict[str, int]:
    counts = Counter(str(record[label_key]) for record in records)
    return {name: counts.get(name, 0) for name in CLASS_NAMES}


def dependency_versions() -> dict[str, str]:
    result = {"python": sys.version.split()[0]}
    for distribution in (
        "numpy",
        "scikit-learn",
        "scipy",
        "torch",
        "monai",
        "nibabel",
    ):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = "not-installed"
    return result


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema: {manifest.get('schema_version')!r}"
        )
    if tuple(manifest.get("classes", [])) != CLASS_NAMES:
        raise ValueError(
            "Manifest class order differs from the fixed observed-management order."
        )
    patients = manifest.get("patients")
    folds = manifest.get("folds")
    if not isinstance(patients, list) or not patients:
        raise ValueError("Manifest patients must be a non-empty list.")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("Manifest must contain exactly five folds.")

    case_ids = [str(record["case_id"]) for record in patients]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Manifest has duplicate case IDs.")
    expected = set(case_ids)
    test_occurrences: Counter[str] = Counter()
    for expected_fold, fold in enumerate(folds):
        if int(fold["fold"]) != expected_fold:
            raise ValueError("Folds must be numbered 0 through 4 in order.")
        train = set(fold["train_case_ids"])
        val = set(fold["val_case_ids"])
        test = set(fold["test_case_ids"])
        if train & val or train & test or val & test:
            raise ValueError(f"Fold {expected_fold} contains overlapping splits.")
        if train | val | test != expected:
            raise ValueError(f"Fold {expected_fold} does not cover every patient.")
        test_occurrences.update(test)
    if set(test_occurrences) != expected or any(
        count != 1 for count in test_occurrences.values()
    ):
        raise ValueError(
            "Across five folds, every patient must occur in the test set exactly once."
        )
