#!/usr/bin/env python3
"""Leakage-safe structured whole-organ SUV logistic-regression baseline.

The feature schema is fixed before cross-validation: for every anatomical ROI
present in the cohort, the script uses FDG and PSMA ``suv_mean`` and
``suv_max`` in a deterministic order.  It reuses the train/validation/test
case IDs frozen in the DL-only manifest.  Within each fold, preprocessing and
model selection are performed without access to that fold's held-out test
patients.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "agentic_pca/DL-only/cache/pet_128x128x384/manifest.json"
DEFAULT_SUV_DIR = ROOT / "agentic_pca/agent_dataset/suv_output_by_patient"
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "agentic_pca/retrieval_agent_inference/outputs"
    / "comparison_baselines_Qwen3.5-9B"
    / "structured_suv_ml"
)

BASELINE_ID = "structured_suv_ml"
SCHEMA_VERSION = "structured-suv-ml-v1"
EXPECTED_PATIENTS = 249
EXPECTED_FOLDS = tuple(range(5))
CLASS_NAMES = (
    "radical_prostatectomy",
    "systemic_treatment",
    "local_treatment",
    "other_examination",
)
TRACERS = ("FDG", "PSMA")
SUV_STATISTICS = ("suv_mean", "suv_max")

# Pre-registered hyperparameter grid.  Ties prefer balanced accuracy, then
# accuracy, then the smaller C (stronger regularisation).
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
SELECTION_METRIC = "validation macro-F1"
BASE_SEED = 20260727
MAX_ITER = 5000


class BaselineValidationError(ValueError):
    """Raised when an input or a split violates the frozen protocol."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
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


def jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
    )


def dependency_versions() -> dict[str, str]:
    result = {"python": sys.version.split()[0]}
    for distribution in ("numpy", "scikit-learn"):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = "not-installed"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--suv-dir", type=Path, default=DEFAULT_SUV_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the complete cohort, feature schema, and frozen splits only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing this baseline's existing output files.",
    )
    return parser.parse_args()


def _require_string(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise BaselineValidationError(f"{description} must be a non-empty string")
    return value


def _require_string_list(value: Any, description: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise BaselineValidationError(
            f"{description} must be a list of non-empty strings"
        )
    if len(value) != len(set(value)):
        raise BaselineValidationError(f"{description} contains duplicate case IDs")
    return list(value)


def validate_manifest(payload: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise BaselineValidationError("DL-only manifest must be a JSON object")
    if tuple(payload.get("classes", ())) != CLASS_NAMES:
        raise BaselineValidationError(
            "Manifest class order differs from the fixed four-class protocol"
        )
    if payload.get("label_mapping_version") != "observed-management-v1":
        raise BaselineValidationError(
            "Manifest label mapping is not observed-management-v1"
        )

    raw_patients = payload.get("patients")
    if not isinstance(raw_patients, list) or len(raw_patients) != EXPECTED_PATIENTS:
        size = len(raw_patients) if isinstance(raw_patients, list) else None
        raise BaselineValidationError(
            f"Manifest must contain exactly {EXPECTED_PATIENTS} patients, found {size}"
        )

    patients: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    patient_keys: set[str] = set()
    for index, raw in enumerate(raw_patients):
        if not isinstance(raw, dict):
            raise BaselineValidationError(f"patients[{index}] must be an object")
        case_id = _require_string(raw.get("case_id"), f"patients[{index}].case_id")
        patient_key = _require_string(
            raw.get("patient_key"), f"patients[{index}].patient_key"
        )
        label = _require_string(raw.get("label"), f"patients[{index}].label")
        label_index = raw.get("label_index")
        if isinstance(label_index, bool) or not isinstance(label_index, int):
            raise BaselineValidationError(
                f"patients[{index}].label_index must be an integer"
            )
        if not 0 <= label_index < len(CLASS_NAMES):
            raise BaselineValidationError(
                f"patients[{index}].label_index is outside the class range"
            )
        if label != CLASS_NAMES[label_index]:
            raise BaselineValidationError(
                f"patients[{index}] label and label_index disagree"
            )
        if case_id in case_ids:
            raise BaselineValidationError(f"Duplicate case ID {case_id!r}")
        if patient_key in patient_keys:
            raise BaselineValidationError(
                f"Duplicate private patient key at patients[{index}]"
            )
        case_ids.add(case_id)
        patient_keys.add(patient_key)
        patients.append(
            {
                "case_id": case_id,
                "patient_key": patient_key,
                "label": label,
                "label_index": label_index,
            }
        )

    raw_folds = payload.get("folds")
    if not isinstance(raw_folds, list) or len(raw_folds) != len(EXPECTED_FOLDS):
        raise BaselineValidationError("Manifest must contain exactly five folds")
    folds: list[dict[str, Any]] = []
    test_counts: Counter[str] = Counter()
    for expected_fold, raw in zip(EXPECTED_FOLDS, raw_folds, strict=True):
        if not isinstance(raw, dict) or raw.get("fold") != expected_fold:
            raise BaselineValidationError(
                f"Expected fold {expected_fold} at folds[{expected_fold}]"
            )
        train = _require_string_list(
            raw.get("train_case_ids"), f"fold {expected_fold} train_case_ids"
        )
        validation = _require_string_list(
            raw.get("val_case_ids"), f"fold {expected_fold} val_case_ids"
        )
        test = _require_string_list(
            raw.get("test_case_ids"), f"fold {expected_fold} test_case_ids"
        )
        train_set, validation_set, test_set = set(train), set(validation), set(test)
        if (
            train_set & validation_set
            or train_set & test_set
            or validation_set & test_set
        ):
            raise BaselineValidationError(
                f"Fold {expected_fold} train/validation/test sets overlap"
            )
        if train_set | validation_set | test_set != case_ids:
            missing = sorted(
                case_ids - (train_set | validation_set | test_set)
            )
            unknown = sorted(
                (train_set | validation_set | test_set) - case_ids
            )
            raise BaselineValidationError(
                f"Fold {expected_fold} does not partition the cohort; "
                f"missing={missing}, unknown={unknown}"
            )
        test_counts.update(test)
        folds.append(
            {
                "fold": expected_fold,
                "train_case_ids": train,
                "val_case_ids": validation,
                "test_case_ids": test,
            }
        )
    if set(test_counts) != case_ids or any(count != 1 for count in test_counts.values()):
        raise BaselineValidationError(
            "Across the five folds every patient must be held out exactly once"
        )
    return patients, folds


def _numeric_or_missing(value: Any, description: str) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineValidationError(f"{description} must be numeric or null")
    number = float(value)
    return number if math.isfinite(number) else math.nan


def load_patient_suv(
    patient_key: str,
    case_id: str,
    suv_dir: Path,
) -> tuple[dict[str, dict[str, dict[str, float]]], list[dict[str, str]]]:
    by_tracer: dict[str, dict[str, dict[str, float]]] = {}
    source_files: list[dict[str, str]] = []
    for tracer in TRACERS:
        path = suv_dir / patient_key / f"{tracer.lower()}_suv_statistics.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = load_json(path)
        if not isinstance(payload, dict):
            raise BaselineValidationError(f"{path} must contain a JSON object")
        if payload.get("patient") != patient_key:
            raise BaselineValidationError(f"Patient identifier mismatch in {path}")
        if str(payload.get("tracer", "")).upper() != tracer:
            raise BaselineValidationError(f"Tracer identifier mismatch in {path}")
        regions = payload.get("regions")
        if not isinstance(regions, list) or not regions:
            raise BaselineValidationError(f"{path}.regions must be a non-empty list")
        roi_map: dict[str, dict[str, float]] = {}
        for region_index, region in enumerate(regions):
            if not isinstance(region, dict):
                raise BaselineValidationError(
                    f"{path}.regions[{region_index}] must be an object"
                )
            roi_name = _require_string(
                region.get("roi_name"),
                f"{path}.regions[{region_index}].roi_name",
            )
            if roi_name != roi_name.strip():
                raise BaselineValidationError(
                    f"{path}.regions[{region_index}].roi_name is not trimmed"
                )
            if roi_name in roi_map:
                raise BaselineValidationError(f"Duplicate ROI {roi_name!r} in {path}")
            roi_map[roi_name] = {
                statistic: _numeric_or_missing(
                    region.get(statistic),
                    f"{path}:{roi_name}.{statistic}",
                )
                for statistic in SUV_STATISTICS
            }
        by_tracer[tracer] = roi_map
        source_files.append(
            {
                "case_id": case_id,
                "tracer": tracer,
                "sha256": sha256_file(path),
            }
        )
    return by_tracer, source_files


def build_feature_matrix(
    patients: Sequence[Mapping[str, Any]],
    suv_dir: Path,
) -> tuple[np.ndarray, list[str], list[dict[str, str]], dict[str, Any]]:
    patient_values: list[dict[str, dict[str, dict[str, float]]]] = []
    source_files: list[dict[str, str]] = []
    roi_names: set[str] = set()
    for record in patients:
        values, files = load_patient_suv(
            str(record["patient_key"]),
            str(record["case_id"]),
            suv_dir,
        )
        patient_values.append(values)
        source_files.extend(files)
        for tracer in TRACERS:
            roi_names.update(values[tracer])
    if not roi_names:
        raise BaselineValidationError("No whole-organ ROI names were found")

    sorted_rois = sorted(roi_names)
    feature_keys = [
        (roi_name, tracer, statistic)
        for roi_name in sorted_rois
        for tracer in TRACERS
        for statistic in SUV_STATISTICS
    ]
    feature_names = [
        f"{roi_name}|{tracer}|{statistic}"
        for roi_name, tracer, statistic in feature_keys
    ]
    matrix = np.full(
        (len(patients), len(feature_keys)),
        np.nan,
        dtype=np.float64,
    )
    for patient_index, values in enumerate(patient_values):
        for feature_index, (roi_name, tracer, statistic) in enumerate(feature_keys):
            roi = values[tracer].get(roi_name)
            if roi is not None:
                matrix[patient_index, feature_index] = roi[statistic]
    finite_counts = np.isfinite(matrix).sum(axis=0)
    if np.any(finite_counts == 0):
        empty = [
            feature_names[index]
            for index in np.flatnonzero(finite_counts == 0).tolist()
        ]
        raise BaselineValidationError(
            f"Feature schema contains globally empty columns: {empty}"
        )
    if np.any(np.isinf(matrix)):
        raise BaselineValidationError("Feature matrix contains infinity")

    feature_audit = {
        "roi_count": len(sorted_rois),
        "feature_count": len(feature_names),
        "ordering": "sorted ROI name, then FDG/PSMA, then suv_mean/suv_max",
        "roi_names": sorted_rois,
        "missing_values": int(np.isnan(matrix).sum()),
        "features_with_missing_values": int(np.count_nonzero(finite_counts < len(patients))),
        "minimum_observed_patients_per_feature": int(finite_counts.min()),
        "maximum_observed_patients_per_feature": int(finite_counts.max()),
    }
    return matrix, feature_names, source_files, feature_audit


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    labels = np.arange(len(CLASS_NAMES), dtype=np.int64)
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "n": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _fit_preprocessor(
    fit_matrix: np.ndarray,
    *transform_matrices: np.ndarray,
) -> tuple[SimpleImputer, StandardScaler, list[np.ndarray]]:
    # ``keep_empty_features`` guarantees a fixed-dimensional transform even if
    # a very rare ROI is absent from this fold's fitting patients.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputed_fit = imputer.fit_transform(fit_matrix)
    scaler = StandardScaler()
    scaled_fit = scaler.fit_transform(imputed_fit)
    transformed = [scaled_fit]
    for matrix in transform_matrices:
        transformed.append(scaler.transform(imputer.transform(matrix)))
    if any(not np.isfinite(matrix).all() for matrix in transformed):
        raise BaselineValidationError(
            "Imputation/scaling produced a non-finite feature value"
        )
    return imputer, scaler, transformed


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    c_value: float,
    seed: int,
) -> LogisticRegression:
    model = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=MAX_ITER,
        tol=1e-6,
        random_state=seed,
    )
    model.fit(features, labels)
    expected_classes = np.arange(len(CLASS_NAMES))
    if not np.array_equal(model.classes_, expected_classes):
        raise BaselineValidationError(
            f"Fitted model class order {model.classes_.tolist()} is not "
            f"{expected_classes.tolist()}"
        )
    if np.any(model.n_iter_ >= MAX_ITER):
        raise RuntimeError(
            f"Logistic regression did not converge within {MAX_ITER} iterations"
        )
    return model


def _selection_key(candidate: Mapping[str, Any]) -> tuple[float, float, float, float]:
    metrics = candidate["validation"]
    return (
        float(metrics["macro_f1"]),
        float(metrics["balanced_accuracy"]),
        float(metrics["accuracy"]),
        -float(candidate["C"]),
    )


def _case_indices(
    case_ids: Sequence[str],
    index_by_case: Mapping[str, int],
) -> np.ndarray:
    return np.asarray([index_by_case[case_id] for case_id in case_ids], dtype=np.int64)


def run_fold(
    *,
    fold: Mapping[str, Any],
    features: np.ndarray,
    labels: np.ndarray,
    patients: Sequence[Mapping[str, Any]],
    index_by_case: Mapping[str, int],
    feature_schema_sha256: str,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_number = int(fold["fold"])
    train_ids = list(fold["train_case_ids"])
    validation_ids = list(fold["val_case_ids"])
    test_ids = list(fold["test_case_ids"])
    train_set, validation_set, test_set = (
        set(train_ids),
        set(validation_ids),
        set(test_ids),
    )
    if (
        train_set & validation_set
        or train_set & test_set
        or validation_set & test_set
    ):
        raise BaselineValidationError(f"Fold {fold_number} contains split leakage")

    train_indices = _case_indices(train_ids, index_by_case)
    validation_indices = _case_indices(validation_ids, index_by_case)
    test_indices = _case_indices(test_ids, index_by_case)
    x_train = features[train_indices]
    y_train = labels[train_indices]
    x_validation = features[validation_indices]
    y_validation = labels[validation_indices]
    x_test = features[test_indices]
    y_test = labels[test_indices]
    if set(y_train.tolist()) != set(range(len(CLASS_NAMES))):
        raise BaselineValidationError(
            f"Fold {fold_number} training set does not contain all classes"
        )

    train_imputer, train_scaler, transformed = _fit_preprocessor(
        x_train,
        x_validation,
    )
    x_train_scaled, x_validation_scaled = transformed
    candidates: list[dict[str, Any]] = []
    fold_seed = BASE_SEED + fold_number
    for c_value in C_GRID:
        model = _fit_logistic(
            x_train_scaled,
            y_train,
            c_value=c_value,
            seed=fold_seed,
        )
        prediction = model.predict(x_validation_scaled)
        candidates.append(
            {
                "C": c_value,
                "validation": classification_metrics(y_validation, prediction),
                "iterations": [int(value) for value in model.n_iter_.tolist()],
            }
        )
    selected = max(candidates, key=_selection_key)
    selected_c = float(selected["C"])

    # C is now frozen.  Refit every learned component on train+validation, then
    # transform the still-unseen test matrix exactly once.
    refit_indices = np.concatenate((train_indices, validation_indices))
    x_refit = features[refit_indices]
    y_refit = labels[refit_indices]
    refit_imputer, refit_scaler, refit_transformed = _fit_preprocessor(
        x_refit,
        x_test,
    )
    x_refit_scaled, x_test_scaled = refit_transformed
    final_model = _fit_logistic(
        x_refit_scaled,
        y_refit,
        c_value=selected_c,
        seed=fold_seed,
    )
    probabilities = final_model.predict_proba(x_test_scaled)
    if probabilities.shape != (len(test_ids), len(CLASS_NAMES)):
        raise BaselineValidationError(
            f"Fold {fold_number} returned an invalid probability shape"
        )
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-7,
        rtol=0.0,
    ):
        raise BaselineValidationError(
            f"Fold {fold_number} returned invalid class probabilities"
        )
    predicted = probabilities.argmax(axis=1)
    test_metrics = classification_metrics(y_test, predicted)

    rows: list[dict[str, Any]] = []
    for local_index, case_id in enumerate(test_ids):
        source = patients[index_by_case[case_id]]
        predicted_index = int(predicted[local_index])
        true_index = int(y_test[local_index])
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "completed",
                "baseline_id": BASELINE_ID,
                "case_id": case_id,
                "fold": fold_number,
                "predicted_index": predicted_index,
                "predicted_class": CLASS_NAMES[predicted_index],
                "true_index": true_index,
                "true_label": str(source["label"]),
                "correct": predicted_index == true_index,
                "probabilities": {
                    class_name: float(probabilities[local_index, class_index])
                    for class_index, class_name in enumerate(CLASS_NAMES)
                },
                "selected_C": selected_c,
                "feature_schema_sha256": feature_schema_sha256,
            }
        )

    # Hashes bind each fitted preprocessing stage to the explicit case IDs it
    # was allowed to see; the test IDs are separately recorded and disjoint.
    train_preprocessing_fingerprint = sha256_json(
        {
            "fit_case_ids": train_ids,
            "imputer_statistics": [
                None if not math.isfinite(float(value)) else float(value)
                for value in train_imputer.statistics_
            ],
            "scaler_mean": [float(value) for value in train_scaler.mean_],
            "scaler_scale": [float(value) for value in train_scaler.scale_],
        }
    )
    refit_preprocessing_fingerprint = sha256_json(
        {
            "fit_case_ids": train_ids + validation_ids,
            "imputer_statistics": [
                None if not math.isfinite(float(value)) else float(value)
                for value in refit_imputer.statistics_
            ],
            "scaler_mean": [float(value) for value in refit_scaler.mean_],
            "scaler_scale": [float(value) for value in refit_scaler.scale_],
        }
    )
    fold_audit = {
        "fold": fold_number,
        "split_counts": {
            "train": len(train_ids),
            "validation": len(validation_ids),
            "test": len(test_ids),
        },
        "train_case_ids": train_ids,
        "validation_case_ids": validation_ids,
        "test_case_ids": test_ids,
        "pairwise_disjoint": True,
        "selection_preprocessing_fit": "train only",
        "selection_model_fit": "train only",
        "hyperparameter_evaluation": "validation only",
        "final_preprocessing_and_model_fit": "train + validation only",
        "test_used_for_fitting_or_selection": False,
        "train_preprocessing_fingerprint": train_preprocessing_fingerprint,
        "refit_preprocessing_fingerprint": refit_preprocessing_fingerprint,
    }
    fold_result = {
        "schema_version": SCHEMA_VERSION,
        "baseline_id": BASELINE_ID,
        "fold": fold_number,
        "selected_C": selected_c,
        "selection_metric": SELECTION_METRIC,
        "selection_tie_break": (
            "balanced accuracy, then accuracy, then smaller C"
        ),
        "candidates": candidates,
        "final_model_iterations": [
            int(value) for value in final_model.n_iter_.tolist()
        ],
        "test": test_metrics,
        "leakage_audit": fold_audit,
    }
    fold_dir = output_dir / "folds" / f"fold_{fold_number}"
    atomic_write_json(fold_dir / "split.json", fold_audit)
    atomic_write_json(fold_dir / "metrics.json", fold_result)
    atomic_write_text(fold_dir / "test_predictions.jsonl", jsonl(rows))
    return rows, fold_result


def validate_oof(
    rows: Sequence[Mapping[str, Any]],
    patients: Sequence[Mapping[str, Any]],
    folds: Sequence[Mapping[str, Any]],
) -> None:
    expected = {str(record["case_id"]) for record in patients}
    observed = [str(row["case_id"]) for row in rows]
    counts = Counter(observed)
    if set(counts) != expected or any(count != 1 for count in counts.values()):
        raise BaselineValidationError(
            "OOF predictions do not contain every patient exactly once"
        )
    expected_fold_by_case = {
        case_id: int(fold["fold"])
        for fold in folds
        for case_id in fold["test_case_ids"]
    }
    for row in rows:
        case_id = str(row["case_id"])
        if int(row["fold"]) != expected_fold_by_case[case_id]:
            raise BaselineValidationError(
                f"OOF fold mismatch for case {case_id}"
            )


def write_oof_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    patients: Sequence[Mapping[str, Any]],
    fold_results: Sequence[Mapping[str, Any]],
    config_fingerprint: str,
    feature_audit: Mapping[str, Any],
) -> dict[str, Any]:
    row_by_case = {str(row["case_id"]): row for row in rows}
    ordered = [row_by_case[str(record["case_id"])] for record in patients]
    y_true = np.asarray([int(row["true_index"]) for row in ordered])
    y_pred = np.asarray([int(row["predicted_index"]) for row in ordered])
    pooled = classification_metrics(y_true, y_pred)

    csv_fields = [
        "case_id",
        "fold",
        "true_index",
        "true_label",
        "predicted_index",
        "predicted_class",
        "correct",
        "selected_C",
        *[f"probability_{name}" for name in CLASS_NAMES],
    ]
    csv_rows = [
        {
            "case_id": row["case_id"],
            "fold": row["fold"],
            "true_index": row["true_index"],
            "true_label": row["true_label"],
            "predicted_index": row["predicted_index"],
            "predicted_class": row["predicted_class"],
            "correct": int(bool(row["correct"])),
            "selected_C": row["selected_C"],
            **{
                f"probability_{name}": row["probabilities"][name]
                for name in CLASS_NAMES
            },
        }
        for row in ordered
    ]
    atomic_write_csv(output_dir / "oof_predictions.csv", csv_fields, csv_rows)
    atomic_write_text(output_dir / "oof_predictions.jsonl", jsonl(ordered))

    summary = {
        "schema_version": SCHEMA_VERSION,
        "baseline_id": BASELINE_ID,
        "analysis": "fixed five-fold out-of-fold classification",
        "completed_at_utc": utc_now(),
        "cohort_size": len(patients),
        "oof_predictions": len(ordered),
        "coverage": len(ordered) / len(patients),
        "classes": list(CLASS_NAMES),
        "config_fingerprint": config_fingerprint,
        "feature_audit": dict(feature_audit),
        "pooled_oof": pooled,
        # Top-level aliases keep the primary endpoints easy to consume.
        "accuracy": pooled["accuracy"],
        "macro_f1": pooled["macro_f1"],
        "balanced_accuracy": pooled["balanced_accuracy"],
        "per_class": pooled["per_class"],
        "folds": [
            {
                "fold": result["fold"],
                "selected_C": result["selected_C"],
                "test": result["test"],
            }
            for result in fold_results
        ],
        "leakage_audit": {
            "frozen_split_source": "DL-only manifest",
            "outer_test_sets_pairwise_disjoint": True,
            "outer_test_sets_cover_cohort_exactly_once": True,
            "test_used_for_imputation": False,
            "test_used_for_scaling": False,
            "test_used_for_hyperparameter_selection": False,
            "test_used_for_model_fitting": False,
        },
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if not args.suv_dir.is_dir():
        raise FileNotFoundError(args.suv_dir)
    manifest = load_json(args.manifest)
    patients, folds = validate_manifest(manifest)
    features, feature_names, source_files, feature_audit = build_feature_matrix(
        patients,
        args.suv_dir,
    )
    labels = np.asarray(
        [int(record["label_index"]) for record in patients],
        dtype=np.int64,
    )
    index_by_case = {
        str(record["case_id"]): index for index, record in enumerate(patients)
    }
    feature_schema = {
        "tracers": list(TRACERS),
        "statistics": list(SUV_STATISTICS),
        "feature_names": feature_names,
        **feature_audit,
    }
    feature_schema_sha256 = sha256_json(feature_schema)
    matrix_sha256 = sha256_bytes(
        np.ascontiguousarray(features, dtype="<f8").tobytes(order="C")
    )

    all_missing_train_features: dict[str, int] = {}
    for fold in folds:
        train_indices = _case_indices(fold["train_case_ids"], index_by_case)
        all_missing_train_features[str(fold["fold"])] = int(
            np.count_nonzero(np.isfinite(features[train_indices]).sum(axis=0) == 0)
        )
    dry_run_payload = {
        "status": "dry_run_ok",
        "baseline_id": BASELINE_ID,
        "patients": len(patients),
        "classes": list(CLASS_NAMES),
        "folds": [
            {
                "fold": fold["fold"],
                "train": len(fold["train_case_ids"]),
                "validation": len(fold["val_case_ids"]),
                "test": len(fold["test_case_ids"]),
            }
            for fold in folds
        ],
        "feature_schema_sha256": feature_schema_sha256,
        "feature_matrix_sha256": matrix_sha256,
        "feature_audit": feature_audit,
        "all_missing_train_features_by_fold": all_missing_train_features,
        "c_grid": list(C_GRID),
        "selection_metric": SELECTION_METRIC,
        "output_dir": str(args.output_dir.resolve()),
    }
    if args.dry_run:
        print(json.dumps(dry_run_payload, ensure_ascii=False, indent=2))
        return 0

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty; use --overwrite or a new --output-dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "schema_version": SCHEMA_VERSION,
        "baseline_id": BASELINE_ID,
        "created_at_utc": utc_now(),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256_file(Path(__file__)),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_fingerprint": manifest.get("manifest_fingerprint"),
        "suv_dir": str(args.suv_dir.resolve()),
        "suv_source_files": source_files,
        "feature_schema_sha256": feature_schema_sha256,
        "feature_matrix_sha256": matrix_sha256,
        "feature_audit": feature_audit,
        "classes": list(CLASS_NAMES),
        "patients": len(patients),
        "folds": len(folds),
        "protocol": {
            "feature_order": (
                "sorted ROI name, then FDG/PSMA, then suv_mean/suv_max"
            ),
            "feature_schema_scope": (
                "unlabelled union of whole-organ ROI names in the fixed cohort"
            ),
            "imputer": "median; fitted on train during selection",
            "scaler": "StandardScaler; fitted on train during selection",
            "classifier": "multinomial LogisticRegression, lbfgs",
            "class_weight": "balanced",
            "c_grid": list(C_GRID),
            "selection_metric": SELECTION_METRIC,
            "selection_tie_break": (
                "balanced accuracy, then accuracy, then smaller C"
            ),
            "post_selection_refit": (
                "new imputer, scaler, and classifier fitted on train+validation"
            ),
            "max_iter": MAX_ITER,
            "tolerance": 1e-6,
            "base_seed": BASE_SEED,
        },
        "dependencies": dependency_versions(),
    }
    config_fingerprint = sha256_json(
        {key: value for key, value in config.items() if key != "created_at_utc"}
    )
    config["config_fingerprint"] = config_fingerprint
    atomic_write_json(output_dir / "config.json", config)
    atomic_write_json(output_dir / "feature_schema.json", feature_schema)

    fold_results: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    for fold in folds:
        rows, fold_result = run_fold(
            fold=fold,
            features=features,
            labels=labels,
            patients=patients,
            index_by_case=index_by_case,
            feature_schema_sha256=feature_schema_sha256,
            output_dir=output_dir,
        )
        oof_rows.extend(rows)
        fold_results.append(fold_result)
        print(
            f"fold={fold['fold']} C={fold_result['selected_C']:g} "
            f"test_accuracy={fold_result['test']['accuracy']:.4f} "
            f"test_macro_f1={fold_result['test']['macro_f1']:.4f}",
            flush=True,
        )

    validate_oof(oof_rows, patients, folds)
    summary = write_oof_outputs(
        output_dir,
        oof_rows,
        patients,
        fold_results,
        config_fingerprint,
        feature_audit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Run directory: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
