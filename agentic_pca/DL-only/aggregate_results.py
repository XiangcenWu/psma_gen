#!/usr/bin/env python3
"""Aggregate five-fold out-of-fold predictions with strict provenance checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SCHEMA_VERSION = "1.0"
EXPECTED_FOLDS = tuple(range(5))
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_727
CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class AggregationError(RuntimeError):
    """Raised when inputs cannot support a valid OOF analysis."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AggregationError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AggregationError(f"Cannot read {path}: {exc}") from exc
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AggregationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AggregationError(f"{path} must contain a JSON object.")
    return value, raw


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise AggregationError(f"Cannot read {path}: {exc}") from exc

    records: list[dict[str, Any]] = []
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            except json.JSONDecodeError as exc:
                raise AggregationError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise AggregationError(
                    f"{path}:{line_number} must contain a JSON object."
                )
            value["_source_line"] = line_number
            records.append(value)
    if not records:
        raise AggregationError(f"{path} contains no prediction records.")
    return records


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AggregationError(f"{context} must be a non-empty string.")
    return value


def _require_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AggregationError(f"{context} must be an integer.")
    return value


def _validate_case_id(value: Any, context: str) -> str:
    case_id = _require_string(value, context)
    if not CASE_ID_RE.fullmatch(case_id):
        raise AggregationError(
            f"{context} must be an opaque filesystem-safe identifier containing "
            "only ASCII letters, digits, '.', '_' or '-'."
        )
    return case_id


def _validate_case_id_list(
    value: Any,
    context: str,
    known_case_ids: set[str],
) -> list[str]:
    if not isinstance(value, list):
        raise AggregationError(f"{context} must be a list.")
    case_ids = [
        _validate_case_id(case_id, f"{context}[{index}]")
        for index, case_id in enumerate(value)
    ]
    duplicates = sorted(
        case_id for case_id, count in Counter(case_ids).items() if count != 1
    )
    if duplicates:
        raise AggregationError(
            f"{context} contains duplicate case IDs: {', '.join(duplicates)}"
        )
    unknown = sorted(set(case_ids) - known_case_ids)
    if unknown:
        raise AggregationError(
            f"{context} contains case IDs absent from manifest patients: "
            f"{', '.join(unknown)}"
        )
    return case_ids


def _validate_manifest(
    manifest: dict[str, Any],
) -> tuple[
    list[str],
    str,
    list[str],
    dict[str, dict[str, Any]],
    dict[int, set[str]],
]:
    classes_raw = manifest.get("classes")
    if not isinstance(classes_raw, list) or len(classes_raw) != 4:
        raise AggregationError("manifest.classes must contain exactly four classes.")
    classes = [
        _require_string(value, f"manifest.classes[{index}]")
        for index, value in enumerate(classes_raw)
    ]
    if len(set(classes)) != len(classes):
        raise AggregationError("manifest.classes must not contain duplicates.")

    preprocessing = manifest.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise AggregationError("manifest.preprocessing must be an object.")
    preprocess_fingerprint = _require_string(
        preprocessing.get("fingerprint"),
        "manifest.preprocessing.fingerprint",
    )

    patients_raw = manifest.get("patients")
    if not isinstance(patients_raw, list) or not patients_raw:
        raise AggregationError("manifest.patients must be a non-empty list.")

    patient_order: list[str] = []
    patient_by_id: dict[str, dict[str, Any]] = {}
    for index, patient in enumerate(patients_raw):
        context = f"manifest.patients[{index}]"
        if not isinstance(patient, dict):
            raise AggregationError(f"{context} must be an object.")
        case_id = _validate_case_id(patient.get("case_id"), f"{context}.case_id")
        if case_id in patient_by_id:
            raise AggregationError(
                f"manifest.patients contains duplicate case ID {case_id!r}."
            )
        label_index = _require_int(
            patient.get("label_index"), f"{context}.label_index"
        )
        if not 0 <= label_index < len(classes):
            raise AggregationError(f"{context}.label_index is outside class range.")
        label = _require_string(patient.get("label"), f"{context}.label")
        if label != classes[label_index]:
            raise AggregationError(
                f"{context} label/index disagree with manifest.classes."
            )
        patient_order.append(case_id)
        patient_by_id[case_id] = {
            "case_id": case_id,
            "label_index": label_index,
            "label": label,
        }

    known_case_ids = set(patient_by_id)
    folds_raw = manifest.get("folds")
    if not isinstance(folds_raw, list):
        raise AggregationError("manifest.folds must be a list.")
    fold_by_id: dict[int, dict[str, Any]] = {}
    expected_test_by_fold: dict[int, set[str]] = {}

    for index, fold_record in enumerate(folds_raw):
        context = f"manifest.folds[{index}]"
        if not isinstance(fold_record, dict):
            raise AggregationError(f"{context} must be an object.")
        fold = _require_int(fold_record.get("fold"), f"{context}.fold")
        if fold in fold_by_id:
            raise AggregationError(f"manifest.folds contains duplicate fold {fold}.")

        train = _validate_case_id_list(
            fold_record.get("train_case_ids"),
            f"{context}.train_case_ids",
            known_case_ids,
        )
        validation = _validate_case_id_list(
            fold_record.get("val_case_ids"),
            f"{context}.val_case_ids",
            known_case_ids,
        )
        test = _validate_case_id_list(
            fold_record.get("test_case_ids"),
            f"{context}.test_case_ids",
            known_case_ids,
        )
        split_sets = [set(train), set(validation), set(test)]
        if (
            split_sets[0] & split_sets[1]
            or split_sets[0] & split_sets[2]
            or split_sets[1] & split_sets[2]
        ):
            raise AggregationError(
                f"{context} train, validation and test partitions overlap."
            )
        partition_union = set().union(*split_sets)
        if partition_union != known_case_ids:
            missing = sorted(known_case_ids - partition_union)
            raise AggregationError(
                f"{context} does not partition every manifest patient; "
                f"missing: {', '.join(missing)}"
            )
        fold_by_id[fold] = fold_record
        expected_test_by_fold[fold] = split_sets[2]

    if set(fold_by_id) != set(EXPECTED_FOLDS):
        raise AggregationError(
            "manifest.folds must contain exactly fold IDs 0, 1, 2, 3 and 4."
        )

    test_counts: Counter[str] = Counter()
    for case_ids in expected_test_by_fold.values():
        test_counts.update(case_ids)
    missing_test = sorted(known_case_ids - set(test_counts))
    repeated_test = sorted(
        case_id for case_id, count in test_counts.items() if count != 1
    )
    if missing_test or repeated_test:
        details: list[str] = []
        if missing_test:
            details.append(f"never tested: {', '.join(missing_test)}")
        if repeated_test:
            details.append(f"tested more than once: {', '.join(repeated_test)}")
        raise AggregationError(
            "manifest five-fold test sets are not an exact OOF partition ("
            + "; ".join(details)
            + ")."
        )

    return (
        classes,
        preprocess_fingerprint,
        patient_order,
        patient_by_id,
        expected_test_by_fold,
    )


def _validate_probability(
    value: Any,
    context: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AggregationError(f"{context} must be numeric.")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise AggregationError(f"{context} must be finite and in [0, 1].")
    return probability


def _validate_prediction(
    raw: dict[str, Any],
    *,
    source: Path,
    expected_fold: int,
    classes: Sequence[str],
    patient_by_id: dict[str, dict[str, Any]],
    preprocess_fingerprint: str,
) -> dict[str, Any]:
    line_number = raw.get("_source_line", "?")
    context = f"{source}:{line_number}"
    if raw.get("status") != "completed":
        raise AggregationError(f"{context} status must be 'completed'.")

    case_id = _validate_case_id(raw.get("case_id"), f"{context}.case_id")
    if case_id not in patient_by_id:
        raise AggregationError(
            f"{context} contains unknown case ID {case_id!r}."
        )
    fold = _require_int(raw.get("fold"), f"{context}.fold")
    if fold != expected_fold:
        raise AggregationError(
            f"{context} declares fold {fold}, expected {expected_fold}."
        )

    predicted_index = _require_int(
        raw.get("predicted_index"), f"{context}.predicted_index"
    )
    if not 0 <= predicted_index < len(classes):
        raise AggregationError(f"{context}.predicted_index is outside class range.")
    predicted_class = _require_string(
        raw.get("predicted_class"), f"{context}.predicted_class"
    )
    if predicted_class != classes[predicted_index]:
        raise AggregationError(
            f"{context} predicted class/index disagree with manifest.classes."
        )

    patient = patient_by_id[case_id]
    true_index = _require_int(raw.get("true_index"), f"{context}.true_index")
    true_label = _require_string(raw.get("true_label"), f"{context}.true_label")
    if (
        true_index != patient["label_index"]
        or true_label != patient["label"]
        or true_label != classes[true_index]
    ):
        raise AggregationError(
            f"{context} ground truth disagrees with manifest."
        )

    probabilities_raw = raw.get("probabilities")
    if not isinstance(probabilities_raw, dict):
        raise AggregationError(f"{context}.probabilities must be an object.")
    if set(probabilities_raw) != set(classes):
        missing = sorted(set(classes) - set(probabilities_raw))
        extra = sorted(set(probabilities_raw) - set(classes))
        raise AggregationError(
            f"{context}.probabilities must have exactly the four manifest classes; "
            f"missing={missing}, extra={extra}."
        )
    probabilities = {
        class_name: _validate_probability(
            probabilities_raw[class_name],
            f"{context}.probabilities[{class_name!r}]",
        )
        for class_name in classes
    }
    probability_sum = sum(probabilities.values())
    if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise AggregationError(
            f"{context} probabilities sum to {probability_sum:.12g}, not 1."
        )
    if probabilities[predicted_class] < max(probabilities.values()) - 1e-10:
        raise AggregationError(
            f"{context} predicted_class is not an argmax of probabilities."
        )

    correct = raw.get("correct")
    if not isinstance(correct, bool):
        raise AggregationError(f"{context}.correct must be boolean.")
    expected_correct = predicted_index == true_index
    if correct != expected_correct:
        raise AggregationError(
            f"{context}.correct disagrees with predicted and true labels."
        )

    checkpoint_sha256 = _require_string(
        raw.get("checkpoint_sha256"), f"{context}.checkpoint_sha256"
    ).lower()
    if not SHA256_RE.fullmatch(checkpoint_sha256):
        raise AggregationError(
            f"{context}.checkpoint_sha256 must be a 64-character hexadecimal SHA256."
        )
    prediction_fingerprint = _require_string(
        raw.get("preprocess_fingerprint"),
        f"{context}.preprocess_fingerprint",
    )
    if prediction_fingerprint != preprocess_fingerprint:
        raise AggregationError(
            f"{context} preprocess_fingerprint disagrees with manifest."
        )
    protocol_fingerprint = _require_string(
        raw.get("protocol_fingerprint"),
        f"{context}.protocol_fingerprint",
    ).lower()
    if not SHA256_RE.fullmatch(protocol_fingerprint):
        raise AggregationError(
            f"{context}.protocol_fingerprint must be a 64-character "
            "hexadecimal SHA256."
        )

    return {
        "case_id": case_id,
        "fold": fold,
        "predicted_index": predicted_index,
        "predicted_class": predicted_class,
        "probabilities": probabilities,
        "true_index": true_index,
        "true_label": true_label,
        "correct": correct,
        "checkpoint_sha256": checkpoint_sha256,
        "preprocess_fingerprint": prediction_fingerprint,
        "protocol_fingerprint": protocol_fingerprint,
    }


def _load_and_validate_predictions(
    output_dir: Path,
    classes: Sequence[str],
    preprocess_fingerprint: str,
    patient_order: Sequence[str],
    patient_by_id: dict[str, dict[str, Any]],
    expected_test_by_fold: dict[int, set[str]],
) -> tuple[list[dict[str, Any]], dict[int, str], str]:
    predictions_by_id: dict[str, dict[str, Any]] = {}
    checkpoint_by_fold: dict[int, str] = {}
    protocol_fingerprints: set[str] = set()

    for fold in EXPECTED_FOLDS:
        path = output_dir / "folds" / f"fold_{fold}" / "test_predictions.jsonl"
        raw_predictions = _read_jsonl(path)
        fold_case_ids: set[str] = set()
        fold_checkpoints: set[str] = set()
        fold_protocols: set[str] = set()

        for raw in raw_predictions:
            prediction = _validate_prediction(
                raw,
                source=path,
                expected_fold=fold,
                classes=classes,
                patient_by_id=patient_by_id,
                preprocess_fingerprint=preprocess_fingerprint,
            )
            case_id = prediction["case_id"]
            if case_id in predictions_by_id:
                previous_fold = predictions_by_id[case_id]["fold"]
                raise AggregationError(
                    f"Case ID {case_id!r} has multiple OOF predictions "
                    f"(folds {previous_fold} and {fold})."
                )
            predictions_by_id[case_id] = prediction
            fold_case_ids.add(case_id)
            fold_checkpoints.add(prediction["checkpoint_sha256"])
            fold_protocols.add(prediction["protocol_fingerprint"])

        expected = expected_test_by_fold[fold]
        if fold_case_ids != expected:
            missing = sorted(expected - fold_case_ids)
            unexpected = sorted(fold_case_ids - expected)
            raise AggregationError(
                f"Fold {fold} predictions disagree with manifest test set; "
                f"missing={missing}, unexpected={unexpected}."
            )
        if len(fold_checkpoints) != 1:
            raise AggregationError(
                f"Fold {fold} predictions reference {len(fold_checkpoints)} "
                "different checkpoints; expected exactly one."
            )
        if len(fold_protocols) != 1:
            raise AggregationError(
                f"Fold {fold} predictions reference {len(fold_protocols)} "
                "different training protocols; expected exactly one."
            )
        checkpoint_by_fold[fold] = next(iter(fold_checkpoints))
        protocol_fingerprints.update(fold_protocols)

    manifest_case_ids = set(patient_order)
    prediction_case_ids = set(predictions_by_id)
    missing = sorted(manifest_case_ids - prediction_case_ids)
    unknown = sorted(prediction_case_ids - manifest_case_ids)
    if missing or unknown or len(predictions_by_id) != len(patient_order):
        raise AggregationError(
            "OOF predictions must contain every manifest case exactly once; "
            f"missing={missing}, unknown={unknown}."
        )

    if len(protocol_fingerprints) != 1:
        raise AggregationError(
            "The five folds were not trained with one identical protocol; "
            f"found {sorted(protocol_fingerprints)}."
        )
    predictions = [predictions_by_id[case_id] for case_id in patient_order]
    return predictions, checkpoint_by_fold, next(iter(protocol_fingerprints))


def _add_warning(warnings: list[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


def _binary_ranking_metrics(
    y_binary: np.ndarray,
    scores: np.ndarray,
) -> tuple[float | None, float | None]:
    positive_count = int(y_binary.sum())
    negative_count = int(y_binary.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None, None

    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_truth = y_binary[order].astype(np.int64, copy=False)
    distinct_ends = np.flatnonzero(np.diff(sorted_scores))
    threshold_ends = np.concatenate(
        (distinct_ends, np.asarray([sorted_truth.size - 1], dtype=np.int64))
    )
    true_positives = np.cumsum(sorted_truth, dtype=np.int64)[threshold_ends]
    false_positives = 1 + threshold_ends - true_positives

    true_positive_rate = np.concatenate(
        ([0.0], true_positives / positive_count)
    )
    false_positive_rate = np.concatenate(
        ([0.0], false_positives / negative_count)
    )
    auroc = float(
        np.sum(
            np.diff(false_positive_rate)
            * (true_positive_rate[1:] + true_positive_rate[:-1])
            * 0.5
        )
    )

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / positive_count
    auprc = float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))
    return auroc, auprc


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: Sequence[str],
    *,
    warning_context: str | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    class_count = len(classes)
    confusion = np.bincount(
        y_true * class_count + y_pred,
        minlength=class_count * class_count,
    ).reshape(class_count, class_count)
    support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    true_positives = np.diag(confusion).astype(np.float64)
    precision = np.divide(
        true_positives,
        predicted_support,
        out=np.zeros(class_count, dtype=np.float64),
        where=predicted_support != 0,
    )
    recall = np.divide(
        true_positives,
        support,
        out=np.zeros(class_count, dtype=np.float64),
        where=support != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(class_count, dtype=np.float64),
        where=(precision + recall) != 0,
    )
    n = int(y_true.size)
    accuracy = float(true_positives.sum() / n)
    supported = support > 0
    balanced_accuracy = (
        float(recall[supported].mean()) if bool(supported.any()) else None
    )
    macro_f1 = float(f1.mean())
    weighted_f1 = float(np.sum(f1 * support) / n)

    per_class: dict[str, dict[str, Any]] = {}
    auroc_values: list[float] = []
    auprc_values: list[float] = []
    all_auroc_defined = True
    all_auprc_defined = True
    for class_index, class_name in enumerate(classes):
        y_binary = y_true == class_index
        auroc, auprc = _binary_ranking_metrics(
            y_binary, probabilities[:, class_index]
        )
        if auroc is None:
            all_auroc_defined = False
            if warnings is not None and warning_context is not None:
                _add_warning(
                    warnings,
                    f"{warning_context}: OVR AUROC for class {class_name!r} is "
                    "undefined because both positive and negative examples are "
                    "not present; stored as null.",
                )
        else:
            auroc_values.append(auroc)
        if auprc is None:
            all_auprc_defined = False
            if warnings is not None and warning_context is not None:
                _add_warning(
                    warnings,
                    f"{warning_context}: OVR AUPRC for class {class_name!r} is "
                    "undefined because both positive and negative examples are "
                    "not present; stored as null.",
                )
        else:
            auprc_values.append(auprc)
        per_class[class_name] = {
            "precision": float(precision[class_index]),
            "recall": float(recall[class_index]),
            "f1": float(f1[class_index]),
            "support": int(support[class_index]),
            "ovr_auroc": auroc,
            "ovr_auprc": auprc,
        }

    if not bool(supported.all()) and warnings is not None and warning_context:
        absent = [
            classes[index]
            for index, is_supported in enumerate(supported)
            if not is_supported
        ]
        _add_warning(
            warnings,
            f"{warning_context}: balanced accuracy averages only represented "
            f"true classes; absent classes={absent}.",
        )

    return {
        "n": n,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_ovr_auroc": (
            float(np.mean(auroc_values)) if all_auroc_defined else None
        ),
        "macro_ovr_auprc": (
            float(np.mean(auprc_values)) if all_auprc_defined else None
        ),
        "per_class": per_class,
        "confusion_matrix": confusion.astype(int).tolist(),
    }


def _mean_sd(
    values: Iterable[float | None],
    *,
    context: str,
    warnings: list[str],
) -> dict[str, Any]:
    values_list = list(values)
    defined = np.asarray(
        [float(value) for value in values_list if value is not None],
        dtype=np.float64,
    )
    if defined.size != len(values_list):
        _add_warning(
            warnings,
            f"{context}: one or more fold values are undefined; mean/SD use "
            "defined folds only.",
        )
    if defined.size == 0:
        return {"mean": None, "sd": None, "n_defined": 0}
    return {
        "mean": float(defined.mean()),
        "sd": float(defined.std(ddof=1)) if defined.size > 1 else None,
        "n_defined": int(defined.size),
    }


def _five_fold_summary(
    fold_metrics: Sequence[dict[str, Any]],
    classes: Sequence[str],
    warnings: list[str],
) -> dict[str, Any]:
    scalar_names = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_ovr_auroc",
        "macro_ovr_auprc",
    )
    summary: dict[str, Any] = {
        name: _mean_sd(
            (fold[name] for fold in fold_metrics),
            context=f"five-fold {name}",
            warnings=warnings,
        )
        for name in scalar_names
    }
    summary["n"] = _mean_sd(
        (fold["n"] for fold in fold_metrics),
        context="five-fold test-set size",
        warnings=warnings,
    )
    per_class: dict[str, Any] = {}
    class_metric_names = (
        "precision",
        "recall",
        "f1",
        "support",
        "ovr_auroc",
        "ovr_auprc",
    )
    for class_name in classes:
        per_class[class_name] = {
            name: _mean_sd(
                (
                    fold["per_class"][class_name][name]
                    for fold in fold_metrics
                ),
                context=f"five-fold {class_name} {name}",
                warnings=warnings,
            )
            for name in class_metric_names
        }
    summary["per_class"] = per_class
    summary["sd_definition"] = "sample standard deviation across folds (ddof=1)"
    return summary


def _percentile_ci(
    values: np.ndarray,
    point_estimate: float | None,
    *,
    context: str,
    requested_replicates: int,
    warnings: list[str],
) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0 or point_estimate is None:
        _add_warning(
            warnings,
            f"{context}: bootstrap confidence interval is undefined and stored "
            "as null.",
        )
        return {
            "point_estimate": point_estimate,
            "lower": None,
            "upper": None,
            "valid_replicates": int(finite.size),
        }
    if finite.size != requested_replicates:
        _add_warning(
            warnings,
            f"{context}: confidence interval uses {finite.size} of "
            f"{requested_replicates} bootstrap replicates.",
        )
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return {
        "point_estimate": float(point_estimate),
        "lower": float(lower),
        "upper": float(upper),
        "valid_replicates": int(finite.size),
    }


def _stratified_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: Sequence[str],
    point_metrics: dict[str, Any],
    *,
    replicates: int,
    seed: int,
    warnings: list[str],
) -> dict[str, Any]:
    class_indices = [
        np.flatnonzero(y_true == class_index)
        for class_index in range(len(classes))
    ]
    rng = np.random.default_rng(seed)
    scalar_names = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_ovr_auroc",
        "macro_ovr_auprc",
    )
    class_metric_names = (
        "precision",
        "recall",
        "f1",
        "ovr_auroc",
        "ovr_auprc",
    )
    scalar_samples = {
        name: np.full(replicates, np.nan, dtype=np.float64)
        for name in scalar_names
    }
    class_samples = {
        class_name: {
            name: np.full(replicates, np.nan, dtype=np.float64)
            for name in class_metric_names
        }
        for class_name in classes
    }

    for replicate in range(replicates):
        sampled_parts = [
            rng.choice(indices, size=indices.size, replace=True)
            for indices in class_indices
            if indices.size
        ]
        sampled_indices = np.concatenate(sampled_parts)
        metrics = _compute_metrics(
            y_true[sampled_indices],
            y_pred[sampled_indices],
            probabilities[sampled_indices],
            classes,
        )
        for name in scalar_names:
            value = metrics[name]
            if value is not None:
                scalar_samples[name][replicate] = value
        for class_name in classes:
            for name in class_metric_names:
                value = metrics["per_class"][class_name][name]
                if value is not None:
                    class_samples[class_name][name][replicate] = value

    intervals = {
        name: _percentile_ci(
            scalar_samples[name],
            point_metrics[name],
            context=f"bootstrap {name}",
            requested_replicates=replicates,
            warnings=warnings,
        )
        for name in scalar_names
    }
    per_class: dict[str, Any] = {}
    for class_name in classes:
        per_class[class_name] = {
            name: _percentile_ci(
                class_samples[class_name][name],
                point_metrics["per_class"][class_name][name],
                context=f"bootstrap {class_name} {name}",
                requested_replicates=replicates,
                warnings=warnings,
            )
            for name in class_metric_names
        }
    intervals["per_class"] = per_class
    return {
        "method": "patient-level percentile bootstrap stratified by true class",
        "confidence_level": 0.95,
        "replicates": replicates,
        "seed": seed,
        "intervals": intervals,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Any) -> None:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AggregationError(f"Cannot serialize {path}: {exc}") from exc
    _atomic_write_text(path, serialized + "\n")


def _atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(path, buffer.getvalue())


def _prediction_arrays(
    predictions: Sequence[dict[str, Any]],
    classes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(
        [prediction["true_index"] for prediction in predictions],
        dtype=np.int64,
    )
    y_pred = np.asarray(
        [prediction["predicted_index"] for prediction in predictions],
        dtype=np.int64,
    )
    probabilities = np.asarray(
        [
            [prediction["probabilities"][class_name] for class_name in classes]
            for prediction in predictions
        ],
        dtype=np.float64,
    )
    return y_true, y_pred, probabilities


def _write_outputs(
    output_dir: Path,
    predictions: Sequence[dict[str, Any]],
    classes: Sequence[str],
    pooled_metrics: dict[str, Any],
    majority_class_baseline: dict[str, Any],
    fold_metrics: Sequence[dict[str, Any]],
    five_fold_summary: dict[str, Any],
    bootstrap: dict[str, Any],
    warnings: Sequence[str],
    manifest_sha256: str,
    preprocess_fingerprint: str,
    protocol_fingerprint: str,
    checkpoint_by_fold: dict[int, str],
) -> None:
    probability_columns = [f"probability_{class_name}" for class_name in classes]
    oof_fields = [
        "case_id",
        "fold",
        "true_index",
        "true_label",
        "predicted_index",
        "predicted_class",
        *probability_columns,
        "correct",
        "checkpoint_sha256",
        "preprocess_fingerprint",
        "protocol_fingerprint",
    ]
    oof_rows: list[dict[str, Any]] = []
    for prediction in predictions:
        row = {
            "case_id": prediction["case_id"],
            "fold": prediction["fold"],
            "true_index": prediction["true_index"],
            "true_label": prediction["true_label"],
            "predicted_index": prediction["predicted_index"],
            "predicted_class": prediction["predicted_class"],
            "correct": prediction["correct"],
            "checkpoint_sha256": prediction["checkpoint_sha256"],
            "preprocess_fingerprint": prediction["preprocess_fingerprint"],
            "protocol_fingerprint": prediction["protocol_fingerprint"],
        }
        for class_name, column in zip(classes, probability_columns):
            row[column] = format(
                prediction["probabilities"][class_name], ".17g"
            )
        oof_rows.append(row)
    _atomic_write_csv(output_dir / "oof_predictions.csv", oof_fields, oof_rows)

    confusion_rows = []
    for row_index, true_label in enumerate(classes):
        row: dict[str, Any] = {"true_label": true_label}
        for column_index, predicted_label in enumerate(classes):
            row[f"predicted_{predicted_label}"] = pooled_metrics[
                "confusion_matrix"
            ][row_index][column_index]
        row["support"] = pooled_metrics["per_class"][true_label]["support"]
        confusion_rows.append(row)
    confusion_fields = [
        "true_label",
        *(f"predicted_{class_name}" for class_name in classes),
        "support",
    ]
    _atomic_write_csv(
        output_dir / "confusion_matrix.csv",
        confusion_fields,
        confusion_rows,
    )

    patients_dir = output_dir / "patients"
    for prediction in predictions:
        patient_result = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "case_id": prediction["case_id"],
            "fold": prediction["fold"],
            "predicted_index": prediction["predicted_index"],
            "predicted_class": prediction["predicted_class"],
            "probabilities": {
                class_name: prediction["probabilities"][class_name]
                for class_name in classes
            },
            "true_index": prediction["true_index"],
            "true_label": prediction["true_label"],
            "correct": prediction["correct"],
            "checkpoint_sha256": prediction["checkpoint_sha256"],
            "preprocess_fingerprint": prediction["preprocess_fingerprint"],
            "protocol_fingerprint": prediction["protocol_fingerprint"],
        }
        _atomic_write_json(
            patients_dir / prediction["case_id"] / "prediction.json",
            patient_result,
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "five-fold out-of-fold classification",
        "classes": list(classes),
        "n_patients": len(predictions),
        "coverage": 1.0,
        "manifest_sha256": manifest_sha256,
        "preprocess_fingerprint": preprocess_fingerprint,
        "protocol_fingerprint": protocol_fingerprint,
        "checkpoint_sha256_by_fold": {
            str(fold): checkpoint_by_fold[fold] for fold in EXPECTED_FOLDS
        },
        "pooled_oof": pooled_metrics,
        "majority_class_baseline": majority_class_baseline,
        "folds": list(fold_metrics),
        "five_fold_mean_sd": five_fold_summary,
        "bootstrap_95_ci": bootstrap,
        "confusion_matrix": {
            "row_definition": "true class",
            "column_definition": "predicted class",
            "labels": list(classes),
            "matrix": pooled_metrics["confusion_matrix"],
        },
        "warnings": list(warnings),
    }
    # summary.json is the completion marker and is therefore published last.
    _atomic_write_json(output_dir / "summary.json", summary)


def aggregate(
    manifest_path: Path,
    output_dir: Path,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if bootstrap_replicates <= 0:
        raise AggregationError("--bootstrap-replicates must be positive.")
    manifest, manifest_raw = _load_json(manifest_path)
    (
        classes,
        preprocess_fingerprint,
        patient_order,
        patient_by_id,
        expected_test_by_fold,
    ) = _validate_manifest(manifest)
    predictions, checkpoint_by_fold, protocol_fingerprint = (
        _load_and_validate_predictions(
            output_dir,
            classes,
            preprocess_fingerprint,
            patient_order,
            patient_by_id,
            expected_test_by_fold,
        )
    )
    y_true, y_pred, probabilities = _prediction_arrays(predictions, classes)

    warnings: list[str] = []
    pooled_metrics = _compute_metrics(
        y_true,
        y_pred,
        probabilities,
        classes,
        warning_context="pooled OOF",
        warnings=warnings,
    )
    majority_index = int(np.bincount(y_true, minlength=len(classes)).argmax())
    majority_predictions = np.full_like(y_true, majority_index)
    majority_probabilities = np.zeros_like(probabilities)
    majority_probabilities[:, majority_index] = 1.0
    majority_class_baseline = {
        "predicted_class": classes[majority_index],
        **_compute_metrics(
            y_true,
            majority_predictions,
            majority_probabilities,
            classes,
        ),
    }
    fold_metrics: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        fold_indices = np.asarray(
            [
                index
                for index, prediction in enumerate(predictions)
                if prediction["fold"] == fold
            ],
            dtype=np.int64,
        )
        metrics = _compute_metrics(
            y_true[fold_indices],
            y_pred[fold_indices],
            probabilities[fold_indices],
            classes,
            warning_context=f"fold {fold}",
            warnings=warnings,
        )
        metrics = {
            "fold": fold,
            "checkpoint_sha256": checkpoint_by_fold[fold],
            **metrics,
        }
        fold_metrics.append(metrics)

    five_fold_summary = _five_fold_summary(fold_metrics, classes, warnings)
    bootstrap = _stratified_bootstrap(
        y_true,
        y_pred,
        probabilities,
        classes,
        pooled_metrics,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        warnings=warnings,
    )
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_outputs(
        output_dir,
        predictions,
        classes,
        pooled_metrics,
        majority_class_baseline,
        fold_metrics,
        five_fold_summary,
        bootstrap,
        warnings,
        manifest_sha256,
        preprocess_fingerprint,
        protocol_fingerprint,
        checkpoint_by_fold,
    )
    return {
        "n_patients": len(predictions),
        "accuracy": pooled_metrics["accuracy"],
        "balanced_accuracy": pooled_metrics["balanced_accuracy"],
        "macro_f1": pooled_metrics["macro_f1"],
        "output_dir": str(output_dir),
        "warnings": warnings,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly aggregate five held-out fold prediction files into pooled "
            "OOF metrics and privacy-preserving per-case results."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("cache/manifest.json"),
        help="Dataset/fold manifest JSON (default: cache/manifest.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Experiment output root containing "
            "folds/fold_{0..4}/test_predictions.jsonl."
        ),
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
        help=(
            "Number of true-class-stratified patient bootstrap replicates "
            f"(default: {DEFAULT_BOOTSTRAP_REPLICATES})."
        ),
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help=f"Bootstrap random seed (default: {DEFAULT_BOOTSTRAP_SEED}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = aggregate(
            args.manifest,
            args.output_dir,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
    except (AggregationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
