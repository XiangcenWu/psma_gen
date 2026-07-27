#!/usr/bin/env python3
"""Compare pooled DL OOF predictions with agent strict-majority predictions.

The comparison is patient paired.  The DL side contributes exactly one
out-of-fold prediction per case.  On the agent side, only valid completed
trajectories vote; a patient prediction is made only when one class receives
strictly more than half of those votes.  Ties, pluralities, and patients
without a completed trajectory abstain and are scored as incorrect.

Only opaque case IDs and fixed observed-management-v1 class labels are written
to the reports.  Patient names and free-text treatments are never loaded into
the output payload.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1.0"
LABEL_MAPPING_VERSION = "observed-management-v1"
CLASS_NAMES = (
    "radical_prostatectomy",
    "systemic_treatment",
    "local_treatment",
    "other_examination",
)
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_SEED = 20_260_727
CASE_ID_RE = re.compile(r"^case_[0-9a-fA-F]{16}$")
INTEGER_RE = re.compile(r"^(0|[1-9][0-9]*)$")
DIGITS_RE = re.compile(r"^[0-9]+$")


class ComparisonError(RuntimeError):
    """Raised when inputs cannot support a strict paired comparison."""


@dataclass(frozen=True)
class DLPrediction:
    case_id: str
    fold: int
    true_index: int
    true_label: str
    predicted_index: int
    predicted_class: str
    correct: bool


@dataclass(frozen=True)
class AgentPatient:
    case_id: str
    observed: str | None
    attempted: int
    completed_predictions: tuple[str, ...]
    prediction: str | None
    abstained: bool
    abstention_reason: str | None
    vote_counts: dict[str, int]

    @property
    def correct(self) -> bool:
        return self.observed is not None and self.prediction == self.observed


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ComparisonError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ComparisonError(f"{path}: top-level JSON value must be an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ComparisonError(f"Cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ComparisonError(f"{context} must be a non-empty string")
    return value.strip()


def _validate_case_id(value: Any, context: str) -> str:
    case_id = _require_nonempty_string(value, context)
    if not CASE_ID_RE.fullmatch(case_id):
        raise ComparisonError(
            f"{context}={case_id!r} is not an opaque case_<16 hex digits> ID"
        )
    return case_id


def _validate_class(value: Any, context: str) -> str:
    label = _require_nonempty_string(value, context)
    if label not in CLASS_NAMES:
        raise ComparisonError(
            f"{context}={label!r} is outside {LABEL_MAPPING_VERSION}: "
            f"{', '.join(CLASS_NAMES)}"
        )
    return label


def _parse_integer(value: Any, context: str) -> int:
    if not isinstance(value, str):
        raise ComparisonError(f"{context} must be an integer-formatted CSV value")
    stripped = value.strip()
    if not INTEGER_RE.fullmatch(stripped):
        raise ComparisonError(f"{context}={value!r} is not a non-negative integer")
    return int(stripped)


def _parse_boolean(value: Any, context: str) -> bool:
    if not isinstance(value, str):
        raise ComparisonError(f"{context} must be a boolean-formatted CSV value")
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ComparisonError(f"{context}={value!r} is not True/False or 1/0")


def _validate_dl_summary(dl_output: Path, prediction_count: int) -> None:
    """Cross-check aggregate metadata when summary.json is available."""

    summary_path = dl_output / "summary.json"
    if not summary_path.is_file():
        return
    summary = _load_json(summary_path)
    classes = summary.get("classes")
    if classes != list(CLASS_NAMES):
        raise ComparisonError(
            f"{summary_path}: classes must exactly equal observed-management-v1 "
            f"order {list(CLASS_NAMES)!r}"
        )
    n_patients = summary.get("n_patients")
    if isinstance(n_patients, bool) or not isinstance(n_patients, int):
        raise ComparisonError(f"{summary_path}: n_patients must be an integer")
    if n_patients != prediction_count:
        raise ComparisonError(
            f"{summary_path}: n_patients={n_patients} but OOF CSV has "
            f"{prediction_count} cases"
        )


def _load_dl_predictions(dl_output: Path) -> tuple[dict[str, DLPrediction], Path]:
    dl_output = dl_output.expanduser().resolve()
    if not dl_output.is_dir():
        raise ComparisonError(f"{dl_output}: --dl-output must be a directory")
    source = dl_output / "oof_predictions.csv"
    if not source.is_file():
        raise ComparisonError(f"{source}: expected pooled OOF prediction CSV")

    required_fields = {
        "case_id",
        "fold",
        "true_index",
        "true_label",
        "predicted_index",
        "predicted_class",
        "correct",
    }
    predictions: dict[str, DLPrediction] = {}
    folds: set[int] = set()
    try:
        handle = source.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ComparisonError(f"Cannot open {source}: {exc}") from exc

    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ComparisonError(f"{source}: CSV header is missing")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            duplicates = sorted(
                name
                for name, count in Counter(reader.fieldnames).items()
                if count > 1
            )
            raise ComparisonError(
                f"{source}: duplicate CSV header fields: {', '.join(duplicates)}"
            )
        missing = sorted(required_fields - set(reader.fieldnames))
        if missing:
            raise ComparisonError(
                f"{source}: missing required columns: {', '.join(missing)}"
            )

        for line_number, row in enumerate(reader, start=2):
            context = f"{source}:{line_number}"
            if None in row:
                raise ComparisonError(f"{context}: row has fields beyond the header")
            if not any(isinstance(value, str) and value.strip() for value in row.values()):
                raise ComparisonError(f"{context}: blank rows are not allowed")

            case_id = _validate_case_id(row["case_id"], f"{context}.case_id")
            if case_id in predictions:
                raise ComparisonError(
                    f"{context}: duplicate OOF prediction for {case_id}"
                )
            fold = _parse_integer(row["fold"], f"{context}.fold")
            if fold not in range(5):
                raise ComparisonError(f"{context}.fold={fold} is outside 0..4")
            true_index = _parse_integer(
                row["true_index"], f"{context}.true_index"
            )
            predicted_index = _parse_integer(
                row["predicted_index"], f"{context}.predicted_index"
            )
            if true_index >= len(CLASS_NAMES):
                raise ComparisonError(
                    f"{context}.true_index={true_index} is outside the class range"
                )
            if predicted_index >= len(CLASS_NAMES):
                raise ComparisonError(
                    f"{context}.predicted_index={predicted_index} is outside "
                    "the class range"
                )
            true_label = _validate_class(
                row["true_label"], f"{context}.true_label"
            )
            predicted_class = _validate_class(
                row["predicted_class"], f"{context}.predicted_class"
            )
            if true_label != CLASS_NAMES[true_index]:
                raise ComparisonError(
                    f"{context}: true label/index disagree with fixed class order"
                )
            if predicted_class != CLASS_NAMES[predicted_index]:
                raise ComparisonError(
                    f"{context}: predicted label/index disagree with fixed class order"
                )
            correct = _parse_boolean(row["correct"], f"{context}.correct")
            derived_correct = predicted_class == true_label
            if correct != derived_correct:
                raise ComparisonError(
                    f"{context}: reported correctness disagrees with labels"
                )
            predictions[case_id] = DLPrediction(
                case_id=case_id,
                fold=fold,
                true_index=true_index,
                true_label=true_label,
                predicted_index=predicted_index,
                predicted_class=predicted_class,
                correct=correct,
            )
            folds.add(fold)

    if not predictions:
        raise ComparisonError(f"{source}: no OOF predictions found")
    if folds != set(range(5)):
        raise ComparisonError(
            f"{source}: expected predictions from folds 0..4, found {sorted(folds)}"
        )
    _validate_dl_summary(dl_output, len(predictions))
    return predictions, source


def _optional_observed_label(
    document: Mapping[str, Any], source: Path
) -> str | None:
    candidates: list[tuple[str, Any]] = []
    for block_name in ("evaluation", "result"):
        block = document.get(block_name)
        if block is None:
            continue
        if not isinstance(block, Mapping):
            raise ComparisonError(f"{source}: {block_name} must be an object or null")
        if "observed_management_category" in block:
            candidates.append(
                (
                    f"{block_name}.observed_management_category",
                    block["observed_management_category"],
                )
            )
    if not candidates:
        return None
    labels = [
        _validate_class(value, f"{source}:{field_name}")
        for field_name, value in candidates
    ]
    if len(set(labels)) != 1:
        raise ComparisonError(
            f"{source}: observed labels disagree inside one trajectory: {labels}"
        )
    return labels[0]


def _completed_prediction(
    document: Mapping[str, Any],
    source: Path,
    observed: str | None,
) -> str:
    if document.get("failure") is not None:
        raise ComparisonError(f"{source}: completed trajectory has non-null failure")
    if observed is None:
        raise ComparisonError(
            f"{source}: completed trajectory lacks observed management category"
        )
    result = document.get("result")
    evaluation = document.get("evaluation")
    if not isinstance(result, Mapping):
        raise ComparisonError(f"{source}: completed result must be an object")
    if not isinstance(evaluation, Mapping):
        raise ComparisonError(f"{source}: completed evaluation must be an object")

    prediction = _validate_class(result.get("answer"), f"{source}:result.answer")
    prediction_block = document.get("prediction")
    if isinstance(prediction_block, Mapping):
        accepted = prediction_block.get("accepted")
        if isinstance(accepted, Mapping) and accepted.get("answer") is not None:
            accepted_prediction = _validate_class(
                accepted["answer"], f"{source}:prediction.accepted.answer"
            )
            if accepted_prediction != prediction:
                raise ComparisonError(
                    f"{source}: result.answer disagrees with accepted prediction"
                )

    derived_correct = prediction == observed
    for block_name, block in (("evaluation", evaluation), ("result", result)):
        if "correct" not in block:
            continue
        reported = block["correct"]
        if not isinstance(reported, bool):
            raise ComparisonError(
                f"{source}: {block_name}.correct must be a boolean"
            )
        if reported != derived_correct:
            raise ComparisonError(
                f"{source}: {block_name}.correct disagrees with prediction "
                "and observed label"
            )
    return prediction


def _strict_majority(
    predictions: Sequence[str],
) -> tuple[str | None, bool, str | None, dict[str, int]]:
    counts = Counter(predictions)
    vote_counts = {class_name: counts[class_name] for class_name in CLASS_NAMES}
    if not predictions:
        return None, True, "no_completed_trajectory", vote_counts
    maximum = max(counts.values())
    winners = [label for label, count in counts.items() if count == maximum]
    if len(winners) != 1:
        return None, True, "tie", vote_counts
    if maximum * 2 <= len(predictions):
        return None, True, "no_strict_majority", vote_counts
    return winners[0], False, None, vote_counts


def _resolve_patients_dir(agent_output: Path) -> tuple[Path, Path]:
    agent_output = agent_output.expanduser().resolve()
    if not agent_output.is_dir():
        raise ComparisonError(f"{agent_output}: --agent-output must be a directory")
    if agent_output.name == "patients":
        return agent_output, agent_output.parent
    patients_dir = agent_output / "patients"
    if not patients_dir.is_dir():
        raise ComparisonError(
            f"{agent_output}: expected patients/case_*/trajectory_*.json"
        )
    return patients_dir, agent_output


def _load_agent_patients(
    agent_output: Path,
) -> tuple[dict[str, AgentPatient], Path, int]:
    patients_dir, experiment_dir = _resolve_patients_dir(agent_output)
    case_dirs = sorted(
        item
        for item in patients_dir.iterdir()
        if item.is_dir() and item.name.startswith("case_")
    )
    if not case_dirs:
        raise ComparisonError(f"{patients_dir}: no case_* directories found")

    patients: dict[str, AgentPatient] = {}
    total_trajectory_files = 0
    for case_dir in case_dirs:
        case_id = _validate_case_id(case_dir.name, f"{case_dir}:directory name")
        trajectory_paths = sorted(case_dir.glob("trajectory_*.json"))
        if not trajectory_paths:
            raise ComparisonError(f"{case_dir}: no trajectory_*.json files found")

        observed: str | None = None
        completed_predictions: list[str] = []
        trajectory_numbers: set[int] = set()
        for source in trajectory_paths:
            total_trajectory_files += 1
            suffix = source.stem.removeprefix("trajectory_")
            if not DIGITS_RE.fullmatch(suffix):
                raise ComparisonError(
                    f"{source}: trajectory filename suffix must be an integer"
                )
            filename_number = int(suffix)
            if filename_number in trajectory_numbers:
                raise ComparisonError(
                    f"{case_dir}: duplicate trajectory number {filename_number}"
                )
            trajectory_numbers.add(filename_number)

            document = _load_json(source)
            document_case_id = _validate_case_id(
                document.get("case_id"), f"{source}:case_id"
            )
            if document_case_id != case_id:
                raise ComparisonError(
                    f"{source}: case_id does not match its patient directory"
                )
            trajectory_number = document.get("trajectory_number")
            if (
                isinstance(trajectory_number, bool)
                or not isinstance(trajectory_number, int)
                or trajectory_number != filename_number
            ):
                raise ComparisonError(
                    f"{source}: trajectory_number does not match filename"
                )

            provenance = document.get("provenance")
            if not isinstance(provenance, Mapping):
                raise ComparisonError(f"{source}: provenance must be an object")
            mapping_version = provenance.get("label_mapping_version")
            if mapping_version != LABEL_MAPPING_VERSION:
                raise ComparisonError(
                    f"{source}: label_mapping_version={mapping_version!r}; "
                    f"expected {LABEL_MAPPING_VERSION!r}"
                )

            status = _require_nonempty_string(
                document.get("status"), f"{source}:status"
            )
            trajectory_observed = _optional_observed_label(document, source)
            if trajectory_observed is not None:
                if observed is None:
                    observed = trajectory_observed
                elif observed != trajectory_observed:
                    raise ComparisonError(
                        f"{case_dir}: observed label changes across trajectories"
                    )
            if status == "completed":
                completed_predictions.append(
                    _completed_prediction(document, source, trajectory_observed)
                )

        prediction, abstained, reason, vote_counts = _strict_majority(
            completed_predictions
        )
        patients[case_id] = AgentPatient(
            case_id=case_id,
            observed=observed,
            attempted=len(trajectory_paths),
            completed_predictions=tuple(completed_predictions),
            prediction=prediction,
            abstained=abstained,
            abstention_reason=reason,
            vote_counts=vote_counts,
        )

    return patients, experiment_dir, total_trajectory_files


def _validate_pairing(
    dl_predictions: Mapping[str, DLPrediction],
    agent_patients: Mapping[str, AgentPatient],
) -> None:
    dl_ids = set(dl_predictions)
    agent_ids = set(agent_patients)
    if dl_ids != agent_ids:
        dl_only = sorted(dl_ids - agent_ids)
        agent_only = sorted(agent_ids - dl_ids)
        raise ComparisonError(
            "DL and agent case sets must match exactly; "
            f"DL-only={dl_only}, agent-only={agent_only}"
        )
    for case_id in sorted(dl_ids):
        dl_label = dl_predictions[case_id].true_label
        agent_label = agent_patients[case_id].observed
        if agent_label is not None and dl_label != agent_label:
            raise ComparisonError(
                f"{case_id}: DL true_label={dl_label!r} disagrees with agent "
                f"observed label={agent_label!r}"
            )
    missing_classes = sorted(
        set(CLASS_NAMES)
        - {prediction.true_label for prediction in dl_predictions.values()}
    )
    if missing_classes:
        raise ComparisonError(
            "Paired cohort must contain all observed-management-v1 classes; "
            f"missing {missing_classes}"
        )


def _paired_cases(
    dl_predictions: Mapping[str, DLPrediction],
    agent_patients: Mapping[str, AgentPatient],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for case_id in sorted(dl_predictions):
        dl = dl_predictions[case_id]
        agent = agent_patients[case_id]
        completed = len(agent.completed_predictions)
        cases.append(
            {
                "case_id": case_id,
                "fold": dl.fold,
                "true_label": dl.true_label,
                "dl_prediction": dl.predicted_class,
                "dl_correct": dl.correct,
                "agent_prediction": agent.prediction,
                "agent_correct": agent.prediction == dl.true_label,
                "agent_ground_truth_crosschecked": agent.observed is not None,
                "agent_abstained": agent.abstained,
                "agent_abstention_reason": agent.abstention_reason,
                "agent_attempted_trajectories": agent.attempted,
                "agent_completed_trajectories": completed,
                "agent_trajectory_coverage": completed / agent.attempted,
                "agent_vote_counts": dict(agent.vote_counts),
            }
        )
    return cases


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _classification_metrics(
    cases: Sequence[Mapping[str, Any]],
    prediction_key: str,
) -> dict[str, Any]:
    n_cases = len(cases)
    if not n_cases:
        raise ComparisonError("Cannot calculate metrics for an empty cohort")
    correct = sum(
        case[prediction_key] == case["true_label"] for case in cases
    )
    covered = sum(case[prediction_key] is not None for case in cases)
    confusion = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    per_class: dict[str, dict[str, int | float | None]] = {}
    f1_values: list[float] = []
    recall_values: list[float] = []

    for case in cases:
        prediction = case[prediction_key]
        if prediction is not None:
            true_index = CLASS_NAMES.index(str(case["true_label"]))
            predicted_index = CLASS_NAMES.index(str(prediction))
            confusion[true_index][predicted_index] += 1

    for label in CLASS_NAMES:
        true_positive = sum(
            case["true_label"] == label and case[prediction_key] == label
            for case in cases
        )
        false_positive = sum(
            case["true_label"] != label and case[prediction_key] == label
            for case in cases
        )
        false_negative = sum(
            case["true_label"] == label and case[prediction_key] != label
            for case in cases
        )
        support = true_positive + false_negative
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, support)
        if precision is None or recall is None or precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1_values.append(f1)
        recall_values.append(0.0 if recall is None else recall)
        per_class[label] = {
            "support": support,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "abstentions": sum(
                case["true_label"] == label and case[prediction_key] is None
                for case in cases
            ),
        }

    return {
        "n_patients": n_cases,
        "correct": correct,
        "accuracy": correct / n_cases,
        "macro_f1": statistics.fmean(f1_values),
        "balanced_accuracy": statistics.fmean(recall_values),
        "covered": covered,
        "coverage": covered / n_cases,
        "abstentions": n_cases - covered,
        "abstention_rate": (n_cases - covered) / n_cases,
        "predicted_distribution": {
            label: sum(case[prediction_key] == label for case in cases)
            for label in CLASS_NAMES
        },
        "per_class": per_class,
        "confusion_matrix": {
            "row_definition": "true class",
            "column_definition": "predicted class; abstentions are omitted",
            "labels": list(CLASS_NAMES),
            "matrix": confusion,
        },
    }


def _metric_triplet(
    cases: Sequence[Mapping[str, Any]],
    sampled_indices: Sequence[int],
    prediction_key: str,
) -> tuple[float, float, float]:
    n = len(sampled_indices)
    correct = 0
    true_positive = Counter({label: 0 for label in CLASS_NAMES})
    false_positive = Counter({label: 0 for label in CLASS_NAMES})
    false_negative = Counter({label: 0 for label in CLASS_NAMES})

    for index in sampled_indices:
        case = cases[index]
        observed = str(case["true_label"])
        prediction = case[prediction_key]
        if prediction == observed:
            correct += 1
        for label in CLASS_NAMES:
            if observed == label and prediction == label:
                true_positive[label] += 1
            elif observed != label and prediction == label:
                false_positive[label] += 1
            elif observed == label and prediction != label:
                false_negative[label] += 1

    f1_values: list[float] = []
    recalls: list[float] = []
    for label in CLASS_NAMES:
        tp = true_positive[label]
        fp = false_positive[label]
        fn = false_negative[label]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        recalls.append(recall)
    return (
        correct / n,
        statistics.fmean(f1_values),
        statistics.fmean(recalls),
    )


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ComparisonError("Cannot calculate percentile of an empty sample")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def _paired_stratified_bootstrap(
    cases: Sequence[Mapping[str, Any]],
    dl_metrics: Mapping[str, Any],
    agent_metrics: Mapping[str, Any],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples < 1:
        raise ComparisonError("--bootstrap-samples must be at least 1")
    strata = {
        label: [
            index
            for index, case in enumerate(cases)
            if case["true_label"] == label
        ]
        for label in CLASS_NAMES
    }
    if any(not indices for indices in strata.values()):
        raise ComparisonError("Every fixed class must be present for stratified bootstrap")

    rng = random.Random(seed)
    metric_names = ("accuracy", "macro_f1", "balanced_accuracy")
    differences: dict[str, list[float]] = {
        metric_name: [] for metric_name in metric_names
    }
    for _ in range(samples):
        sampled_indices: list[int] = []
        for label in CLASS_NAMES:
            indices = strata[label]
            sampled_indices.extend(
                indices[rng.randrange(len(indices))] for _ in range(len(indices))
            )
        dl_values = _metric_triplet(cases, sampled_indices, "dl_prediction")
        agent_values = _metric_triplet(
            cases, sampled_indices, "agent_prediction"
        )
        for metric_name, dl_value, agent_value in zip(
            metric_names, dl_values, agent_values
        ):
            differences[metric_name].append(dl_value - agent_value)

    result: dict[str, Any] = {
        "estimand": "paired patient-level metric difference",
        "direction": "DL_minus_Agent",
        "confidence_level": 0.95,
        "ci_method": "true-class-stratified paired percentile bootstrap",
        "bootstrap_samples": samples,
        "seed": seed,
        "metrics": {},
    }
    for metric_name in metric_names:
        values = sorted(differences[metric_name])
        result["metrics"][metric_name] = {
            "dl": dl_metrics[metric_name],
            "agent": agent_metrics[metric_name],
            "difference": dl_metrics[metric_name] - agent_metrics[metric_name],
            "ci_low": _percentile(values, 0.025),
            "ci_high": _percentile(values, 0.975),
            "bootstrap_standard_error": (
                statistics.stdev(values) if len(values) > 1 else 0.0
            ),
        }
    return result


def _mcnemar_exact(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    both_correct = 0
    dl_only = 0
    agent_only = 0
    both_incorrect = 0
    for case in cases:
        dl_correct = bool(case["dl_correct"])
        agent_correct = bool(case["agent_correct"])
        if dl_correct and agent_correct:
            both_correct += 1
        elif dl_correct:
            dl_only += 1
        elif agent_correct:
            agent_only += 1
        else:
            both_incorrect += 1
    discordant = dl_only + agent_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(dl_only, agent_only)
        lower_tail = sum(
            math.comb(discordant, value) for value in range(lower + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * lower_tail)
    return {
        "unit": "patient correctness",
        "alternative": "two-sided",
        "method": "exact binomial McNemar test",
        "both_correct": both_correct,
        "dl_only_correct": dl_only,
        "agent_only_correct": agent_only,
        "both_incorrect": both_incorrect,
        "discordant_pairs": discordant,
        "odds_ratio": _safe_ratio(dl_only, agent_only),
        "p_value": p_value,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, value: Any) -> None:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ComparisonError(f"Cannot serialize {path}: {exc}") from exc
    _atomic_write_text(path, text + "\n")


def _atomic_write_csv(path: Path, cases: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = (
        "case_id",
        "fold",
        "true_label",
        "dl_prediction",
        "dl_correct",
        "agent_prediction",
        "agent_correct",
        "agent_abstained",
        "agent_abstention_reason",
        "agent_attempted_trajectories",
        "agent_completed_trajectories",
        "agent_trajectory_coverage",
        "agent_vote_counts",
        "agent_ground_truth_crosschecked",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for case in cases:
        writer.writerow(
            {
                field: (
                    json.dumps(
                        case[field],
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    if field == "agent_vote_counts"
                    else case[field]
                )
                for field in fieldnames
            }
        )
    _atomic_write_text(path, buffer.getvalue())


def compare(
    *,
    dl_output: Path,
    agent_output: Path,
    output: Path | None,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Run the strict paired comparison and atomically publish both reports."""

    if bootstrap_samples < 1:
        raise ComparisonError("--bootstrap-samples must be at least 1")
    dl_predictions, dl_csv = _load_dl_predictions(dl_output)
    agent_patients, agent_experiment, trajectory_file_count = (
        _load_agent_patients(agent_output)
    )
    _validate_pairing(dl_predictions, agent_patients)
    cases = _paired_cases(dl_predictions, agent_patients)

    dl_metrics = _classification_metrics(cases, "dl_prediction")
    agent_metrics = _classification_metrics(cases, "agent_prediction")
    completed_trajectories = sum(
        int(case["agent_completed_trajectories"]) for case in cases
    )
    attempted_trajectories = sum(
        int(case["agent_attempted_trajectories"]) for case in cases
    )
    agent_metrics["trajectory"] = {
        "attempted": attempted_trajectories,
        "completed": completed_trajectories,
        "coverage": completed_trajectories / attempted_trajectories,
    }
    agent_metrics["patients_with_completed_trajectory"] = sum(
        int(case["agent_completed_trajectories"]) > 0 for case in cases
    )
    agent_metrics["patient_evaluable_trajectory_coverage"] = (
        agent_metrics["patients_with_completed_trajectory"] / len(cases)
    )
    agent_metrics["ground_truth_crosschecked_from_agent_output"] = sum(
        bool(case["agent_ground_truth_crosschecked"]) for case in cases
    )
    agent_metrics["ground_truth_supplied_by_dl_manifest_only"] = sum(
        not bool(case["agent_ground_truth_crosschecked"]) for case in cases
    )

    bootstrap = _paired_stratified_bootstrap(
        cases,
        dl_metrics,
        agent_metrics,
        samples=bootstrap_samples,
        seed=seed,
    )
    mcnemar = _mcnemar_exact(cases)

    resolved_dl_output = dl_output.expanduser().resolve()
    json_output = (
        output.expanduser().resolve()
        if output is not None
        else resolved_dl_output / "agent_comparison.json"
    )
    csv_output = (
        json_output.with_suffix(".csv")
        if json_output.suffix
        else json_output.with_name(json_output.name + ".csv")
    )
    if json_output == csv_output:
        raise ComparisonError("JSON and CSV output paths must be different")

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "paired DL OOF versus agent strict-majority comparison",
        "label_mapping_version": LABEL_MAPPING_VERSION,
        "classes": list(CLASS_NAMES),
        "strict_majority_definition": (
            "one class must receive strictly more than half of valid completed "
            "trajectory votes; otherwise the patient abstains and is scored incorrect"
        ),
        "n_paired_patients": len(cases),
        "inputs": {
            "dl_output": str(resolved_dl_output),
            "dl_oof_predictions_sha256": _sha256_file(dl_csv),
            "agent_output": str(agent_experiment),
            "agent_trajectory_files": trajectory_file_count,
        },
        "outputs": {
            "json": str(json_output),
            "per_case_csv": str(csv_output),
        },
        "metrics": {
            "dl": dl_metrics,
            "agent": agent_metrics,
        },
        "paired_comparison": {
            "mcnemar_exact": mcnemar,
            "stratified_bootstrap_95_ci": bootstrap,
        },
        "cases": cases,
    }

    # Publish the per-case table first; JSON is the completion marker.
    _atomic_write_csv(csv_output, cases)
    _atomic_write_json(json_output, report)
    return {
        "n_paired_patients": len(cases),
        "dl_accuracy": dl_metrics["accuracy"],
        "agent_accuracy": agent_metrics["accuracy"],
        "accuracy_difference": (
            dl_metrics["accuracy"] - agent_metrics["accuracy"]
        ),
        "mcnemar_p_value": mcnemar["p_value"],
        "json_output": str(json_output),
        "csv_output": str(csv_output),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strict patient-paired comparison of five-fold DL OOF predictions "
            "with agent strict-majority predictions."
        )
    )
    parser.add_argument(
        "--dl-output",
        type=Path,
        required=True,
        help="DL output directory containing oof_predictions.csv.",
    )
    parser.add_argument(
        "--agent-output",
        type=Path,
        required=True,
        help=(
            "Agent run directory containing patients/case_*/trajectory_*.json, "
            "or the patients directory itself."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Comparison JSON path (default: "
            "<dl-output>/agent_comparison.json); a sibling CSV is also written."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        "--bootstrap",
        dest="bootstrap_samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help=f"Paired stratified bootstrap samples (default: {DEFAULT_BOOTSTRAP_SAMPLES}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Bootstrap seed (default: {DEFAULT_SEED}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = compare(
            dl_output=args.dl_output,
            agent_output=args.agent_output,
            output=args.output,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    except ComparisonError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
