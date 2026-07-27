#!/usr/bin/env python3
"""Evaluate trajectory-RAG ablations with patient-cluster-aware statistics.

Each immediate child of ``--output-root`` that contains a ``patients`` directory
is treated as one experiment.  A trajectory is evaluable only when its JSON
status is ``completed`` and the required prediction and observed labels are
present.  Non-completed trajectories contribute to failure/coverage statistics
but are not silently counted as classification errors.

The evaluator is intentionally dependency-light.  All metrics and the paired
patient-cluster bootstrap use the Python standard library.  SciPy is optional
and is used only for the exact McNemar p-value; when unavailable, the p-value is
reported as JSON null with an explicit reason.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import statistics
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class EvaluationError(ValueError):
    """Raised when an input violates an evaluation invariant."""


@dataclass(frozen=True)
class Trajectory:
    """One successfully evaluated trajectory."""

    source: Path
    prediction: str
    observed: str
    correct: bool


@dataclass
class Patient:
    """All discovered trajectory attempts for one patient."""

    patient_id: str
    attempted: int = 0
    trajectories: list[Trajectory] = field(default_factory=list)
    observed: str | None = None
    failure_stages: Counter[str] = field(default_factory=Counter)

    @property
    def evaluated(self) -> int:
        return len(self.trajectories)

    @property
    def correct(self) -> int:
        return sum(item.correct for item in self.trajectories)

    @property
    def mean_accuracy(self) -> float | None:
        if not self.trajectories:
            return None
        return self.correct / len(self.trajectories)


@dataclass
class Experiment:
    """Parsed experiment output."""

    name: str
    path: Path
    patients: dict[str, Patient]

    @property
    def trajectories(self) -> list[Trajectory]:
        return [
            trajectory
            for patient in self.patients.values()
            for trajectory in patient.trajectories
        ]


def _nonempty_label(value: Any, *, field_name: str, source: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(
            f"{source}: {field_name} must be a non-empty string, got {value!r}"
        )
    return value.strip()


def _optional_observed_label(document: Mapping[str, Any], source: Path) -> str | None:
    """Read and cross-check any observed labels present in a document."""

    candidates: list[tuple[str, Any]] = []
    evaluation = document.get("evaluation")
    result = document.get("result")
    if evaluation is not None:
        if not isinstance(evaluation, Mapping):
            raise EvaluationError(f"{source}: evaluation must be an object or null")
        if "observed_management_category" in evaluation:
            candidates.append(
                (
                    "evaluation.observed_management_category",
                    evaluation["observed_management_category"],
                )
            )
    if result is not None:
        if not isinstance(result, Mapping):
            raise EvaluationError(f"{source}: result must be an object or null")
        if "observed_management_category" in result:
            candidates.append(
                (
                    "result.observed_management_category",
                    result["observed_management_category"],
                )
            )

    labels = [
        _nonempty_label(value, field_name=field_name, source=source)
        for field_name, value in candidates
    ]
    if not labels:
        return None
    if len(set(labels)) != 1:
        raise EvaluationError(
            f"{source}: inconsistent observed labels inside one trajectory: {labels}"
        )
    return labels[0]


def _completed_trajectory(
    document: Mapping[str, Any], source: Path, observed: str | None
) -> Trajectory:
    if document.get("failure") is not None:
        raise EvaluationError(
            f"{source}: status is completed but failure is not null"
        )
    if observed is None:
        raise EvaluationError(
            f"{source}: completed trajectory lacks observed_management_category"
        )

    result = document.get("result")
    evaluation = document.get("evaluation")
    if not isinstance(result, Mapping):
        raise EvaluationError(f"{source}: completed trajectory result must be an object")
    if not isinstance(evaluation, Mapping):
        raise EvaluationError(
            f"{source}: completed trajectory evaluation must be an object"
        )
    prediction = _nonempty_label(
        result.get("answer"), field_name="result.answer", source=source
    )

    prediction_block = document.get("prediction")
    if isinstance(prediction_block, Mapping):
        accepted = prediction_block.get("accepted")
        if isinstance(accepted, Mapping) and accepted.get("answer") is not None:
            accepted_answer = _nonempty_label(
                accepted["answer"],
                field_name="prediction.accepted.answer",
                source=source,
            )
            if accepted_answer != prediction:
                raise EvaluationError(
                    f"{source}: result.answer={prediction!r} disagrees with "
                    f"prediction.accepted.answer={accepted_answer!r}"
                )

    correct = prediction == observed
    for block_name, block in (("evaluation", evaluation), ("result", result)):
        if "correct" not in block:
            continue
        reported = block["correct"]
        if not isinstance(reported, bool):
            raise EvaluationError(
                f"{source}: {block_name}.correct must be boolean when present"
            )
        if reported != correct:
            raise EvaluationError(
                f"{source}: {block_name}.correct={reported} disagrees with "
                f"derived correctness ({prediction!r} == {observed!r})"
            )

    return Trajectory(
        source=source,
        prediction=prediction,
        observed=observed,
        correct=correct,
    )


def _failure_stage(document: Mapping[str, Any]) -> str:
    failure = document.get("failure")
    if isinstance(failure, Mapping):
        stage = failure.get("stage")
        if isinstance(stage, str) and stage.strip():
            return stage.strip()
    status = document.get("status")
    return f"status:{status}" if isinstance(status, str) else "unknown"


def _load_experiment(path: Path, *, name: str | None = None) -> Experiment:
    path = path.expanduser().resolve()
    patients_dir = path / "patients"
    if not patients_dir.is_dir():
        # Accept a direct path to patients for a convenient reference CLI.
        if path.name == "patients" and path.is_dir():
            patients_dir = path
            experiment_path = path.parent
        else:
            raise EvaluationError(f"{path}: expected a patients/ directory")
    else:
        experiment_path = path

    case_dirs = sorted(
        item for item in patients_dir.iterdir() if item.is_dir() and item.name.startswith("case_")
    )
    if not case_dirs:
        raise EvaluationError(f"{patients_dir}: no case_* patient directories found")

    patients: dict[str, Patient] = {}
    for case_dir in case_dirs:
        patient_id = case_dir.name
        patient = Patient(patient_id=patient_id)
        trajectory_paths = sorted(case_dir.glob("trajectory_*.json"))

        for source in trajectory_paths:
            patient.attempted += 1
            try:
                raw = source.read_text(encoding="utf-8")
                document = json.loads(raw)
            except (OSError, json.JSONDecodeError) as exc:
                raise EvaluationError(f"{source}: cannot read valid JSON: {exc}") from exc
            if not isinstance(document, Mapping):
                raise EvaluationError(f"{source}: top-level JSON must be an object")

            case_id = document.get("case_id")
            if case_id != patient_id:
                raise EvaluationError(
                    f"{source}: case_id={case_id!r} does not match directory "
                    f"patient ID {patient_id!r}"
                )

            suffix = source.stem.removeprefix("trajectory_")
            trajectory_number = document.get("trajectory_number")
            if suffix.isdigit() and trajectory_number != int(suffix):
                raise EvaluationError(
                    f"{source}: trajectory_number={trajectory_number!r} does not "
                    f"match filename number {int(suffix)}"
                )

            status = document.get("status")
            if not isinstance(status, str) or not status.strip():
                raise EvaluationError(f"{source}: status must be a non-empty string")

            observed = _optional_observed_label(document, source)
            if observed is not None:
                if patient.observed is None:
                    patient.observed = observed
                elif patient.observed != observed:
                    raise EvaluationError(
                        f"{case_dir}: observed label is not constant within patient "
                        f"({patient.observed!r} versus {observed!r} in {source.name})"
                    )

            if status == "completed":
                trajectory = _completed_trajectory(document, source, observed)
                patient.trajectories.append(trajectory)
            else:
                patient.failure_stages[_failure_stage(document)] += 1

        patients[patient_id] = patient

    return Experiment(
        name=name or experiment_path.name,
        path=experiment_path,
        patients=patients,
    )


def _discover_experiments(output_root: Path) -> list[Experiment]:
    output_root = output_root.expanduser().resolve()
    if not output_root.is_dir():
        raise EvaluationError(f"{output_root}: output root is not a directory")
    candidates = sorted(
        child
        for child in output_root.iterdir()
        if child.is_dir() and (child / "patients").is_dir()
    )
    if not candidates:
        raise EvaluationError(
            f"{output_root}: no immediate experiment subdirectories containing "
            "patients/ were found"
        )
    # Preserve the child directory name even when an experiment is a symlink.
    return [_load_experiment(path, name=path.name) for path in candidates]


def _validate_observed_labels_across_runs(
    experiments: Iterable[Experiment],
) -> None:
    """Ensure one patient ID never maps to different ground-truth labels."""

    registry: dict[str, tuple[str, Path]] = {}
    for experiment in experiments:
        for patient_id, patient in experiment.patients.items():
            if patient.observed is None:
                continue
            previous = registry.get(patient_id)
            if previous is None:
                registry[patient_id] = (patient.observed, experiment.path)
                continue
            previous_label, previous_path = previous
            if patient.observed != previous_label:
                raise EvaluationError(
                    f"patient {patient_id}: observed label differs across runs: "
                    f"{previous_label!r} in {previous_path} versus "
                    f"{patient.observed!r} in {experiment.path}"
                )


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _classification_metrics(
    trajectories: Sequence[Trajectory],
) -> dict[str, Any]:
    labels = sorted(
        {trajectory.observed for trajectory in trajectories}
        | {trajectory.prediction for trajectory in trajectories}
    )
    observed_labels = sorted({trajectory.observed for trajectory in trajectories})
    per_class: dict[str, dict[str, int | float | None]] = {}
    f1_values: list[float] = []
    recalls: list[float] = []

    for label in labels:
        true_positive = sum(
            item.observed == label and item.prediction == label
            for item in trajectories
        )
        false_positive = sum(
            item.observed != label and item.prediction == label
            for item in trajectories
        )
        false_negative = sum(
            item.observed == label and item.prediction != label
            for item in trajectories
        )
        support = true_positive + false_negative
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, support)
        if precision is None or recall is None or precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1_values.append(f1)
        if label in observed_labels:
            # support is necessarily nonzero for an observed label.
            recalls.append(0.0 if recall is None else recall)
        per_class[label] = {
            "support": support,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "labels": labels,
        "macro_f1": statistics.fmean(f1_values) if f1_values else None,
        "balanced_accuracy": statistics.fmean(recalls) if recalls else None,
        "per_class": per_class,
    }


def _majority_outcome(patient: Patient) -> tuple[str | None, bool, bool]:
    """Return (prediction, correct, abstained).

    A prediction requires strictly more than half of completed trajectories.
    Every tie or non-majority plurality is an abstention scored incorrect,
    avoiding label-order or ground-truth-dependent tie breaking.
    """

    if not patient.trajectories or patient.observed is None:
        return None, False, False
    counts = Counter(item.prediction for item in patient.trajectories)
    maximum = max(counts.values())
    winners = [label for label, count in counts.items() if count == maximum]
    if len(winners) != 1 or maximum * 2 <= len(patient.trajectories):
        return None, False, True
    prediction = winners[0]
    return prediction, prediction == patient.observed, False


def _majority_classification_metrics(
    patients: Sequence[Patient],
) -> dict[str, Any]:
    """Class metrics at the patient level, scoring ties as abstaining errors."""
    outcomes = [
        (_majority_outcome(patient)[0], patient.observed)
        for patient in patients
        if patient.observed is not None
    ]
    labels = sorted(
        {
            observed
            for _, observed in outcomes
            if isinstance(observed, str)
        }
    )
    per_class: dict[str, dict[str, int | float | None]] = {}
    f1_values: list[float] = []
    recalls: list[float] = []
    for label in labels:
        true_positive = sum(
            prediction == label and observed == label
            for prediction, observed in outcomes
        )
        false_positive = sum(
            prediction == label and observed != label
            for prediction, observed in outcomes
        )
        false_negative = sum(
            observed == label and prediction != label
            for prediction, observed in outcomes
        )
        support = true_positive + false_negative
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, support)
        if precision is None or recall is None or precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1_values.append(f1)
        recalls.append(0.0 if recall is None else recall)
        per_class[label] = {
            "support": support,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "labels": labels,
        "macro_f1": statistics.fmean(f1_values) if f1_values else None,
        "balanced_accuracy": statistics.fmean(recalls) if recalls else None,
        "per_class": per_class,
    }


def _experiment_metrics(experiment: Experiment) -> dict[str, Any]:
    trajectories = experiment.trajectories
    attempted = sum(patient.attempted for patient in experiment.patients.values())
    evaluated = len(trajectories)
    failed = attempted - evaluated
    correct = sum(item.correct for item in trajectories)
    classification = _classification_metrics(trajectories)

    evaluable_patients = [
        patient for patient in experiment.patients.values() if patient.trajectories
    ]
    total_patients = len(experiment.patients)
    patient_mean_values = [
        patient.mean_accuracy
        for patient in evaluable_patients
        if patient.mean_accuracy is not None
    ]
    majority = [_majority_outcome(patient) for patient in evaluable_patients]
    majority_classification = _majority_classification_metrics(
        evaluable_patients
    )
    majority_correct = sum(correct_value for _, correct_value, _ in majority)
    majority_abstentions = sum(abstained for _, _, abstained in majority)
    failure_stages: Counter[str] = Counter()
    for patient in experiment.patients.values():
        failure_stages.update(patient.failure_stages)

    return {
        "name": experiment.name,
        "path": str(experiment.path),
        "trajectory": {
            "attempted": attempted,
            "evaluated": evaluated,
            "correct": correct,
            "failed": failed,
            "accuracy": _safe_ratio(correct, evaluated),
            "coverage": _safe_ratio(evaluated, attempted),
            "failure_rate": _safe_ratio(failed, attempted),
            "macro_f1": classification["macro_f1"],
            "balanced_accuracy": classification["balanced_accuracy"],
        },
        "patient": {
            "total": total_patients,
            "evaluable": len(evaluable_patients),
            "without_evaluable_trajectory": total_patients - len(evaluable_patients),
            "coverage": _safe_ratio(len(evaluable_patients), total_patients),
            "failure_rate": _safe_ratio(
                total_patients - len(evaluable_patients), total_patients
            ),
            "mean_accuracy": (
                statistics.fmean(patient_mean_values)
                if patient_mean_values
                else None
            ),
            "majority_vote_correct": majority_correct,
            "majority_vote_accuracy": _safe_ratio(
                majority_correct, len(evaluable_patients)
            ),
            "majority_vote_abstentions": majority_abstentions,
            "majority_vote_abstention_rate": _safe_ratio(
                majority_abstentions, len(evaluable_patients)
            ),
            "majority_vote_macro_f1": majority_classification["macro_f1"],
            "majority_vote_balanced_accuracy": majority_classification[
                "balanced_accuracy"
            ],
            "majority_vote_per_class": majority_classification["per_class"],
        },
        "classification": classification,
        "failure_stages": dict(sorted(failure_stages.items())),
    }


def _patient_mean_accuracy(
    experiment: Experiment,
    patient_ids: Sequence[str],
) -> float:
    values = [
        experiment.patients[patient_id].mean_accuracy
        for patient_id in patient_ids
    ]
    if any(value is None for value in values):
        raise EvaluationError(
            "internal error: patient-mean accuracy received an unevaluable patient"
        )
    return statistics.fmean(float(value) for value in values)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise EvaluationError("cannot calculate a percentile of an empty sample")
    if not 0.0 <= probability <= 1.0:
        raise EvaluationError(f"invalid percentile probability: {probability}")
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


def _cluster_bootstrap_difference(
    experiment: Experiment,
    reference: Experiment,
    common_patient_ids: Sequence[str],
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    experiment_accuracy = _patient_mean_accuracy(
        experiment,
        common_patient_ids,
    )
    reference_accuracy = _patient_mean_accuracy(
        reference,
        common_patient_ids,
    )
    estimate = experiment_accuracy - reference_accuracy
    random_generator = random.Random(seed)
    patient_count = len(common_patient_ids)
    bootstrap_values: list[float] = []

    for _ in range(samples):
        sampled_ids = [
            common_patient_ids[random_generator.randrange(patient_count)]
            for _ in range(patient_count)
        ]
        difference = _patient_mean_accuracy(
            experiment, sampled_ids
        ) - _patient_mean_accuracy(reference, sampled_ids)
        bootstrap_values.append(difference)

    bootstrap_values.sort()
    return {
        "estimand": "unweighted patient-mean trajectory accuracy difference",
        "direction": "experiment_minus_reference",
        "experiment_accuracy": experiment_accuracy,
        "reference_accuracy": reference_accuracy,
        "difference": estimate,
        "confidence_level": 0.95,
        "ci_method": "patient-cluster percentile bootstrap",
        "ci_low": _percentile(bootstrap_values, 0.025),
        "ci_high": _percentile(bootstrap_values, 0.975),
        "bootstrap_standard_error": (
            statistics.stdev(bootstrap_values) if samples > 1 else 0.0
        ),
        "bootstrap_samples": samples,
        "seed": seed,
    }


def _mcnemar_exact(
    experiment: Experiment,
    reference: Experiment,
    common_patient_ids: Sequence[str],
) -> dict[str, Any]:
    both_correct = 0
    experiment_only = 0
    reference_only = 0
    both_incorrect = 0
    for patient_id in common_patient_ids:
        experiment_correct = _majority_outcome(
            experiment.patients[patient_id]
        )[1]
        reference_correct = _majority_outcome(reference.patients[patient_id])[1]
        if experiment_correct and reference_correct:
            both_correct += 1
        elif experiment_correct:
            experiment_only += 1
        elif reference_correct:
            reference_only += 1
        else:
            both_incorrect += 1

    result: dict[str, Any] = {
        "unit": "patient majority vote",
        "alternative": "two-sided",
        "both_correct": both_correct,
        "experiment_only_correct": experiment_only,
        "reference_only_correct": reference_only,
        "both_incorrect": both_incorrect,
        "discordant_pairs": experiment_only + reference_only,
        "odds_ratio": _safe_ratio(experiment_only, reference_only),
        "p_value": None,
        "scipy_available": False,
        "reason": None,
    }
    try:
        from scipy.stats import binomtest
    except ImportError:
        result["reason"] = (
            "scipy is unavailable; exact McNemar p_value is intentionally null"
        )
        return result

    result["scipy_available"] = True
    discordant = experiment_only + reference_only
    if discordant == 0:
        result["p_value"] = 1.0
    else:
        result["p_value"] = float(
            binomtest(
                experiment_only,
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    return result


def _comparison(
    experiment: Experiment,
    reference: Experiment,
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    experiment_evaluable = {
        patient_id
        for patient_id, patient in experiment.patients.items()
        if patient.trajectories and patient.observed is not None
    }
    reference_evaluable = {
        patient_id
        for patient_id, patient in reference.patients.items()
        if patient.trajectories and patient.observed is not None
    }
    common = sorted(experiment_evaluable & reference_evaluable)
    result: dict[str, Any] = {
        "reference_name": reference.name,
        "reference_path": str(reference.path),
        "common_evaluable_patients": len(common),
        "experiment_only_evaluable_patients": len(
            experiment_evaluable - reference_evaluable
        ),
        "reference_only_evaluable_patients": len(
            reference_evaluable - experiment_evaluable
        ),
        "cluster_bootstrap": None,
        "mcnemar_exact": None,
    }
    if not common:
        result["reason"] = "no common patients with evaluable trajectories"
        return result

    result["cluster_bootstrap"] = _cluster_bootstrap_difference(
        experiment,
        reference,
        common,
        seed=seed,
        samples=bootstrap_samples,
    )
    result["mcnemar_exact"] = _mcnemar_exact(experiment, reference, common)
    return result


def _apply_holm_correction(
    comparisons: Mapping[str, dict[str, Any]],
) -> None:
    """Add Holm-adjusted McNemar p-values across all available comparisons."""
    available: list[tuple[str, float]] = []
    for name, comparison in comparisons.items():
        mcnemar = comparison.get("mcnemar_exact")
        if not isinstance(mcnemar, dict):
            continue
        p_value = mcnemar.get("p_value")
        if isinstance(p_value, (int, float)) and math.isfinite(float(p_value)):
            available.append((name, float(p_value)))
    available.sort(key=lambda item: (item[1], item[0]))
    family_size = len(available)
    running_max = 0.0
    for rank, (name, p_value) in enumerate(available):
        adjusted = min(1.0, (family_size - rank) * p_value)
        running_max = max(running_max, adjusted)
        mcnemar = comparisons[name]["mcnemar_exact"]
        mcnemar["holm_adjusted_p_value"] = running_max
        mcnemar["holm_family_size"] = family_size


def _csv_rows(
    experiments: Sequence[Experiment],
    metrics: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, Mapping[str, Any]],
    reference: Experiment,
    reference_metrics: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    fields = [
        "role",
        "experiment",
        "path",
        "trajectory_attempted",
        "trajectory_evaluated",
        "trajectory_correct",
        "trajectory_failed",
        "trajectory_accuracy",
        "trajectory_coverage",
        "trajectory_failure_rate",
        "macro_f1",
        "balanced_accuracy",
        "patients_total",
        "patients_evaluable",
        "patient_coverage",
        "patient_failure_rate",
        "patient_mean_accuracy",
        "patient_majority_vote_accuracy",
        "patient_majority_vote_macro_f1",
        "patient_majority_vote_balanced_accuracy",
        "patient_majority_vote_abstentions",
        "reference",
        "common_evaluable_patients",
        "common_experiment_accuracy",
        "common_reference_accuracy",
        "accuracy_difference",
        "accuracy_difference_ci_low",
        "accuracy_difference_ci_high",
        "bootstrap_samples",
        "bootstrap_seed",
        "mcnemar_experiment_only_correct",
        "mcnemar_reference_only_correct",
        "mcnemar_discordant_pairs",
        "mcnemar_exact_p_value",
        "mcnemar_holm_adjusted_p_value",
        "mcnemar_scipy_available",
    ]

    def base_row(role: str, item: Mapping[str, Any]) -> dict[str, Any]:
        trajectory = item["trajectory"]
        patient = item["patient"]
        return {
            "role": role,
            "experiment": item["name"],
            "path": item["path"],
            "trajectory_attempted": trajectory["attempted"],
            "trajectory_evaluated": trajectory["evaluated"],
            "trajectory_correct": trajectory["correct"],
            "trajectory_failed": trajectory["failed"],
            "trajectory_accuracy": trajectory["accuracy"],
            "trajectory_coverage": trajectory["coverage"],
            "trajectory_failure_rate": trajectory["failure_rate"],
            "macro_f1": trajectory["macro_f1"],
            "balanced_accuracy": trajectory["balanced_accuracy"],
            "patients_total": patient["total"],
            "patients_evaluable": patient["evaluable"],
            "patient_coverage": patient["coverage"],
            "patient_failure_rate": patient["failure_rate"],
            "patient_mean_accuracy": patient["mean_accuracy"],
            "patient_majority_vote_accuracy": patient[
                "majority_vote_accuracy"
            ],
            "patient_majority_vote_macro_f1": patient[
                "majority_vote_macro_f1"
            ],
            "patient_majority_vote_balanced_accuracy": patient[
                "majority_vote_balanced_accuracy"
            ],
            "patient_majority_vote_abstentions": patient[
                "majority_vote_abstentions"
            ],
        }

    rows = [base_row("reference", reference_metrics)]
    rows[0]["reference"] = reference.name
    for experiment in experiments:
        row = base_row("ablation", metrics[experiment.name])
        comparison = comparisons[experiment.name]
        bootstrap = comparison.get("cluster_bootstrap") or {}
        mcnemar = comparison.get("mcnemar_exact") or {}
        row.update(
            {
                "reference": reference.name,
                "common_evaluable_patients": comparison[
                    "common_evaluable_patients"
                ],
                "common_experiment_accuracy": bootstrap.get(
                    "experiment_accuracy"
                ),
                "common_reference_accuracy": bootstrap.get(
                    "reference_accuracy"
                ),
                "accuracy_difference": bootstrap.get("difference"),
                "accuracy_difference_ci_low": bootstrap.get("ci_low"),
                "accuracy_difference_ci_high": bootstrap.get("ci_high"),
                "bootstrap_samples": bootstrap.get("bootstrap_samples"),
                "bootstrap_seed": bootstrap.get("seed"),
                "mcnemar_experiment_only_correct": mcnemar.get(
                    "experiment_only_correct"
                ),
                "mcnemar_reference_only_correct": mcnemar.get(
                    "reference_only_correct"
                ),
                "mcnemar_discordant_pairs": mcnemar.get("discordant_pairs"),
                "mcnemar_exact_p_value": mcnemar.get("p_value"),
                "mcnemar_holm_adjusted_p_value": mcnemar.get(
                    "holm_adjusted_p_value"
                ),
                "mcnemar_scipy_available": mcnemar.get("scipy_available"),
            }
        )
        rows.append(row)
    return fields, rows


def _render_csv(fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    return output.getvalue()


def _atomic_write_text(path: Path, text: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def evaluate(
    *,
    output_root: Path,
    reference_dir: Path,
    json_out: Path,
    csv_out: Path,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Run evaluation and atomically write JSON and CSV reports."""

    if bootstrap_samples < 1:
        raise EvaluationError("--bootstrap-samples must be at least 1")
    if json_out.expanduser().resolve() == csv_out.expanduser().resolve():
        raise EvaluationError("--json-out and --csv-out must be different paths")

    reference = _load_experiment(reference_dir)
    reference_path = reference.path.expanduser().resolve()
    experiments = [
        experiment
        for experiment in _discover_experiments(output_root)
        if experiment.path.expanduser().resolve() != reference_path
    ]
    names = [experiment.name for experiment in experiments]
    if len(set(names)) != len(names):
        raise EvaluationError(f"experiment names are not unique: {names}")
    _validate_observed_labels_across_runs([reference, *experiments])

    reference_metrics = _experiment_metrics(reference)
    metrics = {
        experiment.name: _experiment_metrics(experiment)
        for experiment in experiments
    }
    comparisons = {
        experiment.name: _comparison(
            experiment,
            reference,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
        )
        for experiment in experiments
    }
    _apply_holm_correction(comparisons)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "output_root": str(output_root.expanduser().resolve()),
            "reference_dir": str(reference_dir.expanduser().resolve()),
            "seed": seed,
            "bootstrap_samples": bootstrap_samples,
        },
        "methodology": {
            "metric_scale": "fractions in [0, 1], except signed accuracy differences",
            "trajectory_accuracy": (
                "correct completed trajectories / completed trajectories; "
                "non-completed attempts are reported through coverage and failure"
            ),
            "patient_mean_accuracy": (
                "unweighted mean of each evaluable patient's trajectory accuracy"
            ),
            "patient_majority_vote_accuracy": (
                "strict-majority prediction versus observed label; ties and "
                "non-majority pluralities are abstentions scored incorrect"
            ),
            "patient_majority_vote_classification": (
                "macro-F1, balanced accuracy, and per-class metrics at the patient "
                "level; abstentions remain false negatives for the observed class"
            ),
            "macro_f1": (
                "unweighted one-vs-rest F1 over the union of observed and predicted "
                "labels on completed trajectories"
            ),
            "balanced_accuracy": (
                "unweighted recall over observed labels on completed trajectories"
            ),
            "cluster_bootstrap": (
                "paired resampling of common patients with replacement; each arm's "
                "unweighted patient-mean trajectory accuracy is recomputed per sample"
            ),
            "mcnemar": (
                "two-sided exact test on paired strict-majority correctness; "
                "abstentions are incorrect; Holm correction spans all reported "
                "ablation-versus-reference comparisons with available p-values"
            ),
            "observed_label_validation": (
                "strict within-patient, within-trajectory, and across-run equality"
            ),
        },
        "reference": reference_metrics,
        "experiments": metrics,
        "comparisons_to_reference": comparisons,
    }

    fields, rows = _csv_rows(
        experiments, metrics, comparisons, reference, reference_metrics
    )
    json_text = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    csv_text = _render_csv(fields, rows)
    _atomic_write_text(json_out, json_text)
    _atomic_write_text(csv_out, csv_text)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every trajectory-RAG ablation under OUTPUT_ROOT and compare "
            "it with a reference run using patient-cluster-aware inference."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help=(
            "directory whose immediate experiment subdirectories each contain "
            "patients/case_*/trajectory_*.json"
        ),
    )
    parser.add_argument(
        "--reference-dir",
        required=True,
        type=Path,
        help=(
            "reference experiment directory containing patients/ (a direct "
            "patients/ path is also accepted)"
        ),
    )
    parser.add_argument(
        "--json-out",
        required=True,
        type=Path,
        help="destination for the complete machine-readable JSON report",
    )
    parser.add_argument(
        "--csv-out",
        required=True,
        type=Path,
        help="destination for the flat reference-and-experiment summary CSV",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260726,
        help="fixed random seed used independently for each paired bootstrap",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="number of paired patient-cluster bootstrap replicates",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = evaluate(
            output_root=args.output_root,
            reference_dir=args.reference_dir,
            json_out=args.json_out,
            csv_out=args.csv_out,
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
        )
    except (EvaluationError, OSError) as exc:
        parser.error(str(exc))

    experiment_count = len(report["experiments"])
    print(
        f"Evaluated {experiment_count} ablation experiment(s); "
        f"JSON: {args.json_out}; CSV: {args.csv_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
