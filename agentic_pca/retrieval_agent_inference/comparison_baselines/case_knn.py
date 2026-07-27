#!/usr/bin/env python3
"""Strict leave-one-patient-out, label-voting case k-NN baseline.

This is intentionally a non-generative comparator.  It reuses the production
trajectory retriever and the base inference pipeline's treatment-blind patient
input, but predicts only from the observed management labels of the five
retrieved reference patients.  The target patient's ``Treatment`` value is not
read until every target prediction has been frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_pca.retrieval_agent_inference.infer.infer import (  # noqa: E402
    DEFAULT_DATASET,
    LABEL_MAPPING_VERSION,
    MANAGEMENT_CATEGORIES,
    SCHEMA_VERSION,
    atomic_write_json,
    build_patient_input,
    load_json,
    rebuild_summary,
    sha256_file,
    sha256_json,
    stable_case_id,
    treatment_to_category,
    utc_now,
)
from agentic_pca.retrieval_agent_inference.trajectory_rag import (  # noqa: E402
    CompletedTrajectoryRetriever,
)


BASELINE_ID = "case_knn"
DEFAULT_SOURCE_RUN = (
    PROJECT_ROOT
    / "agentic_pca/retrieval_agent_inference/outputs/full_run_Qwen3.5_9B"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "agentic_pca/retrieval_agent_inference/outputs"
    / "comparison_baselines_Qwen3.5-9B/case_knn"
)

TOP_K = 5
MAX_PER_CASE = 1
PSA_WEIGHT = 0.15
REASON_CHARS = 20
TIE_ABS_TOLERANCE = 1e-12

TIE_BREAK_POLICY = (
    "largest similarity-weight sum",
    "largest unweighted neighbor count among weight-tied classes",
    "class of the nearest-ranked neighbor among classes still tied",
    "fixed MANAGEMENT_CATEGORIES order as a final deterministic fallback",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="same patient dataset used by the source inference run",
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        default=DEFAULT_SOURCE_RUN,
        help="completed base-inference run used as the labelled case memory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="new run directory; the source run is always read-only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "load and audit the complete corpus, execute every LOO retrieval and "
            "vote, but do not read target Treatment values or write outputs"
        ),
    )
    return parser


def _load_dataset(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, dict) or not payload:
        raise TypeError("Dataset must be a non-empty object keyed by patient.")
    records: dict[str, dict[str, Any]] = {}
    required = {"Report", "Medical History", "PSA", "Treatment"}
    for patient_key, value in payload.items():
        if not isinstance(patient_key, str) or not patient_key:
            raise TypeError("Dataset patient keys must be non-empty strings.")
        if not isinstance(value, dict):
            raise TypeError(f"Dataset record for {patient_key!r} must be an object.")
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(
                f"Dataset record for {patient_key!r} is missing fields: {missing}"
            )
        records[patient_key] = value
    return records


def _source_case_id_salt(source_run: Path) -> str:
    config_path = source_run / "config.json"
    config = load_json(config_path)
    if not isinstance(config, Mapping):
        raise TypeError(f"{config_path} must contain an object.")
    inference_config = config.get("inference_config")
    if not isinstance(inference_config, Mapping):
        raise ValueError(f"{config_path} lacks inference_config.")
    source_label_mapping = inference_config.get("label_mapping_version")
    if source_label_mapping != LABEL_MAPPING_VERSION:
        raise ValueError(
            f"{config_path} uses label mapping {source_label_mapping!r}, expected "
            f"{LABEL_MAPPING_VERSION!r}."
        )
    salt = inference_config.get("case_id_salt")
    if not isinstance(salt, str) or not salt:
        raise ValueError(f"{config_path} lacks a non-empty case_id_salt.")
    return salt


def _validate_separate_paths(source_run: Path, output_dir: Path) -> None:
    source = source_run.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError(
            "--source-run and --output-dir must be separate, non-nested directories."
        )


def _compact_neighbors(
    retrieval: Mapping[str, Any],
    *,
    target_case_id: str,
) -> list[dict[str, Any]]:
    if retrieval.get("excluded_current_patient") is not True:
        raise ValueError("Retriever did not record current-patient exclusion.")
    raw_results = retrieval.get("results")
    if not isinstance(raw_results, list) or len(raw_results) != TOP_K:
        raise ValueError(
            f"LOO retrieval must return exactly {TOP_K} cases; got "
            f"{len(raw_results) if isinstance(raw_results, list) else 'invalid'}."
        )

    neighbors: list[dict[str, Any]] = []
    source_case_ids: set[str] = set()
    for rank, raw in enumerate(raw_results, start=1):
        if not isinstance(raw, Mapping):
            raise TypeError(f"Retrieved result {rank} must be an object.")
        source_case_id = raw.get("source_case_id")
        source_trajectory_id = raw.get("source_trajectory_id")
        evidence_id = raw.get("evidence_id")
        if not isinstance(source_case_id, str) or not source_case_id:
            raise ValueError(f"Retrieved result {rank} has an invalid source_case_id.")
        if source_case_id == target_case_id:
            raise ValueError(
                f"Strict LOO violation: target {target_case_id} retrieved itself."
            )
        if source_case_id in source_case_ids:
            raise ValueError(
                f"max_per_case=1 violation: duplicate source case {source_case_id}."
            )
        source_case_ids.add(source_case_id)
        if not isinstance(source_trajectory_id, str) or not source_trajectory_id:
            raise ValueError(
                f"Retrieved result {rank} has an invalid source_trajectory_id."
            )
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError(f"Retrieved result {rank} has an invalid evidence_id.")

        score = raw.get("retrieval_score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or float(score) < 0
        ):
            raise ValueError(
                f"Retrieved result {rank} has an invalid retrieval_score: {score!r}"
            )
        evaluation = raw.get("historical_evaluation")
        if not isinstance(evaluation, Mapping):
            raise ValueError(
                f"Retrieved result {rank} lacks historical_evaluation."
            )
        label = evaluation.get("observed_management_category")
        if label not in MANAGEMENT_CATEGORIES:
            raise ValueError(
                f"Retrieved result {rank} has an invalid observed label: {label!r}"
            )
        components = raw.get("retrieval_components")
        if not isinstance(components, Mapping):
            raise ValueError(
                f"Retrieved result {rank} lacks retrieval_components."
            )
        neighbors.append(
            {
                "rank": rank,
                "evidence_id": evidence_id,
                "source_case_id": source_case_id,
                "source_trajectory_id": source_trajectory_id,
                "retrieval_score": float(score),
                "retrieval_components": dict(components),
                "observed_management_category": label,
            }
        )
    return neighbors


def _weight_tied(value: float, maximum: float) -> bool:
    return math.isclose(
        value,
        maximum,
        rel_tol=0.0,
        abs_tol=TIE_ABS_TOLERANCE,
    )


def _weighted_vote(
    neighbors: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    if not neighbors:
        raise ValueError("Cannot vote without retrieved neighbors.")

    weight_lists: dict[str, list[float]] = {
        category: [] for category in MANAGEMENT_CATEGORIES
    }
    counts: Counter[str] = Counter()
    for neighbor in neighbors:
        label = str(neighbor["observed_management_category"])
        weight = float(neighbor["retrieval_score"])
        weight_lists[label].append(weight)
        counts[label] += 1

    weight_sums = {
        category: math.fsum(weight_lists[category])
        for category in MANAGEMENT_CATEGORIES
    }
    maximum_weight = max(weight_sums.values())
    weight_tied = [
        category
        for category in MANAGEMENT_CATEGORIES
        if _weight_tied(weight_sums[category], maximum_weight)
    ]
    initial_weight_tie = list(weight_tied)
    resolution = "similarity_weight_sum"

    candidates = weight_tied
    if len(candidates) > 1:
        maximum_count = max(counts[category] for category in candidates)
        candidates = [
            category
            for category in candidates
            if counts[category] == maximum_count
        ]
        resolution = "unweighted_neighbor_count"

    if len(candidates) > 1:
        nearest_tied_label = next(
            str(neighbor["observed_management_category"])
            for neighbor in neighbors
            if neighbor["observed_management_category"] in candidates
        )
        candidates = [nearest_tied_label]
        resolution = "nearest_ranked_neighbor"

    if len(candidates) > 1:
        candidates = [
            next(
                category
                for category in MANAGEMENT_CATEGORIES
                if category in candidates
            )
        ]
        resolution = "fixed_category_order"

    answer = candidates[0]
    total_weight = math.fsum(weight_sums.values())
    normalized_weights = {
        category: (
            weight_sums[category] / total_weight if total_weight > 0 else None
        )
        for category in MANAGEMENT_CATEGORIES
    }
    decision = {
        "answer": answer,
        "weight_field": "retrieval_score",
        "class_weight_sums": weight_sums,
        "class_normalized_weights": normalized_weights,
        "class_neighbor_counts": {
            category: counts[category] for category in MANAGEMENT_CATEGORIES
        },
        "weight_tied_classes": initial_weight_tie,
        "tie_break_applied": len(initial_weight_tie) > 1,
        "resolution": resolution,
        "absolute_tie_tolerance": TIE_ABS_TOLERANCE,
        "tie_break_policy": list(TIE_BREAK_POLICY),
        "fixed_category_order": list(MANAGEMENT_CATEGORIES),
    }
    return answer, decision


def _retrieve_and_vote(
    retriever: CompletedTrajectoryRetriever,
    *,
    patient_key: str,
    patient_input: dict[str, Any],
    target_case_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    retrieval = retriever.search(
        patient_input,
        exclude_patient_key=patient_key,
        top_k=TOP_K,
        max_per_case=MAX_PER_CASE,
        psa_weight=PSA_WEIGHT,
        reason_chars=REASON_CHARS,
        # Organ hints are irrelevant to this non-generative classifier.  An
        # empty set changes neither candidate ranking nor selected cases.
        available_organs=set(),
    )
    if retrieval.get("top_k") != TOP_K:
        raise ValueError("Retriever metadata disagrees with fixed top_k.")
    if retrieval.get("max_trajectories_per_case") != MAX_PER_CASE:
        raise ValueError("Retriever metadata disagrees with fixed max_per_case.")
    if not math.isclose(
        float(retrieval.get("psa_weight", float("nan"))),
        PSA_WEIGHT,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("Retriever metadata disagrees with fixed PSA weight.")
    neighbors = _compact_neighbors(retrieval, target_case_id=target_case_id)
    _, decision = _weighted_vote(neighbors)
    return neighbors, decision


def _freeze_trajectory(
    *,
    case_id: str,
    patient_input: dict[str, Any],
    input_warnings: list[str],
    neighbors: list[dict[str, Any]],
    decision: dict[str, Any],
    config_fingerprint: str,
) -> dict[str, Any]:
    """Create a prediction-frozen record without accepting a target label."""
    answer = str(decision["answer"])
    reason = (
        f"Deterministic {decision['resolution']} over {len(neighbors)} strictly "
        "leave-one-patient-out reference cases."
    )
    trajectory_fingerprint = sha256_json(
        {
            "baseline_id": BASELINE_ID,
            "case_id": case_id,
            "trajectory_number": 1,
            "patient_input": patient_input,
            "neighbors": neighbors,
            "decision": decision,
            "inference_config_fingerprint": config_fingerprint,
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": f"{case_id}_trajectory_001",
        "case_id": case_id,
        "trajectory_number": 1,
        "trajectory_fingerprint": trajectory_fingerprint,
        "created_at_utc": utc_now(),
        "status": "prediction_frozen",
        "input": patient_input,
        "input_warnings": input_warnings,
        "case_knn": {
            "strict_leave_one_patient_out": True,
            "top_k": TOP_K,
            "max_per_case": MAX_PER_CASE,
            "psa_weight": PSA_WEIGHT,
            "reference_label_field": (
                "historical_evaluation.observed_management_category"
            ),
            "reference_predictions_used": False,
            "retrieved_cases": neighbors,
            "vote": decision,
        },
        "evidence_selection": None,
        "retrieved_evidence": {"similar_cases": neighbors},
        "prediction": {
            "attempts": [],
            "accepted": {
                "answer": answer,
                "reason": reason,
            },
        },
        "evaluation": None,
        "result": None,
        "failure": None,
        "provenance": {
            "baseline_id": BASELINE_ID,
            "label_mapping_version": LABEL_MAPPING_VERSION,
            "inference_config_fingerprint": config_fingerprint,
            "treatment_blind_generation": True,
            "target_outcome_blinded": True,
            "reference_outcomes_available": True,
            "target_treatment_access": "after_all_predictions_frozen",
        },
    }


def _attach_evaluation(
    frozen: dict[str, Any],
    treatment_supplier: Callable[[], Any],
) -> dict[str, Any]:
    """Reveal one target Treatment only after its prediction is frozen."""
    if frozen.get("status") != "prediction_frozen":
        raise ValueError("Evaluation requires a prediction_frozen trajectory.")
    accepted = frozen.get("prediction", {}).get("accepted")
    if not isinstance(accepted, Mapping):
        raise ValueError("Frozen trajectory lacks an accepted prediction.")

    # This is the first and only target-label access in the prediction path.
    observed_treatment = treatment_supplier()
    observed_category = treatment_to_category(observed_treatment)
    answer = str(accepted["answer"])
    correct = answer == observed_category
    frozen["evaluation"] = {
        "observed_treatment": observed_treatment,
        "observed_management_category": observed_category,
        "correct": correct,
        "correct_means": (
            "exact agreement with documented management, not optimal care"
        ),
    }
    frozen["result"] = {
        "answer": answer,
        "reason": accepted["reason"],
        "observed_treatment": observed_treatment,
        "observed_management_category": observed_category,
        "correct": correct,
    }
    frozen["status"] = "completed"
    return frozen


def _inference_config(
    *,
    dataset_path: Path,
    source_run: Path,
    case_id_salt: str,
    retriever_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    retrieval_code = (
        PROJECT_ROOT
        / "agentic_pca/retrieval_agent_inference/trajectory_rag/trajectory_rag.py"
    )
    base_infer_code = (
        PROJECT_ROOT
        / "agentic_pca/retrieval_agent_inference/infer/infer.py"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_id": BASELINE_ID,
        "method": (
            "strict LOO case retrieval followed by similarity-weighted voting "
            "over reference observed-management labels"
        ),
        "label_mapping_version": LABEL_MAPPING_VERSION,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "source_run": str(source_run),
        "source_config_fingerprint": retriever_metadata[
            "source_config_fingerprint"
        ],
        "source_corpus_fingerprint": retriever_metadata["corpus_fingerprint"],
        "case_id_salt": case_id_salt,
        "patient_input_builder": "infer.infer.build_patient_input",
        "patient_input_fields": ["report", "medical_history", "psa"],
        "retrieval": {
            "implementation": "CompletedTrajectoryRetriever.search",
            "query_fields": list(retriever_metadata["retrieval_fields"]),
            "top_k": TOP_K,
            "max_per_case": MAX_PER_CASE,
            "psa_weight": PSA_WEIGHT,
            "reference_label_field": (
                "historical_evaluation.observed_management_category"
            ),
            "reference_predictions_used": False,
        },
        "voting": {
            "method": "similarity-weighted class vote",
            "weight_field": "retrieval_score",
            "absolute_tie_tolerance": TIE_ABS_TOLERANCE,
            "tie_break_policy": list(TIE_BREAK_POLICY),
            "fixed_category_order": list(MANAGEMENT_CATEGORIES),
        },
        "case_knn_code_sha256": sha256_file(Path(__file__).resolve()),
        "trajectory_retriever_code_sha256": sha256_file(retrieval_code),
        "base_infer_code_sha256": sha256_file(base_infer_code),
        "target_treatment_access": "after_all_predictions_frozen",
    }


def _audit_existing_output(
    output_dir: Path,
    *,
    config_fingerprint: str,
) -> None:
    trajectories = sorted(
        (output_dir / "patients").glob("case_*/trajectory_*.json")
    )
    if not trajectories:
        return
    config_path = output_dir / "config.json"
    if not config_path.is_file():
        raise ValueError(
            f"{output_dir} contains trajectories but no config.json; use a new "
            "--output-dir."
        )
    existing = load_json(config_path)
    if (
        not isinstance(existing, Mapping)
        or existing.get("inference_config_fingerprint") != config_fingerprint
    ):
        raise ValueError(
            f"{output_dir} contains trajectories from a different configuration; "
            "use a new --output-dir."
        )
    raise ValueError(
        f"{output_dir} already contains {len(trajectories)} trajectory files. "
        "This deterministic baseline never overwrites completed results; use the "
        "existing run or choose a new --output-dir."
    )


def _dry_run_report(
    *,
    dataset: Mapping[str, dict[str, Any]],
    retriever: CompletedTrajectoryRetriever,
    case_id_salt: str,
    retriever_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    predicted: Counter[str] = Counter()
    tie_resolutions: Counter[str] = Counter()
    retrieved_source_cases: set[str] = set()
    for patient_key, record in dataset.items():
        patient_input, _ = build_patient_input(record)
        case_id = stable_case_id(patient_key, case_id_salt)
        neighbors, decision = _retrieve_and_vote(
            retriever,
            patient_key=patient_key,
            patient_input=patient_input,
            target_case_id=case_id,
        )
        predicted[str(decision["answer"])] += 1
        tie_resolutions[str(decision["resolution"])] += 1
        retrieved_source_cases.update(
            str(neighbor["source_case_id"]) for neighbor in neighbors
        )
    return {
        "status": "dry_run_ok",
        "baseline_id": BASELINE_ID,
        "dataset_patients": len(dataset),
        "source_corpus": dict(retriever_metadata),
        "retrieval": {
            "strict_leave_one_patient_out": True,
            "top_k": TOP_K,
            "max_per_case": MAX_PER_CASE,
            "psa_weight": PSA_WEIGHT,
        },
        "audited_target_patients": len(dataset),
        "unique_retrieved_source_cases": len(retrieved_source_cases),
        "predicted_distribution_without_target_label_access": dict(predicted),
        "decision_resolution_counts": dict(tie_resolutions),
        "target_treatment_values_read": 0,
        "outputs_written": False,
        "model_will_load": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = args.dataset.expanduser().resolve()
    source_run = args.source_run.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if not source_run.is_dir():
        raise FileNotFoundError(source_run)
    _validate_separate_paths(source_run, output_dir)

    dataset = _load_dataset(dataset_path)
    case_id_salt = _source_case_id_salt(source_run)
    retriever = CompletedTrajectoryRetriever.from_output_dir(
        source_run,
        dataset_path=dataset_path,
    )
    retriever_metadata = retriever.metadata()

    if args.dry_run:
        report = _dry_run_report(
            dataset=dataset,
            retriever=retriever,
            case_id_salt=case_id_salt,
            retriever_metadata=retriever_metadata,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return report

    inference_config = _inference_config(
        dataset_path=dataset_path,
        source_run=source_run,
        case_id_salt=case_id_salt,
        retriever_metadata=retriever_metadata,
    )
    config_fingerprint = sha256_json(inference_config)
    _audit_existing_output(
        output_dir,
        config_fingerprint=config_fingerprint,
    )

    # Phase 1: freeze every target prediction.  No target Treatment value is
    # accessed anywhere in this phase.
    frozen_by_patient: dict[str, dict[str, Any]] = {}
    case_mapping: dict[str, str] = {}
    for patient_key, record in dataset.items():
        patient_input, input_warnings = build_patient_input(record)
        case_id = stable_case_id(patient_key, case_id_salt)
        if case_id in case_mapping:
            raise ValueError(f"Case-ID collision for {case_id}.")
        case_mapping[case_id] = patient_key
        neighbors, decision = _retrieve_and_vote(
            retriever,
            patient_key=patient_key,
            patient_input=patient_input,
            target_case_id=case_id,
        )
        frozen_by_patient[patient_key] = _freeze_trajectory(
            case_id=case_id,
            patient_input=patient_input,
            input_warnings=input_warnings,
            neighbors=neighbors,
            decision=decision,
            config_fingerprint=config_fingerprint,
        )

    if len(frozen_by_patient) != len(dataset):
        raise RuntimeError("Not every target prediction was frozen.")

    # Phase 2: all predictions are immutable with respect to target labels.
    # Only now may each record's Treatment field be read for evaluation.
    completed: dict[str, dict[str, Any]] = {}
    for patient_key, frozen in frozen_by_patient.items():
        record = dataset[patient_key]
        completed[patient_key] = _attach_evaluation(
            frozen,
            lambda record=record: record["Treatment"],
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "last_invocation_at_utc": utc_now(),
        "inference_config_fingerprint": config_fingerprint,
        "inference_config": inference_config,
        "selection": {
            "selected_patients": len(dataset),
            "num_trajectories": 1,
        },
    }
    atomic_write_json(output_dir / "config.json", config_payload)
    atomic_write_json(
        output_dir / "patient_manifest.json",
        dict(sorted(case_mapping.items())),
    )
    for patient_key, trajectory in completed.items():
        case_id = stable_case_id(patient_key, case_id_salt)
        atomic_write_json(
            output_dir / "patients" / case_id / "trajectory_001.json",
            trajectory,
        )

    summary = rebuild_summary(output_dir)
    summary.update(
        {
            "baseline_id": BASELINE_ID,
            "strict_leave_one_patient_out": True,
            "top_k": TOP_K,
            "max_per_case": MAX_PER_CASE,
            "psa_weight": PSA_WEIGHT,
            "target_treatment_access": "after_all_predictions_frozen",
        }
    )
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Run directory: {output_dir}")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
