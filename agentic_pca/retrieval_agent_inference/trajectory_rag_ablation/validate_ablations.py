#!/usr/bin/env python3
"""Validate the registry, leakage projections, determinism, and full equivalence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_pca.retrieval_agent_inference.infer import infer as base  # noqa: E402
from agentic_pca.retrieval_agent_inference.trajectory_rag import (  # noqa: E402
    CompletedTrajectoryRetriever,
)
from agentic_pca.retrieval_agent_inference.trajectory_rag_ablation.ablation_retriever import (  # noqa: E402
    AblationRetriever,
)
from agentic_pca.retrieval_agent_inference.trajectory_rag_ablation.ablation_specs import (  # noqa: E402
    DEFAULT_REGISTRY,
    load_experiment_registry,
)
from agentic_pca.retrieval_agent_inference.trajectory_rag_ablation.contexts import (  # noqa: E402
    FULL_FINAL_FIELDS,
    FULL_PLANNER_FIELDS,
    build_final_case_context,
    build_planner_context,
    final_system_prompt,
    planner_system_prompt,
)
from agentic_pca.retrieval_agent_inference.trajectory_rag_ablation.infer_ablation import (  # noqa: E402
    DEFAULT_SOURCE_RUN,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-file", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--trajectory-rag-dir", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--dataset", type=Path, default=base.DEFAULT_DATASET)
    parser.add_argument("--suv-dir", type=Path, default=base.DEFAULT_SUV_DIR)
    parser.add_argument("--ablation-seed", type=int, default=20260725)
    parser.add_argument(
        "--max-equivalence-patients",
        type=int,
        help="Limit only for quick smoke validation; omit for the full 249-patient audit.",
    )
    return parser.parse_args()


def contains_key(value: Any, forbidden: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        found.update(set(value) & forbidden)
        for child in value.values():
            found.update(contains_key(child, forbidden))
    elif isinstance(value, list):
        for child in value:
            found.update(contains_key(child, forbidden))
    return found


def assert_context_contract(
    experiment_id: str,
    planner: dict[str, Any] | None,
    final_cases: list[dict[str, Any]],
) -> None:
    if experiment_id == "no_rag" and (planner is not None or final_cases):
        raise AssertionError("no_rag exposes historical context")
    if experiment_id == "planner_only" and final_cases:
        raise AssertionError("planner_only exposes final cases")
    if experiment_id == "final_only" and planner is not None:
        raise AssertionError("final_only exposes planner history")
    if experiment_id == "no_historical_outcomes":
        found = contains_key(
            final_cases,
            {"historical_evaluation", "observed_management_category", "prediction_correct"},
        )
        if found:
            raise AssertionError(
                f"no_historical_outcomes leaks outcome fields: {sorted(found)}"
            )
    if experiment_id == "no_historical_prediction":
        found = contains_key(
            final_cases,
            {"historical_prediction", "prediction_correct"},
        )
        if found:
            raise AssertionError(
                f"no_historical_prediction leaks prediction fields: {sorted(found)}"
            )
    if experiment_id == "patient_context_only":
        allowed = {
            "evidence_id",
            "source_case_id",
            "source_trajectory_id",
            "retrieval_score",
            "retrieval_components",
            "historical_patient_input",
        }
        unexpected = {
            key
            for case in final_cases
            for key in case
            if key not in allowed
        }
        if unexpected:
            raise AssertionError(
                f"patient_context_only has unexpected fields: {sorted(unexpected)}"
            )


def main() -> int:
    args = parse_args()
    registry = load_experiment_registry(args.experiments_file)
    dataset = base.load_json(args.dataset)
    if not isinstance(dataset, dict):
        raise TypeError("Dataset must be an object keyed by patient")
    patient_keys = sorted(dataset)
    if args.max_equivalence_patients is not None:
        if args.max_equivalence_patients < 1:
            raise ValueError("--max-equivalence-patients must be positive")
        patient_keys = patient_keys[: args.max_equivalence_patients]

    core = CompletedTrajectoryRetriever.from_output_dir(
        args.trajectory_rag_dir,
        dataset_path=args.dataset,
    )
    ablation = AblationRetriever.from_output_dir(
        args.trajectory_rag_dir,
        dataset_path=args.dataset,
    )
    full = registry.resolve("full")
    equivalence_count = 0
    sample: tuple[str, dict[str, Any], set[str]] | None = None
    for patient_key in patient_keys:
        patient_input, _ = base.build_patient_input(dataset[patient_key])
        suv = base.load_suv_by_roi(patient_key, args.suv_dir)
        organs = set(base.common_suv_organs(suv))
        expected = core.search(
            patient_input,
            exclude_patient_key=patient_key,
            top_k=full.top_k,
            max_per_case=full.max_per_case,
            psa_weight=full.psa_weight,
            reason_chars=1200,
            available_organs=organs,
        )
        actual = ablation.search(
            patient_input,
            exclude_patient_key=patient_key,
            available_organs=organs,
            spec=full,
            seed=args.ablation_seed,
        )
        if actual != expected:
            raise AssertionError(
                f"Full ablation retrieval differs for patient {patient_key!r}"
            )
        expected_planner = {
            "organ_hints": expected["organ_hints"],
            "literature_query_hints": expected["literature_query_hints"],
            "similar_patient_inputs": [
                {
                    "source_trajectory_id": item["source_trajectory_id"],
                    "retrieval_score": item["retrieval_score"],
                    "historical_patient_input": item[
                        "historical_patient_input"
                    ],
                    "historical_evidence_selection": item[
                        "historical_evidence_selection"
                    ],
                    "historical_literature_sources": [
                        {"source": passage["source"], "page": passage["page"]}
                        for passage in item["historical_literature"]
                    ],
                }
                for item in expected["results"]
            ],
        }
        if build_planner_context(actual, full) != expected_planner:
            raise AssertionError(
                f"Full planner context differs for patient {patient_key!r}"
            )
        full_cases, _ = build_final_case_context(
            actual,
            full,
            current_case_id=base.stable_case_id(
                patient_key,
                "retrieval-agent-inference-v1",
            ),
            ablation_seed=args.ablation_seed,
        )
        if full_cases != expected["results"]:
            raise AssertionError(
                f"Full final CASE context differs for patient {patient_key!r}"
            )
        equivalence_count += 1
        if sample is None:
            sample = (patient_key, patient_input, organs)

    if sample is None:
        raise ValueError("No patient was selected for validation")
    if (
        planner_system_prompt(full)
        != base.TRAJECTORY_RAG_EVIDENCE_SELECTION_SYSTEM_PROMPT
    ):
        raise AssertionError("Full planner prompt differs from production")
    if final_system_prompt(full) != base.TRAJECTORY_RAG_FINAL_PREDICTION_SYSTEM_PROMPT:
        raise AssertionError("Full final prompt differs from production")
    patient_key, patient_input, organs = sample
    projection_checks = 0
    for spec in registry.experiments.values():
        if (
            spec.planner_enabled
            and frozenset(spec.planner_fields) == FULL_PLANNER_FIELDS
            and planner_system_prompt(spec)
            != base.TRAJECTORY_RAG_EVIDENCE_SELECTION_SYSTEM_PROMPT
        ):
            raise AssertionError(f"{spec.id} confounds retrieval with planner prompt")
        if (
            spec.final_enabled
            and frozenset(spec.final_fields) == FULL_FINAL_FIELDS
            and final_system_prompt(spec)
            != base.TRAJECTORY_RAG_FINAL_PREDICTION_SYSTEM_PROMPT
        ):
            raise AssertionError(f"{spec.id} confounds retrieval with final prompt")
        retrieval = None
        if spec.use_retrieval:
            retrieval = ablation.search(
                patient_input,
                exclude_patient_key=patient_key,
                available_organs=organs,
                spec=spec,
                seed=args.ablation_seed,
            )
        planner = build_planner_context(retrieval, spec)
        final_cases, _ = build_final_case_context(
            retrieval,
            spec,
            current_case_id=base.stable_case_id(
                patient_key,
                "retrieval-agent-inference-v1",
            ),
            ablation_seed=args.ablation_seed,
        )
        assert_context_contract(spec.id, planner, final_cases)
        if spec.id == "random_retrieval":
            repeated = ablation.search(
                patient_input,
                exclude_patient_key=patient_key,
                available_organs=organs,
                spec=spec,
                seed=args.ablation_seed,
            )
            if repeated != retrieval:
                raise AssertionError("random_retrieval is not deterministic")
        if retrieval is not None:
            source_cases = {item["source_case_id"] for item in retrieval["results"]}
            current_case = base.stable_case_id(
                patient_key,
                "retrieval-agent-inference-v1",
            )
            if current_case in source_cases:
                raise AssertionError(f"{spec.id} violates leave-one-patient-out")
        projection_checks += 1

    print(
        json.dumps(
            {
                "status": "validation_ok",
                "registry_schema_version": registry.schema_version,
                "registry_sha256": registry.manifest_sha256,
                "experiments": len(registry.experiments),
                "full_equivalence_patients": equivalence_count,
                "projection_contracts_checked": projection_checks,
                "corpus": ablation.metadata(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
