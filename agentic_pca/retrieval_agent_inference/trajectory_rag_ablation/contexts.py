"""Construct stage-specific, leakage-auditable historical contexts."""

from __future__ import annotations

import copy
from typing import Any

from agentic_pca.retrieval_agent_inference.infer import infer as base

from .ablation_specs import (
    ALLOWED_FINAL_FIELDS,
    ALLOWED_PLANNER_FIELDS,
    AblationSpec,
)


FULL_FINAL_FIELDS = frozenset(ALLOWED_FINAL_FIELDS)
FULL_PLANNER_FIELDS = frozenset(ALLOWED_PLANNER_FIELDS)


def build_planner_context(
    retrieval: dict[str, Any] | None,
    spec: AblationSpec,
) -> dict[str, Any] | None:
    """Project retrieval output to exactly what the evidence planner may see."""
    if retrieval is None or not spec.planner_enabled:
        return None
    fields = set(spec.planner_fields)
    context: dict[str, Any] = {}
    if "organ_hints" in fields:
        context["organ_hints"] = copy.deepcopy(retrieval["organ_hints"])
    if "literature_query_hints" in fields:
        context["literature_query_hints"] = copy.deepcopy(
            retrieval["literature_query_hints"]
        )
    per_case_fields = {
        "patient_input",
        "evidence_selection",
        "literature_sources",
    }
    if fields & per_case_fields:
        similar_inputs = []
        for item in retrieval["results"]:
            projected = {
                "source_trajectory_id": item["source_trajectory_id"],
                "retrieval_score": item["retrieval_score"],
            }
            if "patient_input" in fields:
                projected["historical_patient_input"] = copy.deepcopy(
                    item["historical_patient_input"]
                )
            if "evidence_selection" in fields:
                projected["historical_evidence_selection"] = copy.deepcopy(
                    item["historical_evidence_selection"]
                )
            if "literature_sources" in fields:
                projected["historical_literature_sources"] = [
                    {"source": passage["source"], "page": passage["page"]}
                    for passage in item["historical_literature"]
                ]
            similar_inputs.append(projected)
        context["similar_patient_inputs"] = similar_inputs
    return context


def build_final_case_context(
    retrieval: dict[str, Any] | None,
    spec: AblationSpec,
    *,
    current_case_id: str,
    ablation_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project cases for final prediction and apply deterministic label controls."""
    if retrieval is None or not spec.final_enabled:
        return [], {"outcome_permutation_fingerprint": None}
    fields = set(spec.final_fields)
    if fields == FULL_FINAL_FIELDS and not spec.permute_outcomes:
        return copy.deepcopy(retrieval["results"]), {
            "outcome_permutation_fingerprint": None
        }
    cases: list[dict[str, Any]] = []
    for item in retrieval["results"]:
        projected: dict[str, Any] = {
            "evidence_id": item["evidence_id"],
            "source_case_id": item["source_case_id"],
            "source_trajectory_id": item["source_trajectory_id"],
            "retrieval_score": item["retrieval_score"],
            "retrieval_components": copy.deepcopy(item["retrieval_components"]),
        }
        if "patient_input" in fields:
            projected["historical_patient_input"] = copy.deepcopy(
                item["historical_patient_input"]
            )
        if "evidence_selection" in fields:
            projected["historical_evidence_selection"] = copy.deepcopy(
                item["historical_evidence_selection"]
            )
        if "retrieved_suv_organs" in fields:
            projected["historical_retrieved_suv_organs"] = copy.deepcopy(
                item["historical_retrieved_suv_organs"]
            )
        if "literature" in fields:
            projected["historical_literature"] = copy.deepcopy(
                item["historical_literature"]
            )
        prediction: dict[str, Any] = {}
        if "prediction_answer" in fields:
            prediction["answer"] = item["historical_prediction"]["answer"]
        if "prediction_reason" in fields:
            prediction["reason"] = item["historical_prediction"]["reason"]
        if prediction:
            projected["historical_prediction"] = prediction
        evaluation: dict[str, Any] = {}
        if "observed_outcome" in fields:
            evaluation["observed_management_category"] = item[
                "historical_evaluation"
            ]["observed_management_category"]
        if "prediction_correctness" in fields:
            evaluation["prediction_correct"] = item["historical_evaluation"][
                "prediction_correct"
            ]
        if evaluation:
            projected["historical_evaluation"] = evaluation
        cases.append(projected)

    if spec.permute_outcomes:
        for raw_item, case in zip(retrieval["results"], cases, strict=True):
            outcome = raw_item.get("permuted_observed_management_category")
            if not isinstance(outcome, str):
                raise ValueError(
                    "Permuted-outcome ablation is missing the retriever's "
                    "patient-level permuted label"
                )
            case["historical_evaluation"][
                "observed_management_category"
            ] = outcome
            prediction = case.get("historical_prediction", {}).get("answer")
            if "prediction_correct" in case["historical_evaluation"]:
                case["historical_evaluation"][
                    "prediction_correct"
                ] = prediction == outcome
    return cases, {
        "outcome_permutation_fingerprint": retrieval.get(
            "permutation_fingerprint"
        )
    }


def planner_system_prompt(spec: AblationSpec) -> str:
    if not spec.planner_enabled:
        return base.EVIDENCE_SELECTION_SYSTEM_PROMPT
    if frozenset(spec.planner_fields) == FULL_PLANNER_FIELDS:
        return base.TRAJECTORY_RAG_EVIDENCE_SELECTION_SYSTEM_PROMPT
    visible = ", ".join(spec.planner_fields)
    partial_context_prompt = base.EVIDENCE_SELECTION_SYSTEM_PROMPT.replace(
        "1. Use only the supplied pretreatment report, medical history, and PSA.",
        (
            "1. Use the current patient's supplied pretreatment report, medical "
            "history, and PSA together with the explicitly supplied enabled "
            "historical retrieval hints."
        ),
    )
    return (
        partial_context_prompt
        + f"""

	PRE-REGISTERED HISTORICAL-CONTEXT CONDITION
9. Only these historical fields are supplied: {visible}. Omitted fields are
   deliberately unavailable and must not be inferred.
10. Historical content comes only from other patients after leave-one-patient-out
    exclusion. Use it solely as a retrieval hint, never as proof about the current
    patient and never transfer another patient's measurements or findings."""
    )


def final_system_prompt(spec: AblationSpec) -> str:
    if not spec.final_enabled:
        return base.FINAL_PREDICTION_SYSTEM_PROMPT
    if frozenset(spec.final_fields) == FULL_FINAL_FIELDS:
        return base.TRAJECTORY_RAG_FINAL_PREDICTION_SYSTEM_PROMPT
    visible = ", ".join(spec.final_fields)
    return (
        base.FINAL_PREDICTION_SYSTEM_PROMPT
        + f"""

PRE-REGISTERED HISTORICAL-CONTEXT CONDITION
9. Each CASE item contains only these enabled historical fields: {visible}.
   Missing fields are deliberately unavailable; do not infer them.
10. CASE items come only from other patients after leave-one-patient-out
    exclusion. They are analogical evidence, not proof. Never transfer another
    patient's measurements, lesions, or treatment to the current patient.
11. Cite at least one supplied CASE evidence ID in addition to current-patient
    SUV and literature evidence IDs. HIST-SUV and HIST-LIT tokens, if present
    in historical text, are not valid current evidence IDs."""
    )


def retrieval_audit_view(
    retrieval: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Record selection provenance without restoring fields hidden by an ablation."""
    if retrieval is None:
        return None
    metadata_keys = (
        "source_dir",
        "corpus_fingerprint",
        "query_fields",
        "excluded_current_patient",
        "excluded_completed_trajectories",
        "eligible_completed_trajectories",
        "top_k",
        "max_trajectories_per_case",
        "psa_weight",
        "ranking",
        "corpus_filter",
    )
    audit = {
        key: copy.deepcopy(retrieval[key])
        for key in metadata_keys
        if key in retrieval
    }
    audit["selected"] = [
        {
            "evidence_id": item["evidence_id"],
            "source_case_id": item["source_case_id"],
            "source_trajectory_id": item["source_trajectory_id"],
            "retrieval_score": item["retrieval_score"],
        }
        for item in retrieval["results"]
    ]
    audit["retrieval_fingerprint"] = base.sha256_json(retrieval)
    return audit
