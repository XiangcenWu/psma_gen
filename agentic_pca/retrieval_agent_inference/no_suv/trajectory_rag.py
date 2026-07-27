"""Safe leave-one-patient-out retrieval over completed inference trajectories.

The retriever accepts either the legacy full-run trajectory schema or the
literature-only schema produced by ``no_suv/infer.py``.  Legacy structured
whole-organ measurements are never projected into a search result.  The
returned historical prediction is deliberately limited to its categorical
answer because a legacy free-text reason can repeat historical measurements.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize


PATIENT_INPUT_FIELDS = {"report", "medical_history", "psa"}
PSA_FIELDS = {"raw", "comparator", "value", "unit"}
MANAGEMENT_CATEGORIES = {
    "radical_prostatectomy",
    "systemic_treatment",
    "local_treatment",
    "other_examination",
}
SOURCE_EVIDENCE_MODE_LITERATURE = "literature_only"
SOURCE_EVIDENCE_MODE_LEGACY = "legacy_full_run"
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_CASE_ID = re.compile(r"case_[0-9a-f]{16}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        try:
            return json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid strict JSON in {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_dict(value: Any, description: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object in {path}")
    return value


def _require_list(
    value: Any,
    description: str,
    path: Path,
    *,
    nonempty: bool = True,
) -> list[Any]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "a non-empty list" if nonempty else "a list"
        raise ValueError(f"{description} must be {qualifier} in {path}")
    return value


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    description: str,
    path: Path,
) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        raise ValueError(
            f"{description} has invalid keys in {path}: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _require_nonempty_string(value: Any, description: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a non-empty string in {path}")
    return value


def _require_sha256(value: Any, description: str, path: Path) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase SHA-256 in {path}")
    return value


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _normalize_psa(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {
            "raw": value,
            "comparator": "unknown",
            "value": None,
            "unit": "ng/mL",
        }
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return {
            "raw": value,
            "comparator": "equal",
            "value": float(value),
            "unit": "ng/mL",
        }
    text = str(value).strip()
    match = re.fullmatch(r"(>=|>|<=|<)?\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return {
            "raw": value,
            "comparator": "unknown",
            "value": None,
            "unit": "ng/mL",
        }
    comparator = {
        None: "equal",
        ">": "greater_than",
        ">=": "greater_than_or_equal",
        "<": "less_than",
        "<=": "less_than_or_equal",
    }[match.group(1)]
    return {
        "raw": value,
        "comparator": comparator,
        "value": float(match.group(2)),
        "unit": "ng/mL",
    }


def _expected_patient_input(dataset_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "report": dataset_record.get("Report"),
        "medical_history": dataset_record.get("Medical History"),
        "psa": _normalize_psa(dataset_record.get("PSA")),
    }


def _validate_patient_input(
    value: Any,
    description: str,
    path: Path,
) -> dict[str, Any]:
    patient_input = _require_dict(value, description, path)
    _require_exact_keys(patient_input, PATIENT_INPUT_FIELDS, description, path)
    for field in ("report", "medical_history"):
        text = patient_input[field]
        if text is not None and not isinstance(text, str):
            raise ValueError(
                f"{description}.{field} must be a string or null in {path}"
            )
    psa = _require_dict(patient_input["psa"], f"{description}.psa", path)
    _require_exact_keys(psa, PSA_FIELDS, f"{description}.psa", path)
    if psa["comparator"] not in {
        "equal",
        "greater_than",
        "greater_than_or_equal",
        "less_than",
        "less_than_or_equal",
        "unknown",
    }:
        raise ValueError(f"{description}.psa.comparator is invalid in {path}")
    if psa["unit"] != "ng/mL":
        raise ValueError(f"{description}.psa.unit must be 'ng/mL' in {path}")
    if psa["value"] is not None:
        if not _is_finite_number(psa["value"]) or float(psa["value"]) < 0:
            raise ValueError(
                f"{description}.psa.value must be non-negative and finite in {path}"
            )
    return patient_input


def _stable_case_id(patient_key: str, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}:{patient_key}".encode("utf-8")).hexdigest()
    return f"case_{digest[:16]}"


def _numeric_psa(patient_input: dict[str, Any]) -> float | None:
    psa = patient_input["psa"]
    value = psa["value"]
    if not _is_finite_number(value):
        return None
    number = float(value)
    return number if number >= 0 else None


def _retrieval_text(patient_input: dict[str, Any]) -> str:
    """Build the treatment-column-blind report/history retrieval document."""
    return "\n".join(
        (
            f"Report: {patient_input.get('report') or ''}",
            f"Medical history: {patient_input.get('medical_history') or ''}",
        )
    )


def _literature_strategy_text(payload: dict[str, Any]) -> str:
    query = payload["evidence_selection"]["accepted"]["literature_query"]
    return f"Literature query: {query}"


def _validate_generation_attempts(
    value: Any,
    accepted: dict[str, Any],
    description: str,
    path: Path,
) -> None:
    attempts = _require_list(value, f"{description}.attempts", path)
    for expected_number, attempt_value in enumerate(attempts, 1):
        attempt = _require_dict(
            attempt_value,
            f"{description}.attempts[{expected_number - 1}]",
            path,
        )
        _require_exact_keys(
            attempt,
            {
                "attempt_number",
                "seed",
                "valid",
                "validation_errors",
                "raw_response",
            },
            f"{description}.attempts[{expected_number - 1}]",
            path,
        )
        if attempt["attempt_number"] != expected_number:
            raise ValueError(
                f"{description} attempt numbers are not consecutive in {path}"
            )
        if not _is_integer(attempt["seed"]):
            raise ValueError(f"{description} attempt seed must be integer in {path}")
        if not isinstance(attempt["valid"], bool):
            raise ValueError(f"{description} attempt valid must be boolean in {path}")
        errors = _require_list(
            attempt["validation_errors"],
            f"{description}.attempt.validation_errors",
            path,
            nonempty=False,
        )
        if any(not isinstance(error, str) or not error for error in errors):
            raise ValueError(
                f"{description} validation errors must be non-empty strings in {path}"
            )
        if attempt["valid"] == bool(errors):
            raise ValueError(
                f"{description} valid flag and validation errors disagree in {path}"
            )
        raw_response = _require_nonempty_string(
            attempt["raw_response"],
            f"{description}.attempt.raw_response",
            path,
        )
        if attempt["valid"]:
            try:
                raw_payload = json.loads(
                    raw_response,
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=_reject_nonfinite_json_constant,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"{description} valid raw response is not strict JSON in {path}"
                ) from exc
            if raw_payload != accepted:
                raise ValueError(
                    f"{description} valid raw response differs from accepted in {path}"
                )
    if not attempts[-1]["valid"] or any(
        bool(attempt["valid"]) for attempt in attempts[:-1]
    ):
        raise ValueError(
            f"{description} must stop at its first valid final attempt in {path}"
        )


def _validate_literature(
    value: Any,
    path: Path,
) -> tuple[list[dict[str, Any]], set[str]]:
    items = _require_list(value, "retrieved_evidence.literature", path)
    evidence_ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, item_value in enumerate(items, 1):
        item = _require_dict(
            item_value,
            f"retrieved_evidence.literature[{index - 1}]",
            path,
        )
        _require_exact_keys(
            item,
            {"evidence_id", "source", "page", "retrieval_score", "text"},
            f"retrieved_evidence.literature[{index - 1}]",
            path,
        )
        expected_id = f"LIT-{index:03d}"
        if item["evidence_id"] != expected_id:
            raise ValueError(
                f"literature evidence IDs must be consecutive from LIT-001 in {path}"
            )
        _require_nonempty_string(item["source"], "literature source", path)
        if not _is_integer(item["page"]) or item["page"] < 1:
            raise ValueError(f"literature page must be a positive integer in {path}")
        if not _is_finite_number(item["retrieval_score"]):
            raise ValueError(f"literature retrieval score is invalid in {path}")
        _require_nonempty_string(item["text"], "literature text", path)
        evidence_ids.add(expected_id)
        validated.append(item)
    return validated, evidence_ids


def _forbidden_structured_keys(value: Any, prefix: str = "") -> list[str]:
    """Find structured field names that could expose historical measurements."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            location = f"{prefix}.{key}" if prefix else str(key)
            if "suv" in str(key).lower():
                found.append(location)
            found.extend(_forbidden_structured_keys(child, location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_structured_keys(child, f"{prefix}[{index}]"))
    return found


def _validate_projected_similar_trajectories(value: Any, path: Path) -> set[str]:
    items = _require_list(
        value,
        "retrieved_evidence.similar_trajectories",
        path,
    )
    forbidden = _forbidden_structured_keys(items)
    if forbidden:
        raise ValueError(
            "retrieved_evidence.similar_trajectories contains forbidden "
            f"structured fields {forbidden} in {path}"
        )
    evidence_ids: set[str] = set()
    for index, item_value in enumerate(items, 1):
        item = _require_dict(
            item_value,
            f"retrieved_evidence.similar_trajectories[{index - 1}]",
            path,
        )
        expected_id = f"CASE-{index:03d}"
        if item.get("evidence_id") != expected_id:
            raise ValueError(
                f"similar-trajectory IDs must be consecutive from CASE-001 in {path}"
            )
        evidence_ids.add(expected_id)
    return evidence_ids


def _validate_legacy_measurement_collection(
    value: Any,
    selected_organs: list[str],
    path: Path,
) -> set[str]:
    """Validate legacy collection identity without projecting its measurements."""
    items = _require_list(value, "retrieved_evidence legacy measurements", path)
    evidence_ids: set[str] = set()
    retrieved_organs: list[str] = []
    for index, item_value in enumerate(items, 1):
        item = _require_dict(
            item_value,
            f"retrieved_evidence legacy measurements[{index - 1}]",
            path,
        )
        expected_id = f"SUV-{index:03d}"
        if item.get("evidence_id") != expected_id:
            raise ValueError(
                f"legacy measurement IDs must be consecutive from SUV-001 in {path}"
            )
        organ = _require_nonempty_string(
            item.get("organ"),
            "legacy measurement organ",
            path,
        )
        retrieved_organs.append(organ)
        evidence_ids.add(expected_id)
    if retrieved_organs != selected_organs:
        raise ValueError(
            f"legacy retrieved organs differ from accepted selection in {path}"
        )
    return evidence_ids


def _validate_prediction(
    value: Any,
    valid_evidence_ids: set[str],
    *,
    require_legacy_measurement: bool,
    require_case_evidence: bool,
    path: Path,
) -> dict[str, Any]:
    wrapper = _require_dict(value, "prediction", path)
    _require_exact_keys(wrapper, {"attempts", "accepted"}, "prediction", path)
    accepted = _require_dict(wrapper["accepted"], "prediction.accepted", path)
    _require_exact_keys(
        accepted,
        {"answer", "reason", "evidence_ids"},
        "prediction.accepted",
        path,
    )
    if accepted["answer"] not in MANAGEMENT_CATEGORIES:
        raise ValueError(f"prediction answer is invalid in {path}")
    reason = _require_nonempty_string(
        accepted["reason"],
        "prediction reason",
        path,
    )
    if len(reason) < 20:
        raise ValueError(f"prediction reason is too short in {path}")
    cited_ids = _require_list(
        accepted["evidence_ids"],
        "prediction.accepted.evidence_ids",
        path,
    )
    if any(not isinstance(item, str) or not item for item in cited_ids):
        raise ValueError(f"prediction evidence IDs are malformed in {path}")
    if len(set(cited_ids)) != len(cited_ids):
        raise ValueError(f"prediction evidence IDs contain duplicates in {path}")
    unknown = sorted(set(cited_ids) - valid_evidence_ids)
    if unknown:
        raise ValueError(f"prediction cites unknown evidence IDs {unknown} in {path}")
    if not any(item.startswith("LIT-") for item in cited_ids):
        raise ValueError(f"prediction must cite literature evidence in {path}")
    if require_legacy_measurement and not any(
        item.startswith("SUV-") for item in cited_ids
    ):
        raise ValueError(f"legacy prediction must cite measurement evidence in {path}")
    if require_case_evidence and not any(
        item.startswith("CASE-") for item in cited_ids
    ):
        raise ValueError(f"trajectory-RAG prediction must cite CASE evidence in {path}")
    _validate_generation_attempts(
        wrapper["attempts"],
        accepted,
        "prediction",
        path,
    )
    return accepted


def _validate_evaluation_and_result(
    evaluation_value: Any,
    result_value: Any,
    prediction: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    evaluation = _require_dict(evaluation_value, "evaluation", path)
    _require_exact_keys(
        evaluation,
        {
            "observed_treatment",
            "observed_management_category",
            "correct",
            "correct_means",
        },
        "evaluation",
        path,
    )
    observed = evaluation["observed_management_category"]
    if observed not in MANAGEMENT_CATEGORIES:
        raise ValueError(
            f"evaluation.observed_management_category is invalid in {path}"
        )
    if not isinstance(evaluation["correct"], bool):
        raise ValueError(f"evaluation.correct must be boolean in {path}")
    expected_correct = prediction["answer"] == observed
    if evaluation["correct"] != expected_correct:
        raise ValueError(f"evaluation.correct is inconsistent in {path}")
    _require_nonempty_string(
        evaluation["correct_means"],
        "evaluation.correct_means",
        path,
    )

    result = _require_dict(result_value, "result", path)
    _require_exact_keys(
        result,
        {
            "answer",
            "reason",
            "evidence_ids",
            "observed_treatment",
            "observed_management_category",
            "correct",
        },
        "result",
        path,
    )
    expected_result = {
        "answer": prediction["answer"],
        "reason": prediction["reason"],
        "evidence_ids": prediction["evidence_ids"],
        "observed_treatment": evaluation["observed_treatment"],
        "observed_management_category": observed,
        "correct": evaluation["correct"],
    }
    if result != expected_result:
        raise ValueError(f"result is inconsistent with prediction/evaluation in {path}")
    return evaluation


def _validate_completed_payload(
    payload: Any,
    path: Path,
    *,
    source_mode: str,
    source_schema_version: str,
    source_no_suv_schema_version: str | None,
    source_config_fingerprint: str,
    config_has_trajectory_rag: bool,
) -> dict[str, Any]:
    """Validate every semantic field consumed from a completed trajectory."""
    record = _require_dict(payload, "trajectory", path)
    allowed_top_level = {
        "schema_version",
        "no_suv_schema_version",
        "trajectory_id",
        "case_id",
        "trajectory_number",
        "trajectory_fingerprint",
        "created_at_utc",
        "status",
        "input",
        "input_warnings",
        "trajectory_rag",
        "evidence_selection",
        "retrieved_evidence",
        "prediction",
        "evaluation",
        "result",
        "failure",
        "provenance",
    }
    optional_top_level = {"trajectory_rag", "no_suv_schema_version"}
    required_top_level = allowed_top_level - optional_top_level
    missing = sorted(required_top_level - set(record))
    unexpected = sorted(set(record) - allowed_top_level)
    if missing or unexpected:
        raise ValueError(
            f"trajectory has invalid top-level keys in {path}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if record["schema_version"] != source_schema_version:
        raise ValueError(f"trajectory schema_version differs from config in {path}")
    if source_mode == SOURCE_EVIDENCE_MODE_LITERATURE:
        if record.get("no_suv_schema_version") != source_no_suv_schema_version:
            raise ValueError(
                f"trajectory no_suv_schema_version differs from config in {path}"
            )
    elif "no_suv_schema_version" in record:
        raise ValueError(
            f"legacy trajectory unexpectedly declares no_suv_schema_version in {path}"
        )
    if record["status"] != "completed":
        raise ValueError(f"trajectory is not completed in {path}")
    if record["failure"] is not None:
        raise ValueError(f"completed trajectory has non-null failure in {path}")

    case_id = _require_nonempty_string(record["case_id"], "case_id", path)
    if _CASE_ID.fullmatch(case_id) is None:
        raise ValueError(f"case_id has invalid format in {path}")
    number = record["trajectory_number"]
    if not _is_integer(number) or number < 1:
        raise ValueError(f"trajectory_number must be positive integer in {path}")
    expected_trajectory_id = f"{case_id}_trajectory_{number:03d}"
    if record["trajectory_id"] != expected_trajectory_id:
        raise ValueError(f"trajectory_id is inconsistent in {path}")
    if path.parent.name != case_id:
        raise ValueError(f"path case ID differs from payload case_id in {path}")
    if path.name != f"trajectory_{number:03d}.json":
        raise ValueError(f"path trajectory number differs from payload in {path}")
    _require_sha256(
        record["trajectory_fingerprint"],
        "trajectory_fingerprint",
        path,
    )
    created_at = _require_nonempty_string(
        record["created_at_utc"],
        "created_at_utc",
        path,
    )
    try:
        timestamp = datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise ValueError(f"created_at_utc is invalid in {path}") from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"created_at_utc must be timezone-aware in {path}")

    patient_input = _validate_patient_input(record["input"], "input", path)
    warnings = _require_list(
        record["input_warnings"],
        "input_warnings",
        path,
        nonempty=False,
    )
    if any(not isinstance(item, str) or not item for item in warnings):
        raise ValueError(f"input_warnings must contain non-empty strings in {path}")

    trajectory_rag = record.get("trajectory_rag")
    if config_has_trajectory_rag:
        _require_dict(trajectory_rag, "trajectory_rag", path)
    elif trajectory_rag is not None:
        raise ValueError(
            f"trajectory_rag is unexpected for a non-RAG source run in {path}"
        )

    selection_wrapper = _require_dict(
        record["evidence_selection"],
        "evidence_selection",
        path,
    )
    _require_exact_keys(
        selection_wrapper,
        {"attempts", "accepted"},
        "evidence_selection",
        path,
    )
    accepted_selection = _require_dict(
        selection_wrapper["accepted"],
        "evidence_selection.accepted",
        path,
    )
    if source_mode == SOURCE_EVIDENCE_MODE_LITERATURE:
        _require_exact_keys(
            accepted_selection,
            {"literature_query"},
            "evidence_selection.accepted",
            path,
        )
        selected_organs: list[str] = []
    else:
        _require_exact_keys(
            accepted_selection,
            {"suv_organs", "literature_query"},
            "evidence_selection.accepted",
            path,
        )
        selected_organs_raw = _require_list(
            accepted_selection["suv_organs"],
            "evidence_selection.accepted.suv_organs",
            path,
        )
        if any(
            not isinstance(organ, str) or not organ for organ in selected_organs_raw
        ):
            raise ValueError(f"selected legacy organs are malformed in {path}")
        if len(set(selected_organs_raw)) != len(selected_organs_raw):
            raise ValueError(f"selected legacy organs contain duplicates in {path}")
        selected_organs = list(selected_organs_raw)
    literature_query = _require_nonempty_string(
        accepted_selection.get("literature_query"),
        "evidence_selection.accepted.literature_query",
        path,
    )
    if len(literature_query) < 10:
        raise ValueError(f"literature_query is too short in {path}")
    _validate_generation_attempts(
        selection_wrapper["attempts"],
        accepted_selection,
        "evidence_selection",
        path,
    )

    retrieved = _require_dict(
        record["retrieved_evidence"],
        "retrieved_evidence",
        path,
    )
    expected_retrieved_keys = {"literature"}
    if source_mode == SOURCE_EVIDENCE_MODE_LEGACY:
        expected_retrieved_keys.add("suv")
    if config_has_trajectory_rag:
        expected_retrieved_keys.add("similar_trajectories")
    _require_exact_keys(
        retrieved,
        expected_retrieved_keys,
        "retrieved_evidence",
        path,
    )
    _, valid_evidence_ids = _validate_literature(
        retrieved["literature"],
        path,
    )
    if source_mode == SOURCE_EVIDENCE_MODE_LEGACY:
        valid_evidence_ids.update(
            _validate_legacy_measurement_collection(
                retrieved["suv"],
                selected_organs,
                path,
            )
        )
    if config_has_trajectory_rag:
        valid_evidence_ids.update(
            _validate_projected_similar_trajectories(
                retrieved["similar_trajectories"],
                path,
            )
        )

    prediction = _validate_prediction(
        record["prediction"],
        valid_evidence_ids,
        require_legacy_measurement=source_mode == SOURCE_EVIDENCE_MODE_LEGACY,
        require_case_evidence=config_has_trajectory_rag,
        path=path,
    )
    _validate_evaluation_and_result(
        record["evaluation"],
        record["result"],
        prediction,
        path,
    )

    provenance = _require_dict(record["provenance"], "provenance", path)
    if provenance.get("inference_config_fingerprint") != source_config_fingerprint:
        raise ValueError(
            f"trajectory provenance config fingerprint differs in {path}"
        )
    if provenance.get("treatment_blind_generation") is not True:
        raise ValueError(f"trajectory is not marked treatment-blind in {path}")
    if source_mode == SOURCE_EVIDENCE_MODE_LITERATURE:
        if provenance.get("structured_suv_statistics_used") is not False:
            raise ValueError(
                f"literature-only trajectory lacks the no-structured-data marker in {path}"
            )
        if "suv_data_fingerprint" in provenance:
            raise ValueError(
                f"literature-only trajectory contains a structured-data fingerprint in {path}"
            )
        expected_fingerprint = _sha256_json(
            {
                "case_id": case_id,
                "trajectory_number": number,
                "patient_input": patient_input,
                "inference_config_fingerprint": source_config_fingerprint,
            }
        )
    else:
        legacy_data_fingerprint = _require_sha256(
            provenance.get("suv_data_fingerprint"),
            "provenance.suv_data_fingerprint",
            path,
        )
        expected_fingerprint = _sha256_json(
            {
                "case_id": case_id,
                "trajectory_number": number,
                "patient_input": patient_input,
                "suv_data_fingerprint": legacy_data_fingerprint,
                "inference_config_fingerprint": source_config_fingerprint,
            }
        )
    if record["trajectory_fingerprint"] != expected_fingerprint:
        raise ValueError(f"trajectory_fingerprint is inconsistent in {path}")
    return record


def _compact_literature(items: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "source": item["source"],
            "page": item["page"],
            "retrieval_score": float(item["retrieval_score"]),
        }
        for item in items
    ]


@dataclass(frozen=True)
class LiteratureTrajectory:
    path: Path
    patient_key: str
    payload: dict[str, Any]

    @property
    def case_id(self) -> str:
        return str(self.payload["case_id"])

    @property
    def trajectory_id(self) -> str:
        return str(self.payload["trajectory_id"])

    @property
    def patient_input(self) -> dict[str, Any]:
        source = self.payload["input"]
        return {
            "report": source["report"],
            "medical_history": source["medical_history"],
            "psa": dict(source["psa"]),
        }

    @property
    def literature_query(self) -> str:
        return str(
            self.payload["evidence_selection"]["accepted"]["literature_query"]
        )


class LiteratureTrajectoryRetriever:
    """TF-IDF/log-PSA retriever with strict patient-level leave-one-out."""

    def __init__(
        self,
        trajectories: list[LiteratureTrajectory],
        *,
        source_dir: Path,
        source_evidence_mode: str,
        source_config_fingerprint: str,
        corpus_fingerprint: str,
        total_files: int,
        skipped_status_counts: dict[str, int],
        expected_patient_inputs: dict[str, dict[str, Any]],
    ):
        if not trajectories:
            raise ValueError("The trajectory RAG corpus has no completed trajectories.")
        self.trajectories = trajectories
        self.source_dir = source_dir
        self.source_evidence_mode = source_evidence_mode
        self.source_config_fingerprint = source_config_fingerprint
        self.corpus_fingerprint = corpus_fingerprint
        self.total_files = total_files
        self.skipped_status_counts = skipped_status_counts
        self.expected_patient_inputs = expected_patient_inputs
        self.vectorizer = FeatureUnion(
            [
                (
                    "words",
                    TfidfVectorizer(
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=1,
                    ),
                ),
                (
                    "characters",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2 if len(trajectories) >= 2 else 1,
                    ),
                ),
            ]
        )
        documents = [_retrieval_text(item.patient_input) for item in trajectories]
        self.matrix = normalize(self.vectorizer.fit_transform(documents), norm="l2")
        strategy_documents = [
            _literature_strategy_text(item.payload) for item in trajectories
        ]
        self.strategy_matrix = normalize(
            self.vectorizer.transform(strategy_documents),
            norm="l2",
        )
        self.psa_values = [_numeric_psa(item.patient_input) for item in trajectories]

    @classmethod
    def from_output_dir(
        cls,
        source_dir: Path,
        dataset_path: Path,
    ) -> LiteratureTrajectoryRetriever:
        source_dir = Path(source_dir).resolve()
        dataset_path = Path(dataset_path).resolve()
        config_path = source_dir / "config.json"
        manifest_path = source_dir / "patient_manifest.json"
        patients_dir = source_dir / "patients"
        for required_path in (config_path, manifest_path, dataset_path):
            if not required_path.is_file():
                raise FileNotFoundError(required_path)
        if not patients_dir.is_dir():
            raise FileNotFoundError(patients_dir)

        config = _require_dict(_load_json(config_path), "source config", config_path)
        inference_config = _require_dict(
            config.get("inference_config"),
            "source inference_config",
            config_path,
        )
        source_config_fingerprint = _require_sha256(
            config.get("inference_config_fingerprint"),
            "inference_config_fingerprint",
            config_path,
        )
        calculated_config_fingerprint = _sha256_json(inference_config)
        if source_config_fingerprint != calculated_config_fingerprint:
            raise ValueError(
                f"inference_config_fingerprint is inconsistent in {config_path}"
            )
        source_schema_version = _require_nonempty_string(
            inference_config.get("schema_version"),
            "inference_config.schema_version",
            config_path,
        )
        if inference_config.get("label_mapping_version") != "observed-management-v1":
            raise ValueError(f"label mapping version is unsupported in {config_path}")
        _require_sha256(
            inference_config.get("orchestrator_sha256"),
            "inference_config.orchestrator_sha256",
            config_path,
        )
        source_case_id_salt = _require_nonempty_string(
            inference_config.get("case_id_salt"),
            "inference_config.case_id_salt",
            config_path,
        )

        evidence_mode = inference_config.get("evidence_mode")
        if evidence_mode == SOURCE_EVIDENCE_MODE_LITERATURE:
            source_evidence_mode = SOURCE_EVIDENCE_MODE_LITERATURE
            source_no_suv_schema_version: str | None = _require_nonempty_string(
                inference_config.get("no_suv_schema_version"),
                "inference_config.no_suv_schema_version",
                config_path,
            )
            if inference_config.get("structured_suv_statistics_used") is not False:
                raise ValueError(
                    "literature-only source config must set "
                    f"structured_suv_statistics_used=false in {config_path}"
                )
            forbidden_config_keys = sorted(
                {"suv_dir", "max_suv_organs"} & set(inference_config)
            )
            if forbidden_config_keys:
                raise ValueError(
                    "literature-only source config contains forbidden fields "
                    f"{forbidden_config_keys} in {config_path}"
                )
        elif (
            evidence_mode is None
            and "structured_suv_statistics_used" not in inference_config
            and isinstance(inference_config.get("suv_dir"), str)
            and _is_integer(inference_config.get("max_suv_organs"))
        ):
            source_evidence_mode = SOURCE_EVIDENCE_MODE_LEGACY
            source_no_suv_schema_version = None
        else:
            raise ValueError(
                f"Unsupported or ambiguous source evidence mode in {config_path}"
            )
        config_has_trajectory_rag = inference_config.get("trajectory_rag") is not None
        if (
            source_evidence_mode == SOURCE_EVIDENCE_MODE_LEGACY
            and config_has_trajectory_rag
        ):
            raise ValueError(
                "Legacy structured trajectory-RAG output is not an accepted source "
                f"because nested historical content cannot be safely audited: {config_path}"
            )

        dataset_payload = _require_dict(
            _load_json(dataset_path),
            "current dataset",
            dataset_path,
        )
        expected_dataset_hash = _require_sha256(
            inference_config.get("dataset_sha256"),
            "inference_config.dataset_sha256",
            config_path,
        )
        current_dataset_hash = _sha256_file(dataset_path)
        if expected_dataset_hash != current_dataset_hash:
            raise ValueError(
                "Trajectory corpus and current inference dataset differ: "
                f"{expected_dataset_hash!r} != {current_dataset_hash!r}"
            )
        expected_patient_inputs: dict[str, dict[str, Any]] = {}
        for patient_key, dataset_record_value in dataset_payload.items():
            if not isinstance(patient_key, str) or not patient_key:
                raise ValueError(f"dataset patient keys must be non-empty strings")
            dataset_record = _require_dict(
                dataset_record_value,
                f"dataset record for {patient_key!r}",
                dataset_path,
            )
            expected_patient_input = _expected_patient_input(dataset_record)
            _validate_patient_input(
                expected_patient_input,
                f"normalized dataset input for {patient_key!r}",
                dataset_path,
            )
            expected_patient_inputs[patient_key] = expected_patient_input

        case_to_patient = _require_dict(
            _load_json(manifest_path),
            "patient manifest",
            manifest_path,
        )
        if not case_to_patient:
            raise ValueError(f"patient manifest is empty in {manifest_path}")
        manifest_patient_keys: list[str] = []
        for case_id, patient_key in case_to_patient.items():
            if (
                not isinstance(case_id, str)
                or _CASE_ID.fullmatch(case_id) is None
                or not isinstance(patient_key, str)
                or not patient_key
            ):
                raise ValueError(
                    f"patient manifest must map valid case IDs to strings in {manifest_path}"
                )
            if patient_key not in expected_patient_inputs:
                raise ValueError(
                    f"Manifest patient {patient_key!r} is absent from current dataset"
                )
            expected_case_id = _stable_case_id(patient_key, source_case_id_salt)
            if case_id != expected_case_id:
                raise ValueError(
                    f"Manifest case ID {case_id!r} does not match patient "
                    f"{patient_key!r} under the source case-ID salt"
                )
            manifest_patient_keys.append(patient_key)
        duplicates = sorted(
            patient_key
            for patient_key, count in Counter(manifest_patient_keys).items()
            if count > 1
        )
        if duplicates:
            raise ValueError(
                f"Patient manifest maps multiple case IDs to patients: {duplicates}"
            )

        paths = sorted(patients_dir.glob("case_*/trajectory_*.json"))
        if not paths:
            raise FileNotFoundError(
                f"No case_*/trajectory_*.json files found below {patients_dir}"
            )
        discovered_case_ids = {path.parent.name for path in paths}
        unknown_case_ids = sorted(discovered_case_ids - set(case_to_patient))
        missing_case_ids = sorted(set(case_to_patient) - discovered_case_ids)
        if unknown_case_ids or missing_case_ids:
            raise ValueError(
                f"patient manifest and trajectory directories differ: "
                f"unknown={unknown_case_ids}, missing={missing_case_ids}"
            )

        trajectories: list[LiteratureTrajectory] = []
        statuses: Counter[str] = Counter()
        fingerprint_records: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        for path in paths:
            payload = _load_json(path)
            payload_object = _require_dict(payload, "trajectory", path)
            status = payload_object.get("status")
            if not isinstance(status, str) or not status:
                raise ValueError(f"trajectory status must be a non-empty string in {path}")
            statuses[status] += 1
            if status != "completed":
                continue
            record = _validate_completed_payload(
                payload_object,
                path,
                source_mode=source_evidence_mode,
                source_schema_version=source_schema_version,
                source_no_suv_schema_version=source_no_suv_schema_version,
                source_config_fingerprint=source_config_fingerprint,
                config_has_trajectory_rag=config_has_trajectory_rag,
            )
            case_id = str(record["case_id"])
            patient_key = case_to_patient.get(case_id)
            if not isinstance(patient_key, str) or not patient_key:
                raise ValueError(
                    f"Completed trajectory {path} has no patient mapping in {manifest_path}"
                )
            if record["input"] != expected_patient_inputs[patient_key]:
                raise ValueError(
                    f"Completed trajectory input differs from current dataset in {path}"
                )
            trajectory_id = str(record["trajectory_id"])
            if trajectory_id in seen_ids:
                raise ValueError(f"Duplicate trajectory_id in corpus: {trajectory_id}")
            seen_ids.add(trajectory_id)
            trajectories.append(
                LiteratureTrajectory(
                    path=path,
                    patient_key=patient_key,
                    payload=record,
                )
            )
            fingerprint_records.append(
                {
                    "relative_path": str(path.relative_to(source_dir)),
                    "sha256": _sha256_file(path),
                }
            )

        skipped = {
            status: count
            for status, count in sorted(statuses.items())
            if status != "completed"
        }
        corpus_fingerprint = _sha256_json(
            {
                "source_config_fingerprint": source_config_fingerprint,
                "source_evidence_mode": source_evidence_mode,
                "patient_manifest_sha256": _sha256_file(manifest_path),
                "completed_files": fingerprint_records,
            }
        )
        return cls(
            trajectories,
            source_dir=source_dir,
            source_evidence_mode=source_evidence_mode,
            source_config_fingerprint=source_config_fingerprint,
            corpus_fingerprint=corpus_fingerprint,
            total_files=len(paths),
            skipped_status_counts=skipped,
            expected_patient_inputs=expected_patient_inputs,
        )

    def metadata(self) -> dict[str, Any]:
        correctness = Counter(
            bool(item.payload["evaluation"]["correct"]) for item in self.trajectories
        )
        return {
            "source_dir": str(self.source_dir),
            "source_evidence_mode": self.source_evidence_mode,
            "source_config_fingerprint": self.source_config_fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
            "trajectory_files": self.total_files,
            "indexed_completed_trajectories": len(self.trajectories),
            "indexed_correct_trajectories": correctness[True],
            "indexed_incorrect_trajectories": correctness[False],
            "skipped_status_counts": self.skipped_status_counts,
            "indexed_patients": len(
                {item.patient_key for item in self.trajectories}
            ),
            "retrieval_fields": [
                "input.report",
                "input.medical_history",
                "input.psa",
            ],
            "within_case_tie_break": [
                "evidence_selection.literature_query",
            ],
            "historical_structured_suv_values_exposed": False,
            "historical_prediction_reason_exposed": False,
        }

    def search(
        self,
        patient_input: dict[str, Any],
        *,
        exclude_patient_key: str,
        top_k: int = 5,
        max_per_case: int = 1,
        psa_weight: float = 0.15,
    ) -> dict[str, Any]:
        """Exclude the target patient's entire history before deterministic ranking."""
        query_path = Path("<current_patient_input>")
        validated_query = _validate_patient_input(
            patient_input,
            "current patient input",
            query_path,
        )
        if not isinstance(exclude_patient_key, str) or not exclude_patient_key:
            raise ValueError("exclude_patient_key must be a non-empty string.")
        expected_query = self.expected_patient_inputs.get(exclude_patient_key)
        if expected_query is None:
            raise KeyError(
                f"exclude_patient_key {exclude_patient_key!r} is absent from the dataset"
            )
        if validated_query != expected_query:
            raise ValueError(
                "Current patient input does not match exclude_patient_key in the dataset."
            )
        if not _is_integer(top_k) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        if not _is_integer(max_per_case) or max_per_case < 1:
            raise ValueError("max_per_case must be a positive integer.")
        if not _is_finite_number(psa_weight) or not 0 <= float(psa_weight) <= 1:
            raise ValueError("psa_weight must be a finite number in [0, 1].")
        eligible_indices = [
            index
            for index, item in enumerate(self.trajectories)
            if item.patient_key != exclude_patient_key
        ]
        excluded_count = len(self.trajectories) - len(eligible_indices)
        if not eligible_indices:
            raise ValueError(
                "No completed trajectory remains after excluding the current patient."
            )

        query = normalize(
            self.vectorizer.transform([_retrieval_text(validated_query)]),
            norm="l2",
        )
        text_scores = (self.matrix @ query.T).toarray().ravel()
        strategy_scores = (self.strategy_matrix @ query.T).toarray().ravel()
        final_scores = text_scores.copy()
        psa_scores: list[float | None] = [None] * len(self.trajectories)
        effective_psa_weights: list[float] = [0.0] * len(self.trajectories)
        query_psa = _numeric_psa(validated_query)
        if query_psa is not None:
            query_log_psa = math.log1p(query_psa)
            for index in eligible_indices:
                candidate_psa = self.psa_values[index]
                if candidate_psa is None:
                    continue
                psa_score = math.exp(
                    -abs(query_log_psa - math.log1p(candidate_psa))
                )
                psa_scores[index] = psa_score
                effective_psa_weights[index] = float(psa_weight)
                final_scores[index] = (
                    (1.0 - float(psa_weight)) * text_scores[index]
                    + float(psa_weight) * psa_score
                )

        eligible_indices.sort(
            key=lambda index: (
                -float(final_scores[index]),
                -float(strategy_scores[index]),
                self.trajectories[index].trajectory_id,
            )
        )
        selected: list[tuple[int, LiteratureTrajectory]] = []
        per_patient: Counter[str] = Counter()
        for index in eligible_indices:
            item = self.trajectories[index]
            if per_patient[item.patient_key] >= max_per_case:
                continue
            selected.append((index, item))
            per_patient[item.patient_key] += 1
            if len(selected) == top_k:
                break
        if not selected:
            raise ValueError("Trajectory retrieval did not return any eligible result.")

        query_hints: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        for rank, (index, item) in enumerate(selected, 1):
            payload = item.payload
            prediction = payload["prediction"]["accepted"]
            evaluation = payload["evaluation"]
            score = float(final_scores[index])
            query_hints.append(
                {
                    "source_trajectory_id": item.trajectory_id,
                    "retrieval_score": score,
                    "literature_query": item.literature_query,
                }
            )
            result = {
                "evidence_id": f"CASE-{rank:03d}",
                "source_case_id": item.case_id,
                "source_trajectory_id": item.trajectory_id,
                "retrieval_score": score,
                "retrieval_components": {
                    "text_similarity": float(text_scores[index]),
                    "psa_similarity": psa_scores[index],
                    "psa_weight": effective_psa_weights[index],
                    "literature_query_similarity_tie_break": float(
                        strategy_scores[index]
                    ),
                },
                "historical_patient_input": item.patient_input,
                "historical_evidence_selection": {
                    "literature_query": item.literature_query,
                },
                "historical_literature": _compact_literature(
                    payload["retrieved_evidence"]["literature"]
                ),
                "historical_prediction": {
                    "answer": prediction["answer"],
                },
                "historical_evaluation": {
                    "observed_management_category": evaluation[
                        "observed_management_category"
                    ],
                    "prediction_correct": evaluation["correct"],
                },
            }
            forbidden = _forbidden_structured_keys(result)
            if forbidden:
                raise RuntimeError(
                    "Internal error: projected result contains forbidden "
                    f"structured fields {forbidden}"
                )
            results.append(result)

        return {
            "source_dir": str(self.source_dir),
            "source_evidence_mode": self.source_evidence_mode,
            "corpus_fingerprint": self.corpus_fingerprint,
            "query_fields": [
                "input.report",
                "input.medical_history",
                "input.psa",
            ],
            "excluded_current_patient": True,
            "excluded_completed_trajectories": excluded_count,
            "eligible_completed_trajectories": len(eligible_indices),
            "top_k": top_k,
            "returned_trajectories": len(results),
            "max_trajectories_per_case": max_per_case,
            "psa_weight": float(psa_weight),
            "literature_query_hints": query_hints,
            "historical_structured_suv_values_exposed": False,
            "historical_prediction_reason_exposed": False,
            "results": results,
        }


__all__ = ["LiteratureTrajectoryRetriever"]
