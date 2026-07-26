"""Leave-one-patient-out retrieval over completed inference trajectories."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize


PATIENT_INPUT_FIELDS = {"report", "medical_history", "psa"}
MANAGEMENT_CATEGORIES = {
    "radical_prostatectomy",
    "systemic_treatment",
    "local_treatment",
    "other_examination",
}


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


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


def _retrieval_text(patient_input: dict[str, Any]) -> str:
    """Build a treatment-blind document/query from the whitelisted patient input."""
    return "\n".join(
        (
            f"Report: {patient_input.get('report', '')}",
            f"Medical history: {patient_input.get('medical_history', '')}",
        )
    )


def _strategy_text(payload: dict[str, Any]) -> str:
    selection = payload["evidence_selection"]["accepted"]
    organs = " ".join(str(value) for value in selection["suv_organs"])
    return (
        f"Selected SUV organs: {organs}\n"
        f"Literature query: {selection['literature_query']}"
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


def _stable_case_id(patient_key: str, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}:{patient_key}".encode("utf-8")).hexdigest()
    return f"case_{digest[:16]}"


def _numeric_psa(patient_input: dict[str, Any]) -> float | None:
    psa = patient_input.get("psa")
    if not isinstance(psa, dict):
        return None
    value = psa.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _require_dict(value: Any, description: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object in {path}")
    return value


def _require_list(value: Any, description: str, path: Path) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{description} must be a non-empty list in {path}")
    return value


def _validate_completed_payload(payload: Any, path: Path) -> dict[str, Any]:
    """Validate fields consumed by trajectory RAG, without re-evaluating the source run."""
    record = _require_dict(payload, "trajectory", path)
    if record.get("status") != "completed":
        raise ValueError(f"trajectory is not completed in {path}")
    if record.get("failure") is not None:
        raise ValueError(f"completed trajectory has a non-null failure in {path}")
    for key in ("trajectory_id", "case_id"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"{key} must be a non-empty string in {path}")
    patient_input = _require_dict(record.get("input"), "input", path)
    missing_input_fields = sorted(PATIENT_INPUT_FIELDS - set(patient_input))
    if missing_input_fields:
        raise ValueError(
            f"input is missing treatment-blind fields {missing_input_fields} in {path}"
        )
    unexpected_input_fields = sorted(set(patient_input) - PATIENT_INPUT_FIELDS)
    if unexpected_input_fields:
        raise ValueError(
            f"input contains non-whitelisted fields {unexpected_input_fields} in {path}"
        )
    _require_dict(patient_input.get("psa"), "input.psa", path)
    selection = _require_dict(record.get("evidence_selection"), "evidence_selection", path)
    accepted_selection = _require_dict(
        selection.get("accepted"),
        "evidence_selection.accepted",
        path,
    )
    selected_organs = _require_list(
        accepted_selection.get("suv_organs"),
        "evidence_selection.accepted.suv_organs",
        path,
    )
    if any(not isinstance(organ, str) or not organ for organ in selected_organs):
        raise ValueError(f"selected SUV organs must be non-empty strings in {path}")
    if not isinstance(accepted_selection.get("literature_query"), str) or not (
        accepted_selection["literature_query"]
    ):
        raise ValueError(f"literature_query must be a non-empty string in {path}")
    retrieved = _require_dict(record.get("retrieved_evidence"), "retrieved_evidence", path)
    retrieved_suv = _require_list(
        retrieved.get("suv"),
        "retrieved_evidence.suv",
        path,
    )
    retrieved_literature = _require_list(
        retrieved.get("literature"),
        "retrieved_evidence.literature",
        path,
    )
    if any(
        not isinstance(item, dict) or not isinstance(item.get("organ"), str)
        for item in retrieved_suv
    ):
        raise ValueError(f"retrieved SUV items are malformed in {path}")
    if [item["organ"] for item in retrieved_suv] != selected_organs:
        raise ValueError(
            f"retrieved SUV organs differ from the accepted selection in {path}"
        )
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("source"), str)
        or not isinstance(item.get("page"), int)
        for item in retrieved_literature
    ):
        raise ValueError(f"retrieved literature items are malformed in {path}")
    prediction = _require_dict(record.get("prediction"), "prediction", path)
    accepted_prediction = _require_dict(
        prediction.get("accepted"),
        "prediction.accepted",
        path,
    )
    if accepted_prediction.get("answer") not in MANAGEMENT_CATEGORIES:
        raise ValueError(f"prediction answer is invalid in {path}")
    if not isinstance(accepted_prediction.get("reason"), str) or not (
        accepted_prediction["reason"]
    ):
        raise ValueError(f"prediction reason must be a non-empty string in {path}")
    evaluation = _require_dict(record.get("evaluation"), "evaluation", path)
    if not isinstance(evaluation.get("correct"), bool):
        raise ValueError(f"evaluation.correct must be boolean in {path}")
    if evaluation.get("observed_management_category") not in MANAGEMENT_CATEGORIES:
        raise ValueError(
            f"evaluation.observed_management_category is invalid in {path}"
        )
    expected_correct = (
        accepted_prediction.get("answer")
        == evaluation["observed_management_category"]
    )
    if evaluation["correct"] != expected_correct:
        raise ValueError(f"evaluation.correct is inconsistent in {path}")
    if path.parent.name != record["case_id"]:
        raise ValueError(f"path case ID differs from payload case_id in {path}")
    if path.stem not in str(record["trajectory_id"]):
        raise ValueError(f"path trajectory number differs from trajectory_id in {path}")
    return record


def _compact_literature(items: list[Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "source": item.get("source"),
                "page": item.get("page"),
                "retrieval_score": item.get("retrieval_score"),
            }
        )
    return compact


def _historical_reason(value: Any, max_chars: int) -> str:
    """Namespace old evidence IDs so they cannot masquerade as current evidence."""
    text = str(value)
    text = re.sub(r"\b(SUV|LIT)-(\d+)\b", r"HIST-\1-\2", text)
    return text[:max_chars]


@dataclass(frozen=True)
class CompletedTrajectory:
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
            "psa": source["psa"],
        }


class CompletedTrajectoryRetriever:
    """Hybrid text/PSA retriever with strict leave-one-patient-out filtering."""

    def __init__(
        self,
        trajectories: list[CompletedTrajectory],
        *,
        source_dir: Path,
        source_config_fingerprint: str,
        corpus_fingerprint: str,
        total_files: int,
        skipped_status_counts: dict[str, int],
    ):
        if not trajectories:
            raise ValueError("The trajectory RAG corpus has no completed trajectories.")
        self.trajectories = trajectories
        self.source_dir = source_dir
        self.source_config_fingerprint = source_config_fingerprint
        self.corpus_fingerprint = corpus_fingerprint
        self.total_files = total_files
        self.skipped_status_counts = skipped_status_counts
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
                        min_df=2,
                    ),
                ),
            ]
        )
        documents = [_retrieval_text(item.patient_input) for item in trajectories]
        self.matrix = normalize(self.vectorizer.fit_transform(documents), norm="l2")
        strategies = [_strategy_text(item.payload) for item in trajectories]
        self.strategy_matrix = normalize(
            self.vectorizer.transform(strategies),
            norm="l2",
        )
        self.psa_values = [_numeric_psa(item.patient_input) for item in trajectories]

    @classmethod
    def from_output_dir(
        cls,
        source_dir: Path,
        *,
        dataset_path: Path,
    ) -> CompletedTrajectoryRetriever:
        source_dir = source_dir.resolve()
        config_path = source_dir / "config.json"
        manifest_path = source_dir / "patient_manifest.json"
        patients_dir = source_dir / "patients"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        if not patients_dir.is_dir():
            raise FileNotFoundError(patients_dir)

        config = _require_dict(_load_json(config_path), "source config", config_path)
        inference_config = _require_dict(
            config.get("inference_config"),
            "source inference_config",
            config_path,
        )
        dataset_payload = _require_dict(
            _load_json(dataset_path),
            "current dataset",
            dataset_path,
        )
        expected_dataset_hash = inference_config.get("dataset_sha256")
        current_dataset_hash = _sha256_file(dataset_path)
        if expected_dataset_hash != current_dataset_hash:
            raise ValueError(
                "Trajectory corpus and current inference dataset differ: "
                f"{expected_dataset_hash!r} != {current_dataset_hash!r}"
            )
        source_config_fingerprint = config.get("inference_config_fingerprint")
        if not isinstance(source_config_fingerprint, str):
            raise ValueError(
                f"inference_config_fingerprint is missing from {config_path}"
            )
        source_case_id_salt = inference_config.get("case_id_salt")
        if not isinstance(source_case_id_salt, str) or not source_case_id_salt:
            raise ValueError(f"case_id_salt is missing from {config_path}")

        case_to_patient = _require_dict(
            _load_json(manifest_path),
            "patient manifest",
            manifest_path,
        )
        manifest_patient_keys: list[str] = []
        for case_id, patient_key in case_to_patient.items():
            if not isinstance(case_id, str) or not isinstance(patient_key, str):
                raise ValueError(f"patient manifest must map strings in {manifest_path}")
            if patient_key not in dataset_payload:
                raise ValueError(
                    f"Manifest patient {patient_key!r} is absent from the current dataset"
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
        manifest_sha256 = _sha256_file(manifest_path)
        paths = sorted(patients_dir.glob("case_*/trajectory_*.json"))
        if not paths:
            raise FileNotFoundError(
                f"No case_*/trajectory_*.json files found below {patients_dir}"
            )

        trajectories: list[CompletedTrajectory] = []
        statuses: Counter[str] = Counter()
        fingerprint_records: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        for path in paths:
            payload = _load_json(path)
            if not isinstance(payload, dict):
                raise ValueError(f"Trajectory top level must be an object in {path}")
            status = str(payload.get("status", "invalid"))
            statuses[status] += 1
            if status != "completed":
                continue
            record = _validate_completed_payload(payload, path)
            case_id = str(record["case_id"])
            patient_key = case_to_patient.get(case_id)
            if not isinstance(patient_key, str) or not patient_key:
                raise ValueError(
                    f"Completed trajectory {path} has no patient mapping in {manifest_path}"
                )
            dataset_record = _require_dict(
                dataset_payload[patient_key],
                f"dataset record for {patient_key!r}",
                dataset_path,
            )
            if record["input"] != _expected_patient_input(dataset_record):
                raise ValueError(
                    f"Completed trajectory input differs from the current dataset in {path}"
                )
            trajectory_id = str(record["trajectory_id"])
            if trajectory_id in seen_ids:
                raise ValueError(f"Duplicate trajectory_id in corpus: {trajectory_id}")
            seen_ids.add(trajectory_id)
            trajectories.append(
                CompletedTrajectory(
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
            status: count for status, count in sorted(statuses.items()) if status != "completed"
        }
        corpus_fingerprint = _sha256_json(
            {
                "source_config_fingerprint": source_config_fingerprint,
                "patient_manifest_sha256": manifest_sha256,
                "completed_files": fingerprint_records,
            }
        )
        return cls(
            trajectories,
            source_dir=source_dir,
            source_config_fingerprint=source_config_fingerprint,
            corpus_fingerprint=corpus_fingerprint,
            total_files=len(paths),
            skipped_status_counts=skipped,
        )

    def metadata(self) -> dict[str, Any]:
        correctness = Counter(
            bool(item.payload["evaluation"]["correct"]) for item in self.trajectories
        )
        return {
            "source_dir": str(self.source_dir),
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
            "retrieval_fields": ["input.report", "input.medical_history", "input.psa"],
            "within_case_tie_break": [
                "evidence_selection.literature_query",
                "evidence_selection.suv_organs",
            ],
        }

    def search(
        self,
        patient_input: dict[str, Any],
        *,
        exclude_patient_key: str,
        top_k: int,
        max_per_case: int,
        psa_weight: float,
        reason_chars: int,
        available_organs: set[str],
    ) -> dict[str, Any]:
        """Search after excluding every trajectory belonging to the target patient."""
        query = normalize(
            self.vectorizer.transform([_retrieval_text(patient_input)]),
            norm="l2",
        )
        text_scores = (self.matrix @ query.T).toarray().ravel()
        strategy_scores = (self.strategy_matrix @ query.T).toarray().ravel()
        query_psa = _numeric_psa(patient_input)
        final_scores = text_scores.copy()
        psa_scores: list[float | None] = [None] * len(self.trajectories)
        if query_psa is not None:
            query_log_psa = math.log1p(query_psa)
            for index, candidate_psa in enumerate(self.psa_values):
                if candidate_psa is None:
                    continue
                psa_score = math.exp(
                    -abs(query_log_psa - math.log1p(candidate_psa))
                )
                psa_scores[index] = psa_score
                final_scores[index] = (
                    (1.0 - psa_weight) * text_scores[index]
                    + psa_weight * psa_score
                )

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
        eligible_indices.sort(
            key=lambda index: (
                -float(final_scores[index]),
                -float(strategy_scores[index]),
                self.trajectories[index].trajectory_id,
            )
        )

        selected: list[tuple[int, CompletedTrajectory]] = []
        per_case: Counter[str] = Counter()
        for index in eligible_indices:
            item = self.trajectories[index]
            if per_case[item.case_id] >= max_per_case:
                continue
            selected.append((index, item))
            per_case[item.case_id] += 1
            if len(selected) == top_k:
                break
        if not selected:
            raise ValueError("Trajectory retrieval did not return any eligible result.")

        results: list[dict[str, Any]] = []
        organ_scores: Counter[str] = Counter()
        query_hints: list[dict[str, Any]] = []
        for rank, (index, item) in enumerate(selected, 1):
            payload = item.payload
            selection = payload["evidence_selection"]["accepted"]
            retrieved = payload["retrieved_evidence"]
            prediction = payload["prediction"]["accepted"]
            evaluation = payload["evaluation"]
            score = float(final_scores[index])
            selected_organs = list(selection["suv_organs"])
            for organ in selected_organs:
                if organ in available_organs:
                    organ_scores[organ] += score
            query_hints.append(
                {
                    "source_trajectory_id": item.trajectory_id,
                    "retrieval_score": score,
                    "literature_query": selection["literature_query"],
                }
            )
            results.append(
                {
                    "evidence_id": f"CASE-{rank:03d}",
                    "source_case_id": item.case_id,
                    "source_trajectory_id": item.trajectory_id,
                    "retrieval_score": score,
                    "retrieval_components": {
                        "text_similarity": float(text_scores[index]),
                        "psa_similarity": psa_scores[index],
                        "psa_weight": psa_weight if psa_scores[index] is not None else 0.0,
                        "strategy_similarity_tie_break": float(
                            strategy_scores[index]
                        ),
                    },
                    "historical_patient_input": item.patient_input,
                    "historical_evidence_selection": {
                        "suv_organs": selected_organs,
                        "literature_query": selection["literature_query"],
                    },
                    "historical_retrieved_suv_organs": [
                        item.get("organ")
                        for item in retrieved["suv"]
                        if isinstance(item, dict)
                    ],
                    "historical_literature": _compact_literature(
                        retrieved["literature"]
                    ),
                    "historical_prediction": {
                        "answer": prediction["answer"],
                        "reason": _historical_reason(
                            prediction["reason"],
                            reason_chars,
                        ),
                    },
                    "historical_evaluation": {
                        "observed_management_category": evaluation[
                            "observed_management_category"
                        ],
                        "prediction_correct": evaluation["correct"],
                    },
                }
            )

        return {
            "source_dir": str(self.source_dir),
            "corpus_fingerprint": self.corpus_fingerprint,
            "query_fields": ["input.report", "input.medical_history", "input.psa"],
            "excluded_current_patient": True,
            "excluded_completed_trajectories": excluded_count,
            "eligible_completed_trajectories": len(eligible_indices),
            "top_k": top_k,
            "max_trajectories_per_case": max_per_case,
            "psa_weight": psa_weight,
            "organ_hints": [
                {"organ": organ, "weighted_score": float(score)}
                for organ, score in sorted(
                    organ_scores.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
            "literature_query_hints": query_hints,
            "results": results,
        }
