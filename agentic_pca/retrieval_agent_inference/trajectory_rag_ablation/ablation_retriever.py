"""Auditable retrieval ablations over the validated trajectory-RAG corpus.

The corpus is always loaded by :class:`CompletedTrajectoryRetriever`; this
module deliberately does not provide a second, weaker corpus loader.  Every
ranking mode also applies leave-one-patient-out before any other filtering.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sklearn.preprocessing import normalize

from agentic_pca.retrieval_agent_inference.trajectory_rag.trajectory_rag import (
    CompletedTrajectoryRetriever,
    _compact_literature,
    _historical_reason,
    _numeric_psa,
)


_QUERY_FIELD_ORDER = ("report", "medical_history", "psa")
_QUERY_FIELD_SET = frozenset(_QUERY_FIELD_ORDER)
_RANKINGS = frozenset(("similar", "random", "least_similar", "psa_only"))
_CORPUS_FILTERS = frozenset(("all", "correct", "incorrect"))
_MISSING = object()


def _spec_value(spec: Any, key: str, default: Any = _MISSING) -> Any:
    """Read a setting from either a mapping or an attribute-based object."""
    if isinstance(spec, Mapping):
        if key in spec:
            return spec[key]
    elif spec is not None and hasattr(spec, key):
        return getattr(spec, key)
    if default is _MISSING:
        raise ValueError(f"Ablation spec is missing required setting {key!r}.")
    return default


def _spec_name(spec: Any) -> str | None:
    if isinstance(spec, str):
        return spec
    for key in ("name", "id", "key", "ablation"):
        value = _spec_value(spec, key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _query_fields(value: Any) -> tuple[str, ...]:
    if value is None:
        return _QUERY_FIELD_ORDER
    if isinstance(value, str):
        text = value.strip().lower().replace("medical history", "medical_history")
        if text in {"all", "full"}:
            return _QUERY_FIELD_ORDER
        raw_fields: Sequence[Any] = tuple(
            part for part in re.split(r"[\s,+|]+", text) if part
        )
    elif isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray)
    ):
        raw_fields = value
    elif isinstance(value, (set, frozenset)):
        raw_fields = tuple(value)
    else:
        raise TypeError(
            "query_fields must be a string or a sequence/set of field names."
        )

    aliases = {
        "history": "medical_history",
        "input.report": "report",
        "input.medical_history": "medical_history",
        "input.psa": "psa",
    }
    normalized: set[str] = set()
    for raw_field in raw_fields:
        if not isinstance(raw_field, str):
            raise TypeError("Every query_fields entry must be a string.")
        field = aliases.get(raw_field.strip().lower(), raw_field.strip().lower())
        if field not in _QUERY_FIELD_SET:
            raise ValueError(
                f"Unsupported query field {raw_field!r}; expected a subset of "
                f"{list(_QUERY_FIELD_ORDER)}."
            )
        normalized.add(field)
    if not normalized:
        raise ValueError("query_fields cannot be empty.")
    return tuple(field for field in _QUERY_FIELD_ORDER if field in normalized)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _settings(spec: Any) -> dict[str, Any]:
    name = _spec_name(spec)
    if isinstance(spec, str):
        if spec.strip().lower() != "full":
            raise ValueError(
                "A string ablation spec is only supported for the special 'full' run."
            )
        ranking_value = "full"
    else:
        ranking_value = _spec_value(spec, "ranking", "similar")
    if not isinstance(ranking_value, str):
        raise TypeError("ranking must be a string.")
    requested_ranking = ranking_value.strip().lower()
    declared_full = requested_ranking == "full" or (
        isinstance(name, str) and name.lower() == "full"
    )
    ranking = "similar" if requested_ranking == "full" else requested_ranking
    if ranking not in _RANKINGS:
        raise ValueError(
            f"Unsupported ranking {ranking_value!r}; expected one of "
            f"{sorted(_RANKINGS)} or 'full'."
        )

    fields = _query_fields(
        None if isinstance(spec, str) else _spec_value(spec, "query_fields", None)
    )
    corpus_filter = (
        "all"
        if isinstance(spec, str)
        else _spec_value(spec, "corpus_filter", "all")
    )
    if not isinstance(corpus_filter, str):
        raise TypeError("corpus_filter must be a string.")
    corpus_filter = corpus_filter.strip().lower()
    if corpus_filter not in _CORPUS_FILTERS:
        raise ValueError(
            f"Unsupported corpus_filter {corpus_filter!r}; expected one of "
            f"{sorted(_CORPUS_FILTERS)}."
        )

    top_k = _positive_int(
        5 if isinstance(spec, str) else _spec_value(spec, "top_k", 5),
        "top_k",
    )
    max_per_case = _positive_int(
        1 if isinstance(spec, str) else _spec_value(spec, "max_per_case", 1),
        "max_per_case",
    )
    psa_weight_value = (
        0.15 if isinstance(spec, str) else _spec_value(spec, "psa_weight", 0.15)
    )
    if isinstance(psa_weight_value, bool) or not isinstance(
        psa_weight_value, (int, float)
    ):
        raise TypeError("psa_weight must be a real number.")
    psa_weight = float(psa_weight_value)
    if not math.isfinite(psa_weight) or not 0.0 <= psa_weight <= 1.0:
        raise ValueError("psa_weight must be finite and in [0, 1].")
    reason_chars = _positive_int(
        1200 if isinstance(spec, str) else _spec_value(spec, "reason_chars", 1200),
        "reason_chars",
    )
    if reason_chars < 20:
        raise ValueError("reason_chars must be at least 20.")
    permute_outcomes = (
        False
        if isinstance(spec, str)
        else _spec_value(spec, "permute_outcomes", False)
    )
    if not isinstance(permute_outcomes, bool):
        raise TypeError("permute_outcomes must be boolean.")

    if declared_full and (
        ranking != "similar"
        or fields != _QUERY_FIELD_ORDER
        or corpus_filter != "all"
        or permute_outcomes
    ):
        raise ValueError(
            "The special 'full' ablation must use similar ranking, all query "
            "fields, the complete validated corpus, and unmodified outcomes."
        )

    return {
        "name": name,
        "is_full": declared_full,
        "requested_ranking": requested_ranking,
        "ranking": ranking,
        "query_fields": fields,
        "corpus_filter": corpus_filter,
        "top_k": top_k,
        "max_per_case": max_per_case,
        "psa_weight": psa_weight,
        "reason_chars": reason_chars,
        "permute_outcomes": permute_outcomes,
        "delegate": (
            ranking == "similar"
            and fields == _QUERY_FIELD_ORDER
            and corpus_filter == "all"
        ),
    }


def _field_text(patient_input: Mapping[str, Any], fields: tuple[str, ...]) -> str:
    parts: list[str] = []
    if "report" in fields:
        parts.append(f"Report: {patient_input.get('report', '')}")
    if "medical_history" in fields:
        parts.append(
            f"Medical history: {patient_input.get('medical_history', '')}"
        )
    return "\n".join(parts)


def _input_field_labels(fields: tuple[str, ...]) -> list[str]:
    return [f"input.{field}" for field in fields]


def _random_digest(seed: int, patient_key: str, trajectory_id: str) -> str:
    serialized = json.dumps(
        [seed, patient_key, trajectory_id],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _random_score(digest: str) -> float:
    # Digests sort ascending; invert the unit-interval value so a higher displayed
    # score still means an earlier rank, as it does for the similarity modes.
    return 1.0 - int(digest, 16) / float((1 << 256) - 1)


def _audit_settings(
    settings: Mapping[str, Any],
    *,
    patient_input: Mapping[str, Any],
    seed: int,
    delegated: bool,
) -> dict[str, Any]:
    ranking = str(settings["ranking"])
    requested_fields = tuple(settings["query_fields"])
    if ranking == "random":
        effective_fields: tuple[str, ...] = ()
        effective_psa_weight = 0.0
        tie_break = ["sha256(seed, excluded_patient_key, trajectory_id)", "trajectory_id"]
    elif ranking == "psa_only":
        effective_fields = ("psa",)
        effective_psa_weight = 1.0
        tie_break = ["trajectory_id"]
    else:
        effective_fields = requested_fields
        effective_psa_weight = (
            float(settings["psa_weight"])
            if "psa" in effective_fields and _numeric_psa(dict(patient_input)) is not None
            else 0.0
        )
        tie_break = [
            "strategy_similarity",
            "trajectory_id",
        ]
    return {
        "ablation_name": settings["name"],
        "requested_ranking": settings["requested_ranking"],
        "ranking": ranking,
        "requested_query_fields": _input_field_labels(requested_fields),
        "query_fields": _input_field_labels(effective_fields),
        "corpus_filter": settings["corpus_filter"],
        "top_k": settings["top_k"],
        "max_trajectories_per_case": settings["max_per_case"],
        "requested_psa_weight": settings["psa_weight"],
        "effective_psa_weight": effective_psa_weight,
        "reason_chars": settings["reason_chars"],
        "permute_outcomes": settings["permute_outcomes"],
        "random_seed": seed if ranking == "random" else None,
        "tie_break": tie_break,
        "leave_one_patient_out": True,
        "delegated_to_completed_trajectory_retriever": delegated,
    }


class AblationRetriever:
    """Wrap a strictly validated corpus and expose controlled retrieval ablations."""

    def __init__(self, base: CompletedTrajectoryRetriever):
        self.base = base
        self._permutation_cache: dict[int, tuple[dict[str, str], str]] = {}

    @classmethod
    def from_output_dir(
        cls,
        source_dir: Path,
        *,
        dataset_path: Path,
    ) -> AblationRetriever:
        """Load only through the production retriever's strict validation path."""
        base = CompletedTrajectoryRetriever.from_output_dir(
            source_dir,
            dataset_path=dataset_path,
        )
        return cls(base)

    def metadata(self) -> dict[str, Any]:
        metadata = dict(self.base.metadata())
        metadata["ablation_retrieval"] = {
            "rankings": sorted(_RANKINGS),
            "query_fields": [f"input.{field}" for field in _QUERY_FIELD_ORDER],
            "corpus_filters": sorted(_CORPUS_FILTERS),
            "leave_one_patient_out_required": True,
            "outcome_permutation": "global_patient_level_sha256_v1",
        }
        return metadata

    def _outcome_permutation(
        self,
        seed: int,
    ) -> tuple[dict[str, str], str]:
        """Return one global, patient-level label permutation for this corpus."""
        cached = self._permutation_cache.get(seed)
        if cached is not None:
            return cached

        observed_by_case: dict[str, str] = {}
        for item in self.base.trajectories:
            observed = str(
                item.payload["evaluation"]["observed_management_category"]
            )
            previous = observed_by_case.setdefault(item.case_id, observed)
            if previous != observed:
                raise ValueError(
                    "Outcome permutation requires one observed label per source "
                    f"patient, but case {item.case_id!r} has both {previous!r} "
                    f"and {observed!r}."
                )
        case_ids = sorted(observed_by_case)
        if not case_ids:
            raise ValueError("Cannot permute outcomes in an empty reference corpus.")

        donor_case_ids = sorted(
            case_ids,
            key=lambda case_id: (
                hashlib.sha256(
                    (
                        f"{seed}:global-patient-outcome-permutation-v1:"
                        f"{case_id}"
                    ).encode("utf-8")
                ).hexdigest(),
                case_id,
            ),
        )
        permutation = {
            target_case_id: observed_by_case[donor_case_id]
            for target_case_id, donor_case_id in zip(
                case_ids,
                donor_case_ids,
                strict=True,
            )
        }
        fingerprint_payload = {
            "algorithm": "global_patient_level_sha256_v1",
            "seed": seed,
            "corpus_fingerprint": self.base.corpus_fingerprint,
            "assignments": [
                {
                    "source_case_id": case_id,
                    "original_observed_management_category": observed_by_case[
                        case_id
                    ],
                    "permuted_observed_management_category": permutation[case_id],
                }
                for case_id in case_ids
            ],
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self._permutation_cache[seed] = (permutation, fingerprint)
        return permutation, fingerprint

    def _add_permuted_outcomes(
        self,
        result: dict[str, Any],
        *,
        seed: int,
    ) -> None:
        permutation, fingerprint = self._outcome_permutation(seed)
        for item in result["results"]:
            source_case_id = item["source_case_id"]
            try:
                permuted = permutation[source_case_id]
            except KeyError as exc:
                raise ValueError(
                    f"Retrieved case {source_case_id!r} is absent from the "
                    "patient-level outcome permutation."
                ) from exc
            item["permuted_observed_management_category"] = permuted
        result["permutation_fingerprint"] = fingerprint

    def search(
        self,
        patient_input: dict[str, Any],
        exclude_patient_key: str,
        available_organs: set[str],
        spec: Any,
        seed: int,
    ) -> dict[str, Any]:
        """Run one ablation while excluding all trajectories of the query patient."""
        if not isinstance(exclude_patient_key, str) or not exclude_patient_key:
            raise ValueError("exclude_patient_key must be a non-empty string.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        settings = _settings(spec)

        if settings["delegate"]:
            result = self.base.search(
                patient_input,
                exclude_patient_key=exclude_patient_key,
                top_k=settings["top_k"],
                max_per_case=settings["max_per_case"],
                psa_weight=settings["psa_weight"],
                reason_chars=settings["reason_chars"],
                available_organs=available_organs,
            )
            # This is intentionally byte-for-byte shape compatible with the
            # production retrieval result.  The full condition is the positive
            # control and is compared directly against base.search by validation.
            if settings["is_full"]:
                return result
            result["ranking"] = settings["ranking"]
            result["corpus_filter"] = settings["corpus_filter"]
            result["retrieval_settings"] = _audit_settings(
                settings,
                patient_input=patient_input,
                seed=seed,
                delegated=True,
            )
            if settings["permute_outcomes"]:
                self._add_permuted_outcomes(result, seed=seed)
            return result

        trajectories = self.base.trajectories
        # LOO is deliberately first and unconditional.  The correctness filter
        # can only remove additional candidates from this already-safe pool.
        leave_one_out_indices = [
            index
            for index, item in enumerate(trajectories)
            if item.patient_key != exclude_patient_key
        ]
        excluded_count = len(trajectories) - len(leave_one_out_indices)
        if not leave_one_out_indices:
            raise ValueError(
                "No completed trajectory remains after excluding the current patient."
            )

        corpus_filter = settings["corpus_filter"]
        if corpus_filter == "all":
            eligible_indices = leave_one_out_indices
        else:
            expected_correct = corpus_filter == "correct"
            eligible_indices = [
                index
                for index in leave_one_out_indices
                if bool(trajectories[index].payload["evaluation"]["correct"])
                is expected_correct
            ]
        corpus_filtered_count = len(leave_one_out_indices) - len(eligible_indices)
        if not eligible_indices:
            raise ValueError(
                "No completed trajectory remains after leave-one-patient-out and "
                f"corpus_filter={corpus_filter!r}."
            )

        ranking = settings["ranking"]
        text_scores: list[float | None] = [None] * len(trajectories)
        strategy_scores: list[float | None] = [None] * len(trajectories)
        psa_scores: list[float | None] = [None] * len(trajectories)
        final_scores: list[float] = [0.0] * len(trajectories)
        random_digests: dict[int, str] = {}

        if ranking in {"similar", "least_similar"}:
            fields = settings["query_fields"]
            query = normalize(
                self.base.vectorizer.transform([_field_text(patient_input, fields)]),
                norm="l2",
            )
            if "report" in fields and "medical_history" in fields:
                candidate_matrix = self.base.matrix
            else:
                documents = [
                    _field_text(item.patient_input, fields) for item in trajectories
                ]
                candidate_matrix = normalize(
                    self.base.vectorizer.transform(documents),
                    norm="l2",
                )
            raw_text_scores = (candidate_matrix @ query.T).toarray().ravel()
            raw_strategy_scores = (
                self.base.strategy_matrix @ query.T
            ).toarray().ravel()
            for index in range(len(trajectories)):
                text_scores[index] = float(raw_text_scores[index])
                strategy_scores[index] = float(raw_strategy_scores[index])
                final_scores[index] = float(raw_text_scores[index])

            query_psa = (
                _numeric_psa(patient_input) if "psa" in fields else None
            )
            if query_psa is not None:
                query_log_psa = math.log1p(query_psa)
                for index, candidate_psa in enumerate(self.base.psa_values):
                    if candidate_psa is None:
                        continue
                    psa_score = math.exp(
                        -abs(query_log_psa - math.log1p(candidate_psa))
                    )
                    psa_scores[index] = psa_score
                    # Match CompletedTrajectoryRetriever exactly: a candidate
                    # lacking numeric PSA retains its unscaled text score.
                    final_scores[index] = (
                        (1.0 - settings["psa_weight"]) * raw_text_scores[index]
                        + settings["psa_weight"] * psa_score
                    )

            direction = -1.0 if ranking == "similar" else 1.0
            eligible_indices.sort(
                key=lambda index: (
                    direction * final_scores[index],
                    direction * float(strategy_scores[index] or 0.0),
                    trajectories[index].trajectory_id,
                )
            )
        elif ranking == "psa_only":
            query_psa = _numeric_psa(patient_input)
            if query_psa is not None:
                query_log_psa = math.log1p(query_psa)
                for index, candidate_psa in enumerate(self.base.psa_values):
                    if candidate_psa is None:
                        continue
                    psa_scores[index] = math.exp(
                        -abs(query_log_psa - math.log1p(candidate_psa))
                    )
                    final_scores[index] = float(psa_scores[index])
            eligible_indices.sort(
                key=lambda index: (
                    psa_scores[index] is None,
                    -float(psa_scores[index] or 0.0),
                    trajectories[index].trajectory_id,
                )
            )
        else:
            for index in eligible_indices:
                digest = _random_digest(
                    seed,
                    exclude_patient_key,
                    trajectories[index].trajectory_id,
                )
                random_digests[index] = digest
                final_scores[index] = _random_score(digest)
            eligible_indices.sort(
                key=lambda index: (
                    random_digests[index],
                    trajectories[index].trajectory_id,
                )
            )

        selected: list[tuple[int, Any]] = []
        per_case: Counter[str] = Counter()
        for index in eligible_indices:
            item = trajectories[index]
            if per_case[item.case_id] >= settings["max_per_case"]:
                continue
            selected.append((index, item))
            per_case[item.case_id] += 1
            if len(selected) == settings["top_k"]:
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
            components: dict[str, Any] = {
                "text_similarity": text_scores[index],
                "psa_similarity": psa_scores[index],
                "psa_weight": (
                    1.0
                    if ranking == "psa_only" and psa_scores[index] is not None
                    else (
                        settings["psa_weight"]
                        if psa_scores[index] is not None
                        else 0.0
                    )
                ),
                "strategy_similarity_tie_break": strategy_scores[index],
            }
            if ranking == "random":
                components["random_sha256"] = random_digests[index]
            results.append(
                {
                    "evidence_id": f"CASE-{rank:03d}",
                    "source_case_id": item.case_id,
                    "source_trajectory_id": item.trajectory_id,
                    "retrieval_score": score,
                    "retrieval_components": components,
                    "historical_patient_input": item.patient_input,
                    "historical_evidence_selection": {
                        "suv_organs": selected_organs,
                        "literature_query": selection["literature_query"],
                    },
                    "historical_retrieved_suv_organs": [
                        evidence.get("organ")
                        for evidence in retrieved["suv"]
                        if isinstance(evidence, dict)
                    ],
                    "historical_literature": _compact_literature(
                        retrieved["literature"]
                    ),
                    "historical_prediction": {
                        "answer": prediction["answer"],
                        "reason": _historical_reason(
                            prediction["reason"],
                            settings["reason_chars"],
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

        if ranking == "random":
            effective_fields: tuple[str, ...] = ()
            effective_psa_weight = 0.0
        elif ranking == "psa_only":
            effective_fields = ("psa",)
            effective_psa_weight = 1.0
        else:
            effective_fields = settings["query_fields"]
            effective_psa_weight = (
                settings["psa_weight"]
                if "psa" in effective_fields
                and _numeric_psa(patient_input) is not None
                else 0.0
            )

        result = {
            "source_dir": str(self.base.source_dir),
            "corpus_fingerprint": self.base.corpus_fingerprint,
            "query_fields": _input_field_labels(effective_fields),
            "excluded_current_patient": True,
            "excluded_completed_trajectories": excluded_count,
            "eligible_before_corpus_filter": len(leave_one_out_indices),
            "corpus_filtered_out_trajectories": corpus_filtered_count,
            "eligible_completed_trajectories": len(eligible_indices),
            "top_k": settings["top_k"],
            "max_trajectories_per_case": settings["max_per_case"],
            "psa_weight": effective_psa_weight,
            "ranking": ranking,
            "corpus_filter": corpus_filter,
            "retrieval_settings": _audit_settings(
                settings,
                patient_input=patient_input,
                seed=seed,
                delegated=False,
            ),
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
        if settings["permute_outcomes"]:
            self._add_permuted_outcomes(result, seed=seed)
        return result
