"""Load and validate the pre-registered TMI ablation matrix."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY = Path(__file__).with_name("experiments.json")
ALLOWED_RANKINGS = {"similar", "random", "least_similar", "psa_only"}
ALLOWED_QUERY_FIELDS = {"report", "medical_history", "psa"}
ALLOWED_CORPUS_FILTERS = {"all", "correct", "incorrect"}
ALLOWED_PLANNER_FIELDS = {
    "organ_hints",
    "literature_query_hints",
    "patient_input",
    "evidence_selection",
    "literature_sources",
}
ALLOWED_FINAL_FIELDS = {
    "patient_input",
    "evidence_selection",
    "retrieved_suv_organs",
    "literature",
    "prediction_answer",
    "prediction_reason",
    "observed_outcome",
    "prediction_correctness",
}


@dataclass(frozen=True)
class AblationSpec:
    """One immutable experimental condition."""

    id: str
    group: str
    role: str
    description: str
    use_retrieval: bool
    ranking: str
    query_fields: tuple[str, ...]
    corpus_filter: str
    planner_fields: tuple[str, ...]
    final_fields: tuple[str, ...]
    top_k: int
    max_per_case: int
    psa_weight: float
    permute_outcomes: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["query_fields"] = list(self.query_fields)
        payload["planner_fields"] = list(self.planner_fields)
        payload["final_fields"] = list(self.final_fields)
        return payload

    @property
    def planner_enabled(self) -> bool:
        return self.use_retrieval and bool(self.planner_fields)

    @property
    def final_enabled(self) -> bool:
        return self.use_retrieval and bool(self.final_fields)


@dataclass(frozen=True)
class ExperimentRegistry:
    schema_version: str
    study_name: str
    primary_endpoint: str
    manifest_path: Path
    manifest_sha256: str
    experiments: dict[str, AblationSpec]

    def resolve(self, experiment_id: str) -> AblationSpec:
        try:
            return self.experiments[experiment_id]
        except KeyError as exc:
            available = ", ".join(self.experiments)
            raise KeyError(
                f"Unknown experiment ID {experiment_id!r}; choose one of: {available}"
            ) from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_keys(payload: dict[str, Any], keys: set[str], description: str) -> None:
    missing = sorted(keys - set(payload))
    if missing:
        raise ValueError(f"{description} is missing fields: {missing}")


def _string_tuple(
    value: Any,
    *,
    allowed: set[str],
    description: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise TypeError(f"{description} must be {qualifier}")
    if any(not isinstance(item, str) for item in value):
        raise TypeError(f"{description} entries must be strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{description} contains duplicates")
    unknown = sorted(set(result) - allowed)
    if unknown:
        raise ValueError(f"{description} contains unknown values: {unknown}")
    return result


def _validate_spec(payload: dict[str, Any]) -> AblationSpec:
    required = {
        "id",
        "group",
        "role",
        "description",
        "use_retrieval",
        "ranking",
        "query_fields",
        "corpus_filter",
        "planner_fields",
        "final_fields",
        "top_k",
        "max_per_case",
        "psa_weight",
        "permute_outcomes",
    }
    _require_keys(payload, required, "experiment")
    experiment_id = payload["id"]
    if not isinstance(experiment_id, str) or not re.fullmatch(
        r"[a-z][a-z0-9_]*", experiment_id
    ):
        raise ValueError(f"Invalid experiment ID: {experiment_id!r}")
    for key in ("group", "role", "description"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise TypeError(f"{experiment_id}.{key} must be a non-empty string")
    if not isinstance(payload["use_retrieval"], bool):
        raise TypeError(f"{experiment_id}.use_retrieval must be boolean")
    if payload["ranking"] not in ALLOWED_RANKINGS:
        raise ValueError(f"{experiment_id}.ranking is invalid")
    if payload["corpus_filter"] not in ALLOWED_CORPUS_FILTERS:
        raise ValueError(f"{experiment_id}.corpus_filter is invalid")
    query_fields = _string_tuple(
        payload["query_fields"],
        allowed=ALLOWED_QUERY_FIELDS,
        description=f"{experiment_id}.query_fields",
        allow_empty=False,
    )
    planner_fields = _string_tuple(
        payload["planner_fields"],
        allowed=ALLOWED_PLANNER_FIELDS,
        description=f"{experiment_id}.planner_fields",
        allow_empty=True,
    )
    final_fields = _string_tuple(
        payload["final_fields"],
        allowed=ALLOWED_FINAL_FIELDS,
        description=f"{experiment_id}.final_fields",
        allow_empty=True,
    )
    for key in ("top_k", "max_per_case"):
        if isinstance(payload[key], bool) or not isinstance(payload[key], int):
            raise TypeError(f"{experiment_id}.{key} must be an integer")
        if payload[key] < 1:
            raise ValueError(f"{experiment_id}.{key} must be positive")
    weight = payload["psa_weight"]
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise TypeError(f"{experiment_id}.psa_weight must be numeric")
    if not 0 <= float(weight) <= 1:
        raise ValueError(f"{experiment_id}.psa_weight must be in [0, 1]")
    if not isinstance(payload["permute_outcomes"], bool):
        raise TypeError(f"{experiment_id}.permute_outcomes must be boolean")
    if not payload["use_retrieval"] and (planner_fields or final_fields):
        raise ValueError(
            f"{experiment_id} disables retrieval but still exposes historical fields"
        )
    if "prediction_correctness" in final_fields and not {
        "prediction_answer",
        "observed_outcome",
    }.issubset(final_fields):
        raise ValueError(
            f"{experiment_id} exposes correctness without prediction and outcome"
        )
    if "prediction_reason" in final_fields and "prediction_answer" not in final_fields:
        raise ValueError(
            f"{experiment_id} exposes prediction reason without prediction answer"
        )
    if payload["permute_outcomes"] and "observed_outcome" not in final_fields:
        raise ValueError(
            f"{experiment_id} permutes outcomes but does not expose outcomes"
        )
    if payload["ranking"] == "psa_only" and query_fields != ("psa",):
        raise ValueError(f"{experiment_id} PSA-only ranking requires only PSA")
    return AblationSpec(
        id=experiment_id,
        group=payload["group"].strip(),
        role=payload["role"].strip(),
        description=payload["description"].strip(),
        use_retrieval=payload["use_retrieval"],
        ranking=payload["ranking"],
        query_fields=query_fields,
        corpus_filter=payload["corpus_filter"],
        planner_fields=planner_fields,
        final_fields=final_fields,
        top_k=payload["top_k"],
        max_per_case=payload["max_per_case"],
        psa_weight=float(weight),
        permute_outcomes=payload["permute_outcomes"],
    )


def load_experiment_registry(
    path: Path = DEFAULT_REGISTRY,
) -> ExperimentRegistry:
    path = path.resolve()
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("Experiment registry must be a JSON object")
    _require_keys(
        payload,
        {
            "schema_version",
            "study_name",
            "primary_endpoint",
            "defaults",
            "experiments",
        },
        "registry",
    )
    defaults = payload["defaults"]
    experiments = payload["experiments"]
    if not isinstance(defaults, dict):
        raise TypeError("registry.defaults must be an object")
    if not isinstance(experiments, list) or not experiments:
        raise TypeError("registry.experiments must be a non-empty list")
    resolved: dict[str, AblationSpec] = {}
    for entry in experiments:
        if not isinstance(entry, dict):
            raise TypeError("Each experiment must be an object")
        spec = _validate_spec({**defaults, **entry})
        if spec.id in resolved:
            raise ValueError(f"Duplicate experiment ID: {spec.id}")
        resolved[spec.id] = spec
    return ExperimentRegistry(
        schema_version=str(payload["schema_version"]),
        study_name=str(payload["study_name"]),
        primary_endpoint=str(payload["primary_endpoint"]),
        manifest_path=path,
        manifest_sha256=_sha256_file(path),
        experiments=resolved,
    )
