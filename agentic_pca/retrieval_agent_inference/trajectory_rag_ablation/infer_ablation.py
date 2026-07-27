#!/usr/bin/env python3
"""Run one pre-registered trajectory-RAG ablation with strict JSON inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_pca.retrieval_agent_inference.infer import infer as base  # noqa: E402
from agentic_pca.retrieval_agent_inference.trajectory_rag_ablation.ablation_retriever import (  # noqa: E402
    AblationRetriever,
)
from agentic_pca.retrieval_agent_inference.trajectory_rag_ablation.ablation_specs import (  # noqa: E402
    DEFAULT_REGISTRY,
    AblationSpec,
    ExperimentRegistry,
    load_experiment_registry,
)
from agentic_pca.retrieval_agent_inference.trajectory_rag_ablation.contexts import (  # noqa: E402
    build_final_case_context,
    build_planner_context,
    final_system_prompt,
    planner_system_prompt,
    retrieval_audit_view,
)


DEFAULT_SOURCE_RUN = (
    ROOT
    / "agentic_pca/retrieval_agent_inference/outputs/full_run_Qwen3.5_9B"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "agentic_pca/retrieval_agent_inference/outputs"
    / "trajectory_rag_ablation_Qwen3.5-9B_on_Qwen3.5-9B"
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id")
    parser.add_argument("--experiments-file", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--list-experiments", action="store_true")
    parser.add_argument("--dataset", type=Path, default=base.DEFAULT_DATASET)
    parser.add_argument("--suv-dir", type=Path, default=base.DEFAULT_SUV_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--pdf-cache", type=Path, default=base.DEFAULT_CACHE)
    parser.add_argument("--model-path", type=Path, default=base.DEFAULT_MODEL)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--trajectory-rag-dir", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-trajectories", type=int, default=5)
    parser.add_argument("--max-json-retries", type=int, default=3)
    parser.add_argument("--max-suv-organs", type=int, default=6)
    parser.add_argument("--literature-top-k", type=int, default=3)
    parser.add_argument("--pdf-snippet-chars", type=int, default=1800)
    parser.add_argument("--chunk-words", type=int, default=350)
    parser.add_argument("--overlap-words", type=int, default=70)
    parser.add_argument("--rebuild-pdf-index", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=30000)
    parser.add_argument("--planner-max-new-tokens", type=int, default=500)
    parser.add_argument("--final-max-new-tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--ablation-seed", type=int, default=20260725)
    parser.add_argument("--case-id-salt", default="retrieval-agent-inference-v1")
    parser.add_argument("--trajectory-rag-reason-chars", type=int, default=1200)
    parser.add_argument(
        "--patient",
        action="append",
        help="Exact dataset patient key; repeatable.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--invalid-report-policy",
        choices=("fail", "warn"),
        default="warn",
        help="The matched ablation default is warn, as in the main trajectory-RAG run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock-model", action="store_true")
    parser.add_argument("--mock-invalid-first", action="store_true")
    return parser.parse_args()


def resolve_spec(
    args: argparse.Namespace,
) -> tuple[ExperimentRegistry, AblationSpec]:
    registry = load_experiment_registry(args.experiments_file)
    if args.list_experiments:
        print(
            json.dumps(
                {
                    "schema_version": registry.schema_version,
                    "experiments": [
                        spec.to_dict() for spec in registry.experiments.values()
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(0)
    if not args.experiment_id:
        raise ValueError("--experiment-id is required unless --list-experiments is used")
    spec = registry.resolve(args.experiment_id)
    # The base validator and config builder consume these names. Their values
    # are resolved only from the immutable experiment registry.
    args.trajectory_rag_top_k = spec.top_k
    args.trajectory_rag_max_per_case = spec.max_per_case
    args.trajectory_rag_psa_weight = spec.psa_weight
    return registry, spec


def validate_ablation_args(args: argparse.Namespace) -> None:
    base.validate_args(args)
    if args.ablation_seed < 0:
        raise ValueError("--ablation-seed cannot be negative")
    if args.trajectory_rag_dir is None:
        raise ValueError("--trajectory-rag-dir is required for corpus provenance")


def ablation_config(
    args: argparse.Namespace,
    registry: ExperimentRegistry,
    spec: AblationSpec,
    pdf_chunks_fingerprint: str,
    trajectory_rag_metadata: dict[str, Any],
) -> dict[str, Any]:
    config = base.inference_config(
        args,
        pdf_chunks_fingerprint,
        trajectory_rag_metadata,
    )
    planner_prompt = planner_system_prompt(spec)
    prediction_prompt = final_system_prompt(spec)
    config.update(
        {
            "entrypoint": str(Path(__file__).resolve()),
            "ablation_schema_version": registry.schema_version,
            "ablation_registry": str(registry.manifest_path),
            "ablation_registry_sha256": registry.manifest_sha256,
            "ablation_spec": spec.to_dict(),
            "ablation_seed": args.ablation_seed,
            "ablation_runner_sha256": base.sha256_file(Path(__file__)),
            "ablation_specs_code_sha256": base.sha256_file(
                Path(__file__).with_name("ablation_specs.py")
            ),
            "ablation_retriever_code_sha256": base.sha256_file(
                Path(__file__).with_name("ablation_retriever.py")
            ),
            "ablation_context_code_sha256": base.sha256_file(
                Path(__file__).with_name("contexts.py")
            ),
            "evidence_selection_prompt_sha256": sha256_text(planner_prompt),
            "final_prediction_prompt_sha256": sha256_text(prediction_prompt),
            "historical_context_controls": {
                "planner_enabled": spec.planner_enabled,
                "final_enabled": spec.final_enabled,
                "case_citation_required": spec.final_enabled,
                "historical_outcomes_exposed": (
                    "observed_outcome" in spec.final_fields
                ),
                "historical_correctness_exposed": (
                    "prediction_correctness" in spec.final_fields
                ),
            },
        }
    )
    return config


def generate_ablation_trajectory(
    generator: base.LocalGenerator | base.MockGenerator,
    pdf_retriever: base.PdfRetriever,
    *,
    case_id: str,
    trajectory_number: int,
    patient_input: dict[str, Any],
    input_warnings: list[str],
    suv_by_roi: dict[str, dict[str, dict[str, Any]]],
    suv_data_fingerprint: str,
    args: argparse.Namespace,
    base_seed: int,
    inference_config_fingerprint: str,
    spec: AblationSpec,
    retrieval: dict[str, Any] | None,
) -> dict[str, Any]:
    trajectory_id = f"{case_id}_trajectory_{trajectory_number:03d}"
    organs = base.common_suv_organs(suv_by_roi)
    fingerprint = base.trajectory_fingerprint(
        case_id,
        trajectory_number,
        patient_input,
        suv_data_fingerprint,
        inference_config_fingerprint,
    )
    planner_context = build_planner_context(retrieval, spec)
    final_cases, projection_audit = build_final_case_context(
        retrieval,
        spec,
        current_case_id=case_id,
        ablation_seed=args.ablation_seed,
    )
    retrieval_audit = retrieval_audit_view(retrieval)
    common_metadata: dict[str, Any] = {
        "schema_version": base.SCHEMA_VERSION,
        "trajectory_id": trajectory_id,
        "case_id": case_id,
        "trajectory_number": trajectory_number,
        "trajectory_fingerprint": fingerprint,
        "created_at_utc": base.utc_now(),
        "status": "failed",
        "input": patient_input,
        "input_warnings": input_warnings,
        "trajectory_rag": retrieval_audit,
        "ablation": {
            "experiment_id": spec.id,
            "spec": spec.to_dict(),
            "planner_context": planner_context,
            "final_case_context": final_cases,
            "projection_audit": projection_audit,
        },
        "evidence_selection": None,
        "retrieved_evidence": None,
        "prediction": None,
        "evaluation": None,
        "result": None,
        "failure": None,
        "provenance": {
            "model_path": str(args.model_path),
            "base_seed": base_seed,
            "ablation_seed": args.ablation_seed,
            "suv_data_fingerprint": suv_data_fingerprint,
            "label_mapping_version": base.LABEL_MAPPING_VERSION,
            "inference_config_fingerprint": inference_config_fingerprint,
            "treatment_blind_generation": True,
            "target_outcome_blinded": True,
            "reference_outcomes_available_in_corpus": retrieval is not None,
            "reference_outcomes_exposed_to_model": (
                "observed_outcome" in spec.final_fields
            ),
        },
    }
    if (
        "report_is_xml_placeholder" in input_warnings
        and args.invalid_report_policy == "fail"
    ):
        common_metadata["failure"] = {
            "stage": "input_validation",
            "error": (
                "Report is the placeholder System.Xml.XmlElement. "
                "Use --invalid-report-policy warn only for an explicit "
                "history-and-PSA fallback."
            ),
        }
        return common_metadata

    try:
        planner_payload: dict[str, Any] = {
            "task": "select_evidence",
            "current_patient_input": patient_input,
            "available_suv_organs": organs,
            "maximum_suv_organs": args.max_suv_organs,
        }
        if planner_context is not None:
            planner_payload["trajectory_rag"] = planner_context
        request_model, request_attempts = base.call_strict_json(
            generator,
            stage="evidence_selection",
            system_prompt=planner_system_prompt(spec),
            base_payload=planner_payload,
            validator=lambda parsed: base.validate_evidence_request(
                parsed,
                set(organs),
                args.max_suv_organs,
            ),
            schema=base.EvidenceRequest.model_json_schema(),
            args=args,
            base_seed=base_seed,
            max_new_tokens=args.planner_max_new_tokens,
        )
    except base.StrictJSONGenerationError as exc:
        common_metadata["evidence_selection"] = {
            "attempts": exc.attempts,
            "accepted": None,
        }
        common_metadata["failure"] = {"stage": exc.stage, "error": str(exc)}
        return common_metadata

    request = base.EvidenceRequest.model_validate(
        request_model.model_dump(),
        strict=True,
    )
    suv_evidence = base.retrieve_suv_evidence(request, suv_by_roi)
    literature_evidence = base.retrieve_literature_evidence(
        request,
        pdf_retriever,
        args.literature_top_k,
        args.pdf_snippet_chars,
    )
    retrieved: dict[str, list[dict[str, Any]]] = {
        "suv": suv_evidence,
        "literature": literature_evidence,
    }
    if final_cases:
        retrieved["similar_trajectories"] = final_cases
    valid_evidence_ids = {
        item["evidence_id"]
        for collection in retrieved.values()
        for item in collection
    }
    common_metadata["evidence_selection"] = {
        "attempts": request_attempts,
        "accepted": request.model_dump(mode="json"),
    }
    common_metadata["retrieved_evidence"] = retrieved

    try:
        prediction_model, prediction_attempts = base.call_strict_json(
            generator,
            stage="final_prediction",
            system_prompt=final_system_prompt(spec),
            base_payload={
                "task": "predict_management",
                "current_patient_input": patient_input,
                "evidence_request": request.model_dump(mode="json"),
                "retrieved_evidence": retrieved,
                "allowed_management_categories": list(
                    base.MANAGEMENT_CATEGORIES
                ),
            },
            validator=lambda parsed: base.validate_final_prediction(
                parsed,
                valid_evidence_ids,
                require_case_evidence=bool(final_cases),
            ),
            schema=base.FinalPrediction.model_json_schema(),
            args=args,
            base_seed=base_seed + 10_000,
            max_new_tokens=args.final_max_new_tokens,
        )
    except base.StrictJSONGenerationError as exc:
        common_metadata["prediction"] = {
            "attempts": exc.attempts,
            "accepted": None,
        }
        common_metadata["failure"] = {"stage": exc.stage, "error": str(exc)}
        return common_metadata

    prediction = base.FinalPrediction.model_validate(
        prediction_model.model_dump(),
        strict=True,
    )
    common_metadata["prediction"] = {
        "attempts": prediction_attempts,
        "accepted": prediction.model_dump(mode="json"),
    }
    common_metadata["status"] = "prediction_frozen"
    return common_metadata


def load_completed_if_reusable(
    path: Path,
    *,
    expected_fingerprint: str,
    expected_spec: AblationSpec,
    expected_retrieval: dict[str, Any] | None,
    current_treatment: Any,
    suv_by_roi: dict[str, dict[str, dict[str, Any]]],
    pdf_retriever: base.PdfRetriever,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not path.exists() or args.overwrite:
        return None
    try:
        existing = base.load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Existing trajectory is not valid JSON: {path}") from exc
    if not isinstance(existing, dict):
        raise ValueError(f"Existing trajectory top level is not an object: {path}")
    if existing.get("trajectory_fingerprint") != expected_fingerprint:
        raise ValueError(
            f"Existing trajectory fingerprint differs: {path}. "
            "Use --overwrite or a new --output-dir."
        )
    if existing.get("status") != "completed":
        return None
    errors: list[str] = []
    ablation = existing.get("ablation")
    if not isinstance(ablation, dict):
        errors.append("ablation metadata is missing")
    else:
        if ablation.get("experiment_id") != expected_spec.id:
            errors.append("ablation experiment ID differs")
        expected_planner = build_planner_context(expected_retrieval, expected_spec)
        expected_cases, expected_projection = build_final_case_context(
            expected_retrieval,
            expected_spec,
            current_case_id=str(existing.get("case_id")),
            ablation_seed=args.ablation_seed,
        )
        if ablation.get("planner_context") != expected_planner:
            errors.append("planner context differs from current projection")
        if ablation.get("final_case_context") != expected_cases:
            errors.append("final case context differs from current projection")
        if ablation.get("projection_audit") != expected_projection:
            errors.append("projection audit differs")
    if existing.get("trajectory_rag") != retrieval_audit_view(expected_retrieval):
        errors.append("retrieval audit differs from current corpus/search")

    accepted_request = (
        existing.get("evidence_selection", {}).get("accepted")
        if isinstance(existing.get("evidence_selection"), dict)
        else None
    )
    available_organs = set(base.common_suv_organs(suv_by_roi))
    request: base.EvidenceRequest | None = None
    if not isinstance(accepted_request, dict):
        errors.append("accepted evidence request is missing")
    else:
        request, request_errors = base.validate_evidence_request(
            accepted_request,
            available_organs,
            args.max_suv_organs,
        )
        errors.extend(f"evidence request: {error}" for error in request_errors)
    retrieved = existing.get("retrieved_evidence")
    valid_ids: set[str] = set()
    expected_cases, _ = build_final_case_context(
        expected_retrieval,
        expected_spec,
        current_case_id=str(existing.get("case_id")),
        ablation_seed=args.ablation_seed,
    )
    if not isinstance(retrieved, dict):
        errors.append("retrieved_evidence is missing")
    elif request is not None:
        expected_suv = base.retrieve_suv_evidence(request, suv_by_roi)
        expected_literature = base.retrieve_literature_evidence(
            request,
            pdf_retriever,
            args.literature_top_k,
            args.pdf_snippet_chars,
        )
        if retrieved.get("suv") != expected_suv:
            errors.append("retrieved SUV evidence differs from source")
        if retrieved.get("literature") != expected_literature:
            errors.append("retrieved literature differs from source")
        if expected_cases:
            if retrieved.get("similar_trajectories") != expected_cases:
                errors.append("final similar-case evidence differs")
        elif "similar_trajectories" in retrieved:
            errors.append("unexpected final similar-case evidence")
        for key in ("suv", "literature", "similar_trajectories"):
            items = retrieved.get(key, [])
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and isinstance(
                        item.get("evidence_id"), str
                    ):
                        valid_ids.add(item["evidence_id"])

    accepted_prediction = (
        existing.get("prediction", {}).get("accepted")
        if isinstance(existing.get("prediction"), dict)
        else None
    )
    prediction: base.FinalPrediction | None = None
    if not isinstance(accepted_prediction, dict):
        errors.append("accepted prediction is missing")
    else:
        prediction, prediction_errors = base.validate_final_prediction(
            accepted_prediction,
            valid_ids,
            require_case_evidence=bool(expected_cases),
        )
        errors.extend(f"prediction: {error}" for error in prediction_errors)
    evaluation = existing.get("evaluation")
    if prediction is None or not isinstance(evaluation, dict):
        errors.append("evaluation cannot be validated")
    else:
        expected_observed = base.treatment_to_category(current_treatment)
        expected_correct = prediction.answer == expected_observed
        if evaluation.get("observed_treatment") != current_treatment:
            errors.append("observed treatment differs from dataset")
        if evaluation.get("observed_management_category") != expected_observed:
            errors.append("observed category differs from dataset")
        if evaluation.get("correct") != expected_correct:
            errors.append("correctness differs from prediction/outcome")
        expected_result = {
            "answer": prediction.answer,
            "reason": prediction.reason,
            "evidence_ids": prediction.evidence_ids,
            "observed_treatment": current_treatment,
            "observed_management_category": expected_observed,
            "correct": expected_correct,
        }
        if existing.get("result") != expected_result:
            errors.append("result differs from prediction/evaluation")
    if errors:
        raise ValueError(
            f"Existing completed trajectory failed validation: {path}: "
            + "; ".join(errors)
            + ". Use --overwrite to regenerate it."
        )
    return existing


def dry_run_payload(
    args: argparse.Namespace,
    registry: ExperimentRegistry,
    spec: AblationSpec,
    dataset: dict[str, dict[str, Any]],
    patients: list[str],
    labels: Counter[str],
    retriever: AblationRetriever,
) -> dict[str, Any]:
    return {
        "status": "dry_run_ok",
        "ablation_schema_version": registry.schema_version,
        "experiment": spec.to_dict(),
        "registry_sha256": registry.manifest_sha256,
        "dataset_patients": len(dataset),
        "selected_patients": len(patients),
        "trajectories_per_patient": args.num_trajectories,
        "selected_label_distribution": dict(labels),
        "invalid_report_placeholders": sum(
            str(dataset[patient].get("Report", "")).strip()
            == "System.Xml.XmlElement"
            for patient in patients
        ),
        "invalid_report_policy": args.invalid_report_policy,
        "pdf_files": len(base.pdf_manifest(args.pdf_dir)),
        "trajectory_rag": retriever.metadata(),
        "model_will_load": False,
    }


def main() -> int:
    args = parse_args()
    registry, spec = resolve_spec(args)
    validate_ablation_args(args)
    dataset = base.load_json(args.dataset)
    if not isinstance(dataset, dict):
        raise TypeError("Dataset must be a JSON object keyed by patient")
    patients = base.select_patients(dataset, args)
    labels = base.validate_dataset_records(dataset, patients, args.suv_dir)
    ablation_retriever = AblationRetriever.from_output_dir(
        args.trajectory_rag_dir,
        dataset_path=args.dataset,
    )
    metadata = ablation_retriever.metadata()
    print(
        f"Experiment {spec.id}: indexed "
        f"{metadata['indexed_completed_trajectories']} completed trajectories "
        f"from {metadata['indexed_patients']} patients.",
        flush=True,
    )
    if args.dry_run:
        print(
            json.dumps(
                dry_run_payload(
                    args,
                    registry,
                    spec,
                    dataset,
                    patients,
                    labels,
                    ablation_retriever,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    chunks = base.load_or_build_chunks(
        args.pdf_dir,
        args.pdf_cache,
        args.chunk_words,
        args.overlap_words,
        args.rebuild_pdf_index,
    )
    pdf_chunks_fingerprint = base.sha256_json(
        [
            {"source": chunk.source, "page": chunk.page, "text": chunk.text}
            for chunk in chunks
        ]
    )
    pdf_retriever = base.PdfRetriever(chunks)
    print(
        f"Indexed {len(chunks)} chunks from "
        f"{len(base.pdf_manifest(args.pdf_dir))} PDF files.",
        flush=True,
    )

    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / spec.id)
    output_resolved = output_dir.resolve()
    source_resolved = args.trajectory_rag_dir.resolve()
    if (
        output_resolved == source_resolved
        or source_resolved in output_resolved.parents
        or output_resolved in source_resolved.parents
    ):
        raise ValueError(
            "--output-dir and --trajectory-rag-dir must be separate, "
            "non-nested directories"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "patients").mkdir(exist_ok=True)

    config = ablation_config(
        args,
        registry,
        spec,
        pdf_chunks_fingerprint,
        metadata,
    )
    config_fingerprint = base.sha256_json(config)
    config_payload = {
        "last_invocation_at_utc": base.utc_now(),
        "inference_config_fingerprint": config_fingerprint,
        "inference_config": config,
        "selection": {
            "selected_patients": len(patients),
            "num_trajectories": args.num_trajectories,
            "start_index": args.start_index,
            "max_patients": args.max_patients,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        },
    }
    config_path = output_dir / "config.json"
    existing_files = list(
        (output_dir / "patients").glob("case_*/trajectory_*.json")
    )
    if config_path.exists():
        previous = base.load_json(config_path)
        if previous.get("inference_config_fingerprint") != config_fingerprint:
            if existing_files:
                raise ValueError(
                    f"Inference config differs from existing {config_path}; "
                    "mixed-config runs are forbidden. Use a new output directory."
                )
            if not args.overwrite:
                raise ValueError(
                    f"Config differs from existing empty {config_path}; "
                    "use --overwrite or a new output directory."
                )
    elif existing_files:
        raise ValueError(
            f"{output_dir} contains trajectories but no config.json; "
            "use a new output directory."
        )
    base.atomic_write_json(config_path, config_payload)
    case_mapping = {
        base.stable_case_id(patient, args.case_id_salt): patient
        for patient in patients
    }
    base.update_manifest(output_dir, case_mapping)

    generator: base.LocalGenerator | base.MockGenerator
    if args.mock_model:
        generator = base.MockGenerator(invalid_first=args.mock_invalid_first)
    else:
        generator = base.LocalGenerator(args.model_path, args.device)

    any_failed = False
    total = len(patients) * args.num_trajectories
    progress = 0
    for patient_key in patients:
        case_id = base.stable_case_id(patient_key, args.case_id_salt)
        record = dataset[patient_key]
        patient_input, input_warnings = base.build_patient_input(record)
        suv_by_roi = base.load_suv_by_roi(patient_key, args.suv_dir)
        suv_data_fingerprint = base.sha256_json(suv_by_roi)
        available_organs = set(base.common_suv_organs(suv_by_roi))
        retrieval: dict[str, Any] | None = None
        if spec.use_retrieval:
            retrieval = ablation_retriever.search(
                patient_input,
                exclude_patient_key=patient_key,
                available_organs=available_organs,
                spec=spec,
                seed=args.ablation_seed,
            )
        for trajectory_number in range(1, args.num_trajectories + 1):
            progress += 1
            output_path = (
                output_dir
                / "patients"
                / case_id
                / f"trajectory_{trajectory_number:03d}.json"
            )
            expected_fingerprint = base.trajectory_fingerprint(
                case_id,
                trajectory_number,
                patient_input,
                suv_data_fingerprint,
                config_fingerprint,
            )
            existing = load_completed_if_reusable(
                output_path,
                expected_fingerprint=expected_fingerprint,
                expected_spec=spec,
                expected_retrieval=retrieval,
                current_treatment=record["Treatment"],
                suv_by_roi=suv_by_roi,
                pdf_retriever=pdf_retriever,
                args=args,
            )
            if existing is not None:
                print(
                    f"[{progress}/{total}] {case_id} trajectory "
                    f"{trajectory_number}: already completed; skipping",
                    flush=True,
                )
                continue
            print(
                f"[{progress}/{total}] {case_id} trajectory "
                f"{trajectory_number}: generating {spec.id}",
                flush=True,
            )
            base_seed = base.stable_trajectory_seed(
                args.seed,
                case_id,
                trajectory_number,
            )
            try:
                trajectory = generate_ablation_trajectory(
                    generator,
                    pdf_retriever,
                    case_id=case_id,
                    trajectory_number=trajectory_number,
                    patient_input=patient_input,
                    input_warnings=input_warnings,
                    suv_by_roi=suv_by_roi,
                    suv_data_fingerprint=suv_data_fingerprint,
                    args=args,
                    base_seed=base_seed,
                    inference_config_fingerprint=config_fingerprint,
                    spec=spec,
                    retrieval=retrieval,
                )
                trajectory = base.attach_evaluation(
                    trajectory,
                    record["Treatment"],
                )
                base.atomic_write_json(output_path, trajectory)
                if trajectory["status"] != "completed":
                    any_failed = True
                    print(
                        f"[{progress}/{total}] {trajectory['trajectory_id']}: "
                        f"failed at {trajectory['failure']['stage']}",
                        flush=True,
                    )
                    if args.fail_fast:
                        raise base.PersistedTrajectoryFailure(
                            trajectory["failure"]["error"]
                        )
                else:
                    evaluation = trajectory["evaluation"]
                    print(
                        f"[{progress}/{total}] {trajectory['trajectory_id']}: "
                        f"answer={trajectory['prediction']['accepted']['answer']}, "
                        f"correct={evaluation['correct']}",
                        flush=True,
                    )
            except base.PersistedTrajectoryFailure:
                raise
            except Exception as exc:
                any_failed = True
                failure = {
                    "schema_version": base.SCHEMA_VERSION,
                    "trajectory_id": (
                        f"{case_id}_trajectory_{trajectory_number:03d}"
                    ),
                    "case_id": case_id,
                    "trajectory_number": trajectory_number,
                    "trajectory_fingerprint": expected_fingerprint,
                    "created_at_utc": base.utc_now(),
                    "status": "failed",
                    "input": patient_input,
                    "input_warnings": input_warnings,
                    "trajectory_rag": retrieval_audit_view(retrieval),
                    "ablation": {
                        "experiment_id": spec.id,
                        "spec": spec.to_dict(),
                    },
                    "evidence_selection": None,
                    "retrieved_evidence": None,
                    "prediction": None,
                    "evaluation": None,
                    "result": None,
                    "failure": {
                        "stage": "orchestration",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    "provenance": {
                        "model_path": str(args.model_path),
                        "base_seed": base_seed,
                        "ablation_seed": args.ablation_seed,
                        "suv_data_fingerprint": suv_data_fingerprint,
                        "label_mapping_version": base.LABEL_MAPPING_VERSION,
                        "inference_config_fingerprint": config_fingerprint,
                        "treatment_blind_generation": True,
                        "target_outcome_blinded": True,
                    },
                }
                base.atomic_write_json(output_path, failure)
                print(
                    f"[{progress}/{total}] {case_id}: failed: {exc}",
                    flush=True,
                )
                if args.fail_fast:
                    raise
        base.rebuild_summary(output_dir)

    summary = base.rebuild_summary(output_dir)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Run directory: {output_dir}", flush=True)
    failed_in_run = int(summary["status_counts"].get("failed", 0))
    return 1 if any_failed or failed_in_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
