#!/usr/bin/env python3
"""Evaluate heterogeneous comparison runs against the main trajectory-RAG run.

The shared trajectory evaluator expects every experiment to be an immediate
child of one output root.  Comparison runs are intentionally stored in several
locations, so this wrapper creates a temporary directory containing named
symlinks to the requested runs and delegates all metric/statistical work to
``trajectory_rag_ablation.evaluate_ablations``.  Source runs are never copied
or modified.

``structured_suv_ml`` is deliberately excluded because its prediction artifact
format is not the per-trajectory JSON format consumed by the shared evaluator.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_pca.retrieval_agent_inference.trajectory_rag_ablation import (  # noqa: E402
    evaluate_ablations as shared_evaluator,
)


DEFAULT_BASELINE_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "agentic_pca/retrieval_agent_inference/outputs"
    / "comparison_baselines_Qwen3.5-9B"
)
DEFAULT_INFER_DIR = (
    PROJECT_ROOT
    / "agentic_pca/retrieval_agent_inference/outputs"
    / "full_run_Qwen3.5_9B"
)
DEFAULT_REFERENCE_DIR = (
    PROJECT_ROOT
    / "agentic_pca/retrieval_agent_inference/outputs"
    / "trajectory_rag_full_run_Qwen3.59BonQwen3.59B"
)
DEFAULT_SEED = 20_260_726
DEFAULT_BOOTSTRAP_SAMPLES = 10_000

DIRECT_BASELINE_IDS = (
    "clinical_only",
    "suv_only",
    "clinical_suv_no_retrieval",
)
STAGED_INFER_NAME = "pdf_rag_agent"
STAGED_CASE_KNN_NAME = "case_knn"
EXPERIMENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the three direct Qwen baselines, case k-NN, and the "
            "existing PDF-RAG infer run against the main trajectory-RAG run."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "structured_suv_ml is intentionally not included: it has a "
            "different, non-trajectory output format and must be reported by "
            "its own evaluator."
        ),
    )
    parser.add_argument(
        "--baseline-output-root",
        "--output-root",
        dest="baseline_output_root",
        type=Path,
        default=DEFAULT_BASELINE_OUTPUT_ROOT,
        help=(
            "root containing clinical_only/, suv_only/, "
            "clinical_suv_no_retrieval/, and case_knn/"
        ),
    )
    parser.add_argument(
        "--clinical-only-dir",
        type=Path,
        help=(
            "explicit clinical_only run directory; defaults to "
            "BASELINE_OUTPUT_ROOT/clinical_only"
        ),
    )
    parser.add_argument(
        "--suv-only-dir",
        type=Path,
        help=(
            "explicit suv_only run directory; defaults to "
            "BASELINE_OUTPUT_ROOT/suv_only"
        ),
    )
    parser.add_argument(
        "--clinical-suv-no-retrieval-dir",
        "--clinical-suv-dir",
        dest="clinical_suv_no_retrieval_dir",
        type=Path,
        help=(
            "explicit clinical_suv_no_retrieval run directory; defaults to "
            "BASELINE_OUTPUT_ROOT/clinical_suv_no_retrieval"
        ),
    )
    parser.add_argument(
        "--case-knn-dir",
        type=Path,
        help=(
            "explicit case_knn run directory; defaults to "
            "BASELINE_OUTPUT_ROOT/case_knn"
        ),
    )
    parser.add_argument(
        "--infer-dir",
        type=Path,
        default=DEFAULT_INFER_DIR,
        help="existing Qwen3.5-9B PDF-RAG infer run directory",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help="main trajectory-RAG run used as the statistical reference",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help=(
            "complete JSON report destination; defaults to "
            "BASELINE_OUTPUT_ROOT/evaluation/baselines_vs_trajectory_rag.json"
        ),
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        help=(
            "flat CSV report destination; defaults to "
            "BASELINE_OUTPUT_ROOT/evaluation/baselines_vs_trajectory_rag.csv"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="fixed seed used independently for each paired cluster bootstrap",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="paired patient-cluster bootstrap replicates",
    )
    return parser


def _as_run_dir(path: Path, *, description: str) -> Path:
    """Resolve and validate one trajectory-format experiment directory."""
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(
            f"{description} does not exist or is not a directory: {resolved}"
        )

    # Match the convenience accepted by the shared evaluator while staging a
    # uniform run directory that always contains a patients/ child.
    run_dir = resolved.parent if resolved.name == "patients" else resolved
    patients_dir = run_dir / "patients"
    if not patients_dir.is_dir():
        raise ValueError(f"{description} lacks a patients/ directory: {run_dir}")
    if not any(
        child.is_dir() and child.name.startswith("case_")
        for child in patients_dir.iterdir()
    ):
        raise ValueError(
            f"{description} has no case_* patient directories: {patients_dir}"
        )
    return run_dir


def _resolved_experiments(args: argparse.Namespace) -> dict[str, Path]:
    root = args.baseline_output_root.expanduser()
    requested = {
        DIRECT_BASELINE_IDS[0]: (
            args.clinical_only_dir or root / DIRECT_BASELINE_IDS[0]
        ),
        DIRECT_BASELINE_IDS[1]: args.suv_only_dir or root / DIRECT_BASELINE_IDS[1],
        DIRECT_BASELINE_IDS[2]: (
            args.clinical_suv_no_retrieval_dir
            or root / DIRECT_BASELINE_IDS[2]
        ),
        STAGED_CASE_KNN_NAME: args.case_knn_dir or root / STAGED_CASE_KNN_NAME,
        STAGED_INFER_NAME: args.infer_dir,
    }
    experiments: dict[str, Path] = {}
    for name, path in requested.items():
        if not EXPERIMENT_NAME_RE.fullmatch(name):
            raise ValueError(f"invalid staged experiment name: {name!r}")
        experiments[name] = _as_run_dir(
            path,
            description=f"{name} experiment",
        )
    return experiments


def _validate_distinct_sources(
    experiments: Mapping[str, Path],
    reference_dir: Path,
) -> None:
    by_source: dict[Path, str] = {}
    for name, source in experiments.items():
        previous = by_source.get(source)
        if previous is not None:
            raise ValueError(
                f"{name} and {previous} resolve to the same run directory: {source}"
            )
        by_source[source] = name
    if reference_dir in by_source:
        raise ValueError(
            "reference run is also configured as an experiment "
            f"({by_source[reference_dir]}): {reference_dir}"
        )


def _stage_experiments(experiments: Mapping[str, Path], staging_root: Path) -> None:
    """Create read-only-by-convention directory links for shared discovery."""
    for name, source in sorted(experiments.items()):
        destination = staging_root / name
        destination.symlink_to(source, target_is_directory=True)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    baseline_root = args.baseline_output_root.expanduser()
    json_out = (
        args.json_out
        or baseline_root / "evaluation" / "baselines_vs_trajectory_rag.json"
    )
    csv_out = (
        args.csv_out
        or baseline_root / "evaluation" / "baselines_vs_trajectory_rag.csv"
    )
    experiments = _resolved_experiments(args)
    reference_dir = _as_run_dir(
        args.reference_dir,
        description="trajectory-RAG reference",
    )
    _validate_distinct_sources(experiments, reference_dir)

    with tempfile.TemporaryDirectory(prefix="psma-baseline-evaluation-") as temporary:
        staging_root = Path(temporary)
        _stage_experiments(experiments, staging_root)
        return shared_evaluator.evaluate(
            output_root=staging_root,
            reference_dir=reference_dir,
            json_out=json_out,
            csv_out=csv_out,
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = evaluate(args)
    except (shared_evaluator.EvaluationError, OSError, ValueError) as exc:
        parser.error(str(exc))

    names = ", ".join(sorted(report["experiments"]))
    default_report_dir = args.baseline_output_root / "evaluation"
    json_out = args.json_out or default_report_dir / "baselines_vs_trajectory_rag.json"
    csv_out = args.csv_out or default_report_dir / "baselines_vs_trajectory_rag.csv"
    print(
        f"Evaluated {len(report['experiments'])} comparison baselines "
        f"({names}). JSON: {json_out}; CSV: {csv_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
