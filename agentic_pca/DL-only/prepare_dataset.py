#!/usr/bin/env python3
"""Audit and cache paired PET volumes, then freeze leakage-safe five-fold splits.

The original FDG and PSMA NIfTI files are acquired on different grids. Each
tracer is therefore canonicalized, cropped by its corresponding CT-derived body
mask, robustly normalized and resized independently before the two arrays are
stacked. This is body-normalized early fusion, not voxel-level FDG-to-PSMA
registration. CT and masks define only the crop and are never model inputs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from common import (
    CLASS_NAMES,
    DEFAULT_CACHE_DIR,
    DEFAULT_CASE_ID_SALT,
    DEFAULT_DATASET,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_SEED,
    LABEL_MAPPING_VERSION,
    PREPROCESSING_VERSION,
    SCHEMA_VERSION,
    atomic_write_json,
    class_counts,
    dependency_versions,
    file_stat_fingerprint,
    load_json,
    sha256_file,
    sha256_json,
    stable_case_id,
    treatment_to_category,
    utc_now,
    validate_manifest,
)


PET_FILENAMES = {
    "fdg": "fdgPT_series1.nii.gz",
    "psma": "psmaPT_series1.nii.gz",
}
BODY_MASK_FILENAMES = {
    "fdg": "fdgCT_body_mask.nii.gz",
    "psma": "psmaCT_body_mask.nii.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a uniform 2-channel PET cache and fixed five-fold split."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--target-shape",
        type=int,
        nargs=3,
        default=(128, 128, 384),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--lower-percentile", type=float, default=1.0)
    parser.add_argument("--upper-percentile", type=float, default=99.0)
    parser.add_argument("--crop-margin-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--case-id-salt", default=DEFAULT_CASE_ID_SALT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected-patients", type=int, default=249)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate labels, paths and NIfTI headers without creating the cache.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace cache entries even when a matching sidecar already exists.",
    )
    parser.add_argument(
        "--skip-source-content-hash",
        action="store_true",
        help=(
            "Skip SHA256 of raw PET/body-mask files. This is faster but weakens "
            "provenance; the default is to hash source contents."
        ),
    )
    return parser.parse_args()


def resolve_patient_directories(
    patient_keys: Sequence[str],
    image_root: Path,
) -> dict[str, Path]:
    batch_dirs = sorted(path for path in image_root.iterdir() if path.is_dir())
    by_name: dict[str, list[Path]] = {}
    for batch_dir in batch_dirs:
        for patient_dir in batch_dir.iterdir():
            if patient_dir.is_dir():
                by_name.setdefault(patient_dir.name, []).append(patient_dir.resolve())

    resolved: dict[str, Path] = {}
    errors: list[str] = []
    for patient_key in patient_keys:
        matches = by_name.get(patient_key, [])
        if len(matches) != 1:
            errors.append(
                f"{patient_key!r}: expected one exact directory, found "
                f"{len(matches)} ({matches})"
            )
            continue
        missing = [
            filename
            for filename in (
                *PET_FILENAMES.values(),
                *BODY_MASK_FILENAMES.values(),
            )
            if not (matches[0] / filename).is_file()
        ]
        if missing:
            errors.append(f"{matches[0]}: missing {missing}")
            continue
        resolved[patient_key] = matches[0]
    if errors:
        raise ValueError("Image mapping audit failed:\n" + "\n".join(errors))
    return resolved


def nifti_header(path: Path) -> dict[str, Any]:
    image = nib.load(str(path))
    # Do not call nib.as_closest_canonical here: for .nii.gz files it
    # materializes the complete volume. Header audit must remain metadata-only.
    orientation = nib.orientations.io_orientation(image.affine)
    canonical_shape = [
        int(image.shape[index])
        for index in np.argsort(orientation[:, 0].astype(int))
    ]
    return {
        "shape": list(image.shape),
        "canonical_shape": canonical_shape,
        "dtype": str(image.get_data_dtype()),
        "zooms": [float(value) for value in image.header.get_zooms()[:3]],
        "orientation": "".join(nib.aff2axcodes(image.affine)),
        "canonical_orientation": "RAS",
        "affine": np.asarray(image.affine, dtype=float).round(8).tolist(),
        "file": file_stat_fingerprint(path),
    }


def audit_records(
    dataset: Mapping[str, Any],
    directories: Mapping[str, Path],
    case_id_salt: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    same_shape = 0
    same_affine = 0
    fdg_shapes: Counter[str] = Counter()
    psma_shapes: Counter[str] = Counter()

    for patient_key in sorted(dataset):
        source = dataset[patient_key]
        if not isinstance(source, Mapping):
            raise TypeError(f"{patient_key}: dataset record is not an object")
        label = treatment_to_category(source.get("Treatment"))
        patient_dir = directories[patient_key]
        fdg_path = patient_dir / PET_FILENAMES["fdg"]
        psma_path = patient_dir / PET_FILENAMES["psma"]
        fdg_body_mask_path = patient_dir / BODY_MASK_FILENAMES["fdg"]
        psma_body_mask_path = patient_dir / BODY_MASK_FILENAMES["psma"]
        fdg_header = nifti_header(fdg_path)
        psma_header = nifti_header(psma_path)
        fdg_shapes[str(tuple(fdg_header["shape"]))] += 1
        psma_shapes[str(tuple(psma_header["shape"]))] += 1
        if fdg_header["shape"] == psma_header["shape"]:
            same_shape += 1
        if np.allclose(
            np.asarray(fdg_header["affine"]),
            np.asarray(psma_header["affine"]),
            atol=1e-5,
            rtol=0.0,
        ):
            same_affine += 1

        records.append(
            {
                "case_id": stable_case_id(patient_key, case_id_salt),
                "patient_key": patient_key,
                "label": label,
                "label_index": CLASS_NAMES.index(label),
                "batch": patient_dir.parent.name,
                "patient_dir": str(patient_dir),
                "fdg_path": str(fdg_path),
                "psma_path": str(psma_path),
                "fdg_body_mask_path": str(fdg_body_mask_path),
                "psma_body_mask_path": str(psma_body_mask_path),
                "source_headers": {
                    "fdg": fdg_header,
                    "psma": psma_header,
                    "fdg_body_mask_file": file_stat_fingerprint(
                        fdg_body_mask_path
                    ),
                    "psma_body_mask_file": file_stat_fingerprint(
                        psma_body_mask_path
                    ),
                    "same_shape": fdg_header["shape"] == psma_header["shape"],
                    "same_affine_atol_1e-5": bool(
                        np.allclose(
                            np.asarray(fdg_header["affine"]),
                            np.asarray(psma_header["affine"]),
                            atol=1e-5,
                            rtol=0.0,
                        )
                    ),
                },
            }
        )

    audit = {
        "patients": len(records),
        "class_counts": class_counts(records),
        "fdg_shape_counts": dict(sorted(fdg_shapes.items())),
        "psma_shape_counts": dict(sorted(psma_shapes.items())),
        "same_array_shape": same_shape,
        "same_affine_atol_1e-5": same_affine,
        "warning": (
            "FDG and PSMA are independently body-normalized. Equal output array "
            "shape does not imply voxel-level cross-tracer registration."
        ),
    }
    return records, audit


def _body_bbox_on_pet_grid(
    canonical_pet: nib.spatialimages.SpatialImage,
    body_mask_path: Path,
    margin_fraction: float,
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Map a CT-derived body-mask bounding box onto the tracer's PET grid."""
    body_mask = nib.load(str(body_mask_path))
    if len(body_mask.shape) != 3:
        raise ValueError(f"{body_mask_path}: expected a 3-D body mask")
    mask_array = np.asanyarray(body_mask.dataobj)
    starts: list[int] = []
    stops: list[int] = []
    for axis in range(3):
        reduction_axes = tuple(value for value in range(3) if value != axis)
        occupied = np.any(mask_array, axis=reduction_axes)
        indices = np.flatnonzero(occupied)
        if indices.size == 0:
            raise ValueError(f"{body_mask_path}: body mask is empty")
        starts.append(int(indices[0]))
        stops.append(int(indices[-1]) + 1)
    del mask_array

    # Transform the eight body-box edges from CT-mask voxel coordinates through
    # physical space into the canonical PET voxel coordinates.
    edge_options = [
        (starts[axis] - 0.5, stops[axis] - 0.5) for axis in range(3)
    ]
    corners = np.asarray(
        [
            [coordinate[0], coordinate[1], coordinate[2], 1.0]
            for coordinate in product(*edge_options)
        ],
        dtype=np.float64,
    )
    world = (body_mask.affine @ corners.T).T
    pet_voxels = (np.linalg.inv(canonical_pet.affine) @ world.T).T[:, :3]
    pet_starts = np.floor(pet_voxels.min(axis=0)).astype(int)
    pet_stops = np.ceil(pet_voxels.max(axis=0)).astype(int) + 1
    for axis in range(3):
        margin = int(
            round((pet_stops[axis] - pet_starts[axis]) * margin_fraction)
        )
        pet_starts[axis] = max(0, int(pet_starts[axis]) - margin)
        pet_stops[axis] = min(
            int(canonical_pet.shape[axis]),
            int(pet_stops[axis]) + margin,
        )
        if pet_stops[axis] <= pet_starts[axis]:
            raise ValueError(
                f"{body_mask_path}: mapped body box is empty on PET axis {axis}"
            )
    metadata = {
        "body_mask_path": str(body_mask_path),
        "body_mask_shape": list(body_mask.shape),
        "body_mask_bbox_start_stop": [starts, stops],
    }
    return pet_starts.tolist(), pet_stops.tolist(), metadata


def _normalize_and_resize(
    path: Path,
    body_mask_path: Path,
    target_shape: Sequence[int],
    lower_percentile: float,
    upper_percentile: float,
    margin_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = nib.as_closest_canonical(nib.load(str(path)))
    array = np.asarray(source.dataobj, dtype=np.float32)
    original_shape = list(array.shape)
    if array.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D PET array, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: PET contains NaN or infinity")
    negative_voxels = int(np.count_nonzero(array < 0))
    array = np.maximum(array, 0.0)
    starts, stops, body_metadata = _body_bbox_on_pet_grid(
        source,
        body_mask_path,
        margin_fraction,
    )
    cropped = array[
        starts[0] : stops[0],
        starts[1] : stops[1],
        starts[2] : stops[2],
    ]
    positive = cropped[cropped > 0]
    if positive.size == 0:
        raise ValueError(f"{path}: cropped PET contains no positive voxels")
    lower, upper = np.percentile(
        positive,
        [lower_percentile, upper_percentile],
    ).astype(float)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError(
            f"{path}: invalid robust intensity interval [{lower}, {upper}]"
        )
    normalized = np.clip((cropped - lower) / (upper - lower), 0.0, 1.0)
    tensor = torch.from_numpy(np.ascontiguousarray(normalized))[None, None]
    with torch.inference_mode():
        resized = F.interpolate(
            tensor,
            size=tuple(int(value) for value in target_shape),
            mode="trilinear",
            align_corners=False,
        )[0, 0]
    output = resized.numpy().astype(np.float16, copy=False)
    metadata = {
        "source_shape_canonical": original_shape,
        "pet_crop_bbox_start_stop": [starts, stops],
        "cropped_shape": list(cropped.shape),
        "positive_voxels_used_for_percentiles": int(positive.size),
        "negative_voxels_clipped": negative_voxels,
        "intensity_lower": lower,
        "intensity_upper": upper,
        "output_min": float(output.min()),
        "output_max": float(output.max()),
        **body_metadata,
    }
    return output, metadata


def _cache_one(payload: Mapping[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    record = dict(payload["record"])
    cache_path = Path(record["cache_path"])
    sidecar_path = cache_path.with_suffix(".json")
    fingerprint = str(payload["preprocess_fingerprint"])
    expected_shape = (2, *tuple(payload["target_shape"]))
    source_files = {
        key: file_stat_fingerprint(Path(record[key]))
        for key in (
            "fdg_path",
            "psma_path",
            "fdg_body_mask_path",
            "psma_body_mask_path",
        )
    }
    if payload.get("hash_source_contents", True):
        for key, metadata in source_files.items():
            metadata["sha256"] = sha256_file(Path(record[key]))
    source_fingerprint = sha256_json(source_files)

    if cache_path.exists() and sidecar_path.exists() and not payload["overwrite"]:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if (
            sidecar.get("preprocess_fingerprint") == fingerprint
            and sidecar.get("source_fingerprint") == source_fingerprint
        ):
            array = np.load(cache_path, mmap_mode="r", allow_pickle=False)
            if array.shape != expected_shape or array.dtype != np.float16:
                raise ValueError(
                    f"{cache_path}: invalid cache shape/dtype "
                    f"{array.shape}/{array.dtype}"
                )
            cache_sha256 = sha256_file(cache_path)
            if sidecar.get("cache_sha256") == cache_sha256:
                record["preprocessing"] = sidecar["modalities"]
                record["cache_source_fingerprint"] = source_fingerprint
                record["source_files"] = source_files
                record["cache_sha256"] = cache_sha256
                return record

    modalities: dict[str, Any] = {}
    arrays: list[np.ndarray] = []
    for tracer in ("fdg", "psma"):
        array, metadata = _normalize_and_resize(
            Path(record[f"{tracer}_path"]),
            Path(record[f"{tracer}_body_mask_path"]),
            payload["target_shape"],
            float(payload["lower_percentile"]),
            float(payload["upper_percentile"]),
            float(payload["margin_fraction"]),
        )
        arrays.append(array)
        modalities[tracer] = metadata
    stacked = np.stack(arrays, axis=0)
    if stacked.shape != expected_shape or not np.isfinite(stacked).all():
        raise ValueError(f"{record['case_id']}: invalid output tensor")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, stacked, allow_pickle=False)
    os.replace(temporary, cache_path)
    cache_sha256 = sha256_file(cache_path)
    atomic_write_json(
        sidecar_path,
        {
            "schema_version": SCHEMA_VERSION,
            "case_id": record["case_id"],
            "preprocess_fingerprint": fingerprint,
            "source_files": source_files,
            "source_fingerprint": source_fingerprint,
            "cache_sha256": cache_sha256,
            "shape": list(stacked.shape),
            "dtype": str(stacked.dtype),
            "modalities": modalities,
        },
    )
    record["preprocessing"] = modalities
    record["cache_source_fingerprint"] = source_fingerprint
    record["source_files"] = source_files
    record["cache_sha256"] = cache_sha256
    return record


def build_folds(
    records: Sequence[Mapping[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    indices = np.arange(len(records))
    labels = np.asarray([int(record["label_index"]) for record in records])
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds: list[dict[str, Any]] = []
    for fold_number, (outer_train, test) in enumerate(outer.split(indices, labels)):
        inner = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=seed + fold_number,
        )
        relative_train, relative_val = next(
            inner.split(outer_train, labels[outer_train])
        )
        train = outer_train[relative_train]
        val = outer_train[relative_val]

        def case_ids(values: np.ndarray) -> list[str]:
            return [str(records[int(index)]["case_id"]) for index in values]

        def counts(values: np.ndarray) -> dict[str, int]:
            return class_counts([records[int(index)] for index in values])

        folds.append(
            {
                "fold": fold_number,
                "split_seed": seed + fold_number,
                "train_case_ids": case_ids(train),
                "val_case_ids": case_ids(val),
                "test_case_ids": case_ids(test),
                "class_counts": {
                    "train": counts(train),
                    "val": counts(val),
                    "test": counts(test),
                },
            }
        )
    return folds


def main() -> int:
    args = parse_args()
    if not 0 <= args.lower_percentile < args.upper_percentile <= 100:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100.")
    if not 0 <= args.crop_margin_fraction <= 0.25:
        raise ValueError("--crop-margin-fraction must be between 0 and 0.25.")
    if any(value <= 0 for value in args.target_shape):
        raise ValueError("--target-shape values must be positive.")

    dataset = load_json(args.dataset)
    if not isinstance(dataset, dict):
        raise TypeError("Dataset must be an object keyed by patient name.")
    if args.expected_patients and len(dataset) != args.expected_patients:
        raise ValueError(
            f"Expected {args.expected_patients} patients, found {len(dataset)}."
        )
    directories = resolve_patient_directories(sorted(dataset), args.image_root)
    records, audit = audit_records(dataset, directories, args.case_id_salt)
    if set(record["label"] for record in records) != set(CLASS_NAMES):
        raise ValueError("The audited cohort does not contain all four classes.")

    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if args.audit_only:
        return 0

    preprocessing_config = {
        "version": PREPROCESSING_VERSION,
        "target_shape_xyz": list(args.target_shape),
        "canonical_orientation": "RAS",
        "geometry": (
            "each tracer independently cropped with its CT-derived body-mask "
            "bounding box and resized; not cross-tracer registered"
        ),
        "crop_source": (
            "fdgCT_body_mask.nii.gz and psmaCT_body_mask.nii.gz are used only "
            "to define tracer-specific bounding boxes; CT/masks are not model inputs"
        ),
        "crop_margin_fraction": args.crop_margin_fraction,
        "intensity_scaling": {
            "population": "positive voxels within each independently cropped tracer",
            "lower_percentile": args.lower_percentile,
            "upper_percentile": args.upper_percentile,
            "output_range": [0.0, 1.0],
        },
        "interpolation": "trilinear_align_corners_false",
        "cache_dtype": "float16",
        "source_content_sha256": not args.skip_source_content_hash,
        "channel_order": ["FDG", "PSMA"],
        "implementation_sha256": {
            "prepare_dataset.py": sha256_file(Path(__file__)),
            "common.py": sha256_file(Path(__file__).with_name("common.py")),
        },
    }
    preprocess_fingerprint = sha256_json(preprocessing_config)
    volumes_dir = args.cache_dir.resolve() / "volumes"
    for record in records:
        record["cache_path"] = str(volumes_dir / f"{record['case_id']}.npy")

    payloads = [
        {
            "record": record,
            "target_shape": list(args.target_shape),
            "lower_percentile": args.lower_percentile,
            "upper_percentile": args.upper_percentile,
            "margin_fraction": args.crop_margin_fraction,
            "preprocess_fingerprint": preprocess_fingerprint,
            "overwrite": args.overwrite,
            "hash_source_contents": not args.skip_source_content_hash,
        }
        for record in records
    ]
    processed: list[dict[str, Any]] = []
    if args.workers <= 1:
        for index, payload in enumerate(payloads, start=1):
            processed.append(_cache_one(payload))
            print(f"[{index}/{len(payloads)}] cached {payload['record']['case_id']}")
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = {
                executor.submit(_cache_one, payload): payload["record"]["case_id"]
                for payload in payloads
            }
            for index, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                case_id = futures[future]
                processed.append(future.result())
                print(f"[{index}/{len(futures)}] cached {case_id}", flush=True)
    processed.sort(key=lambda record: str(record["patient_key"]))
    duplicate_cache_hashes = {
        digest: sorted(
            str(record["case_id"])
            for record in processed
            if record["cache_sha256"] == digest
        )
        for digest in {record["cache_sha256"] for record in processed}
        if sum(record["cache_sha256"] == digest for record in processed) > 1
    }
    if duplicate_cache_hashes:
        raise ValueError(
            "Exact duplicate preprocessed PET pairs were found across case IDs; "
            f"these must be resolved before patient-level CV: {duplicate_cache_hashes}"
        )

    folds = build_folds(processed, args.seed)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "dataset": {
            "path": str(args.dataset.resolve()),
            "sha256": sha256_file(args.dataset),
            "patients": len(dataset),
        },
        "image_root": str(args.image_root.resolve()),
        "case_id_salt": args.case_id_salt,
        "label_mapping_version": LABEL_MAPPING_VERSION,
        "classes": list(CLASS_NAMES),
        "seed": args.seed,
        "dependencies": dependency_versions(),
        "audit": audit,
        "preprocessing": {
            **preprocessing_config,
            "fingerprint": preprocess_fingerprint,
        },
        "patients": processed,
        "folds": folds,
    }
    validate_manifest(manifest)
    manifest["manifest_fingerprint"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "created_at_utc"}
    )
    atomic_write_json(args.cache_dir.resolve() / "manifest.json", manifest)
    atomic_write_json(
        args.cache_dir.resolve() / "patient_manifest.json",
        {
            record["case_id"]: record["patient_key"]
            for record in processed
        },
    )
    print(
        f"Prepared {len(processed)} patients at {args.cache_dir.resolve()}; "
        f"preprocess_fingerprint={preprocess_fingerprint}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
