import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import ants
except ImportError as exc:
    raise ImportError(
        "registerANTs_warped_pet_to_nii.py requires ANTsPy/antspyx. "
        "Install it in the registration environment, for example: "
        "pip install antspyx"
    ) from exc


DEFAULT_INPUT_DIRS = ["/data2/xiangcen/data/pet_gen/processed/batch3_h5_v2/"]
DEFAULT_OUTPUT_DIR = "/data2/xiangcen/data/pet_gen/ants_registered_nii_10"
REQUIRED_DATASETS = (
    "fdg_ct",
    "fdg_pt",
    "fdg_mask",
    "psma_ct",
    "psma_pt",
    "psma_mask",
)
OUTPUT_NAMES = (
    "fdg_pet.nii.gz",
    "fdg_ct.nii.gz",
    "fdg_mask.nii.gz",
    "psma_pet.nii.gz",
    "psma_ct.nii.gz",
    "psma_mask.nii.gz",
    "warped_fdg_pet.nii.gz",
    "warped_fdg_ct.nii.gz",
    "warped_fdg_mask.nii.gz",
)


def volume_to_numpy(dataset, dataset_name, dtype=np.float32):
    array = np.asarray(dataset)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(
            f"Dataset '{dataset_name}' must have shape (X, Y, Z) or "
            f"(1, X, Y, Z), got {array.shape}."
        )
    return array.astype(dtype, copy=False)


def spacing_from_h5(h5_file, attribute_name):
    if attribute_name not in h5_file.attrs:
        raise KeyError(f"Missing H5 attribute: {attribute_name}")

    spacing = tuple(float(value) for value in h5_file.attrs[attribute_name])
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError(f"Invalid {attribute_name}: {spacing}")
    return spacing


def to_ants_image(array, spacing):
    return ants.from_numpy(array.astype(np.float32, copy=False), spacing=spacing)


def mask_to_uint32(array, mask_name):
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Mask '{mask_name}' contains non-finite values.")

    rounded = np.rint(array)
    if np.any(rounded < 0) or np.any(rounded > np.iinfo(np.uint32).max):
        raise ValueError(f"Mask '{mask_name}' contains labels outside uint32 range.")
    return rounded.astype(np.uint32)


def mask_to_ants_image(array, reference, mask_name):
    mask = mask_to_uint32(array, mask_name)
    return ants.from_numpy(
        mask,
        origin=reference.origin,
        spacing=reference.spacing,
        direction=reference.direction,
    )


def validate_shapes(volumes):
    fdg_shape = volumes["fdg_ct"].shape
    psma_shape = volumes["psma_ct"].shape

    for name in ("fdg_pt", "fdg_mask"):
        if volumes[name].shape != fdg_shape:
            raise ValueError(
                f"{name} shape {volumes[name].shape} differs from "
                f"fdg_ct shape {fdg_shape}."
            )
    for name in ("psma_pt", "psma_mask"):
        if volumes[name].shape != psma_shape:
            raise ValueError(
                f"{name} shape {volumes[name].shape} differs from "
                f"psma_ct shape {psma_shape}."
            )


def register_case(source_h5, type_of_transform, image_interpolator):
    for dataset_name in REQUIRED_DATASETS:
        if dataset_name not in source_h5:
            raise KeyError(f"Missing H5 dataset: {dataset_name}")

    fdg_spacing = spacing_from_h5(source_h5, "fdg_spacing")
    psma_spacing = spacing_from_h5(source_h5, "psma_spacing")
    volumes = {
        name: volume_to_numpy(source_h5[name], name) for name in REQUIRED_DATASETS
    }
    validate_shapes(volumes)

    fdg_ct = to_ants_image(volumes["fdg_ct"], fdg_spacing)
    fdg_pet = to_ants_image(volumes["fdg_pt"], fdg_spacing)
    fdg_mask = mask_to_ants_image(volumes["fdg_mask"], fdg_ct, "fdg_mask")
    psma_ct = to_ants_image(volumes["psma_ct"], psma_spacing)
    psma_pet = to_ants_image(volumes["psma_pt"], psma_spacing)
    psma_mask = mask_to_ants_image(volumes["psma_mask"], psma_ct, "psma_mask")

    with tempfile.TemporaryDirectory(prefix="ants_fdg_to_psma_nii_") as tmpdir:
        registration = ants.registration(
            fixed=psma_ct,
            moving=fdg_ct,
            type_of_transform=type_of_transform,
            outprefix=os.path.join(tmpdir, "ants_"),
        )
        transforms = registration["fwdtransforms"]
        warped_fdg_pet = ants.apply_transforms(
            fixed=psma_ct,
            moving=fdg_pet,
            transformlist=transforms,
            interpolator=image_interpolator,
        )
        warped_fdg_ct = ants.apply_transforms(
            fixed=psma_ct,
            moving=fdg_ct,
            transformlist=transforms,
            interpolator=image_interpolator,
        )
        warped_fdg_mask_float = ants.apply_transforms(
            fixed=psma_ct,
            moving=fdg_mask,
            transformlist=transforms,
            interpolator="nearestNeighbor",
        )

    warped_fdg_mask = mask_to_ants_image(
        warped_fdg_mask_float.numpy(), psma_ct, "warped_fdg_mask"
    )
    return {
        "fdg_pet.nii.gz": fdg_pet,
        "fdg_ct.nii.gz": fdg_ct,
        "fdg_mask.nii.gz": fdg_mask,
        "psma_pet.nii.gz": psma_pet,
        "psma_ct.nii.gz": psma_ct,
        "psma_mask.nii.gz": psma_mask,
        "warped_fdg_pet.nii.gz": warped_fdg_pet,
        "warped_fdg_ct.nii.gz": warped_fdg_ct,
        "warped_fdg_mask.nii.gz": warped_fdg_mask,
    }


def read_patient_name(source_h5, source_path):
    if "patient_name" not in source_h5:
        return source_path.stem

    value = source_h5["patient_name"][()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif isinstance(value, np.ndarray):
        value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
    return str(value).strip() or source_path.stem


def safe_patient_name(patient_name, fallback):
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", patient_name).strip("._")
    return safe_name or fallback


def collect_h5_files(input_dirs, max_cases):
    source_paths = []
    for input_dir_value in input_dirs:
        input_dir = Path(input_dir_value).expanduser().resolve()
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        source_paths.extend(input_dir.rglob("*.h5"))

    unique_paths = sorted(set(source_paths), key=lambda path: str(path))
    if max_cases is not None:
        unique_paths = unique_paths[:max_cases]
    return unique_paths


def output_is_complete(patient_dir):
    return patient_dir.is_dir() and all(
        (patient_dir / output_name).is_file() for output_name in OUTPUT_NAMES
    )


def save_case(images, patient_dir):
    patient_dir.mkdir(parents=True, exist_ok=True)
    for output_name in OUTPUT_NAMES:
        ants.image_write(images[output_name], str(patient_dir / output_name))


def process_all(args):
    if args.max_cases is not None and args.max_cases <= 0:
        raise ValueError("--max_cases must be greater than zero.")

    source_paths = collect_h5_files(args.input_dirs, args.max_cases)
    if not source_paths:
        raise RuntimeError("No .h5 files were found in the input directories.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> Selected H5 files: {len(source_paths)}")
    print(">>> Registration optimization: fdg_ct -> psma_ct")
    print(">>> Saving FDG, PSMA, and warped FDG PET/CT/mask as NIfTI")
    print(">>> Mask interpolation: nearestNeighbor")
    print(f">>> Output directory: {output_dir}")

    completed = 0
    skipped = 0
    failures = []
    used_patient_names = set()

    for source_path in tqdm(source_paths, desc="Registering patients"):
        try:
            with h5py.File(source_path, "r") as source_h5:
                patient_name = safe_patient_name(
                    read_patient_name(source_h5, source_path), source_path.stem
                )
                if patient_name in used_patient_names:
                    patient_name = f"{patient_name}_{source_path.stem}"
                used_patient_names.add(patient_name)
                patient_dir = output_dir / patient_name

                if not args.overwrite and output_is_complete(patient_dir):
                    skipped += 1
                    continue

                images = register_case(
                    source_h5=source_h5,
                    type_of_transform=args.type_of_transform,
                    image_interpolator=args.image_interpolator,
                )
                save_case(images, patient_dir)
                completed += 1
        except Exception as exc:
            failures.append((source_path, str(exc)))
            tqdm.write(f"FAILED: {source_path}: {exc}")

    print(f">>> Completed: {completed} | Skipped: {skipped} | Failed: {len(failures)}")
    if failures:
        failure_lines = "\n".join(
            f"  {source_path}: {message}" for source_path, message in failures
        )
        raise RuntimeError(
            f"Registration failed for {len(failures)} patient(s):\n{failure_lines}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Register FDG CT to PSMA CT, apply the transform to FDG PET/CT/mask, "
            "and save the original and warped volumes as per-patient NIfTI files."
        )
    )
    parser.add_argument(
        "--input_dirs",
        type=str,
        nargs="+",
        default=DEFAULT_INPUT_DIRS,
        help="One or more directories searched recursively for H5 files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory containing one output folder per patient.",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=10,
        help="Maximum number of sorted H5 cases to process (default: 10).",
    )
    parser.add_argument(
        "--type_of_transform",
        type=str,
        default="SyN",
        help="ANTs transform type, for example Rigid, Affine, SyN, or SyNOnly.",
    )
    parser.add_argument(
        "--image_interpolator",
        type=str,
        default="linear",
        help="ANTs interpolator for PET and CT. Masks always use nearestNeighbor.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute patients whose nine output files already exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    process_all(parse_args())
