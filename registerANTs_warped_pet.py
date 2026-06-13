import argparse
import os
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
        "registerANTs_warped_pet.py requires ANTsPy/antspyx. Install it in "
        "the environment used for registration, for example: pip install antspyx"
    ) from exc


DEFAULT_INPUT_DIRS = [
    "/data2/xiangcen/data/pet_gen/processed/batch1_h5_v2",
    "/data2/xiangcen/data/pet_gen/processed/batch2_h5_v2",
    "/data2/xiangcen/data/pet_gen/processed/batch3_h5_v2",
]
DEFAULT_OUTPUT_DIR = "/data2/xiangcen/data/pet_gen/processed/warped_fdg_pet_h5"
REQUIRED_DATASETS = ("fdg_ct", "fdg_pt", "psma_ct", "psma_pt")


def volume_to_numpy(dataset, dataset_name):
    array = np.asarray(dataset, dtype=np.float32)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(
            f"Dataset '{dataset_name}' must have shape (X, Y, Z) or "
            f"(1, X, Y, Z), got {array.shape}."
        )
    return array


def spacing_from_h5(h5_file, attribute_name):
    if attribute_name not in h5_file.attrs:
        raise KeyError(f"Missing H5 attribute: {attribute_name}")

    spacing = tuple(float(value) for value in h5_file.attrs[attribute_name])
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError(f"Invalid {attribute_name}: {spacing}")
    return spacing


def to_ants_image(array, spacing):
    return ants.from_numpy(array.astype(np.float32, copy=False), spacing=spacing)


def register_fdg_pet(source_h5, type_of_transform, interpolator):
    for dataset_name in REQUIRED_DATASETS:
        if dataset_name not in source_h5:
            raise KeyError(f"Missing H5 dataset: {dataset_name}")

    fdg_spacing = spacing_from_h5(source_h5, "fdg_spacing")
    psma_spacing = spacing_from_h5(source_h5, "psma_spacing")

    fdg_ct = volume_to_numpy(source_h5["fdg_ct"], "fdg_ct")
    fdg_pet = volume_to_numpy(source_h5["fdg_pt"], "fdg_pt")
    psma_ct = volume_to_numpy(source_h5["psma_ct"], "psma_ct")
    psma_pet = volume_to_numpy(source_h5["psma_pt"], "psma_pt")

    if fdg_ct.shape != fdg_pet.shape:
        raise ValueError(
            f"fdg_ct and fdg_pt shapes differ: {fdg_ct.shape} vs {fdg_pet.shape}"
        )
    if psma_ct.shape != psma_pet.shape:
        raise ValueError(
            f"psma_ct and psma_pt shapes differ: {psma_ct.shape} vs {psma_pet.shape}"
        )

    moving_ct = to_ants_image(fdg_ct, fdg_spacing)
    moving_pet = to_ants_image(fdg_pet, fdg_spacing)
    fixed_ct = to_ants_image(psma_ct, psma_spacing)

    with tempfile.TemporaryDirectory(prefix="ants_fdg_pet_to_psma_") as tmpdir:
        registration = ants.registration(
            fixed=fixed_ct,
            moving=moving_ct,
            type_of_transform=type_of_transform,
            outprefix=os.path.join(tmpdir, "ants_"),
        )
        warped_pet = ants.apply_transforms(
            fixed=fixed_ct,
            moving=moving_pet,
            transformlist=registration["fwdtransforms"],
            interpolator=interpolator,
        )

    # Keep the same channel-first layout as fdg_pt and psma_pt in the source H5.
    return warped_pet.numpy().astype(np.float32, copy=False)[None, ...], psma_spacing


def copy_h5_with_warped_pet(
    source_path,
    output_path,
    dataset_name,
    type_of_transform,
    interpolator,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None
    try:
        with h5py.File(source_path, "r") as source_h5:
            warped_pet, warped_spacing = register_fdg_pet(
                source_h5=source_h5,
                type_of_transform=type_of_transform,
                interpolator=interpolator,
            )

            with tempfile.NamedTemporaryFile(
                prefix=f".{output_path.stem}_",
                suffix=".h5.tmp",
                dir=output_path.parent,
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)

            with h5py.File(temp_path, "w") as output_h5:
                for attribute_name, value in source_h5.attrs.items():
                    output_h5.attrs[attribute_name] = value
                for key in source_h5.keys():
                    source_h5.copy(key, output_h5)

                if dataset_name in output_h5:
                    del output_h5[dataset_name]
                warped_dataset = output_h5.create_dataset(
                    dataset_name,
                    data=warped_pet,
                    compression="gzip",
                )
                warped_dataset.attrs["spacing"] = warped_spacing
                warped_dataset.attrs["moving_image"] = "fdg_pt"
                warped_dataset.attrs["fixed_space"] = "psma_pt"
                warped_dataset.attrs["optimized_from"] = "fdg_ct_to_psma_ct"
                warped_dataset.attrs["type_of_transform"] = type_of_transform
                warped_dataset.attrs["interpolator"] = interpolator

        os.replace(temp_path, output_path)
        temp_path = None
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def collect_h5_files(input_dirs, output_dir):
    output_root = output_dir.resolve()
    cases = []
    used_labels = set()

    for input_index, input_dir_value in enumerate(input_dirs, start=1):
        input_dir = Path(input_dir_value).expanduser().resolve()
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        if input_dir == output_root or output_root.is_relative_to(input_dir):
            raise ValueError(
                f"Output directory must not be inside an input directory: {input_dir}"
            )

        label = input_dir.name or f"input_{input_index}"
        if label in used_labels:
            label = f"{label}_{input_index}"
        used_labels.add(label)

        for source_path in sorted(input_dir.rglob("*.h5")):
            relative_path = source_path.relative_to(input_dir)
            output_path = output_root / label / relative_path
            cases.append((source_path, output_path))

    return cases


def output_is_complete(output_path, dataset_name):
    if not output_path.is_file():
        return False
    try:
        with h5py.File(output_path, "r") as output_h5:
            return dataset_name in output_h5
    except OSError:
        return False


def process_all(args):
    output_dir = Path(args.output_dir).expanduser()
    cases = collect_h5_files(args.input_dirs, output_dir)
    if not cases:
        raise RuntimeError("No .h5 files were found in the input directories.")

    print(f">>> Found {len(cases)} H5 files")
    print(">>> Registration optimization: fdg_ct -> psma_ct")
    print(f">>> Warped image: fdg_pt -> psma_pt space ({args.dataset_name})")
    print(f">>> Output directory: {output_dir.resolve()}")

    completed = 0
    skipped = 0
    failures = []
    for source_path, output_path in tqdm(cases, desc="Registering H5 files"):
        if not args.overwrite and output_is_complete(output_path, args.dataset_name):
            skipped += 1
            continue

        try:
            copy_h5_with_warped_pet(
                source_path=source_path,
                output_path=output_path,
                dataset_name=args.dataset_name,
                type_of_transform=args.type_of_transform,
                interpolator=args.interpolator,
            )
            completed += 1
        except Exception as exc:
            failures.append((source_path, str(exc)))
            tqdm.write(f"FAILED: {source_path}: {exc}")

    print(f">>> Completed: {completed} | Skipped: {skipped} | Failed: {len(failures)}")
    if failures:
        failure_lines = "\n".join(f"  {path}: {message}" for path, message in failures)
        raise RuntimeError(f"Registration failed for {len(failures)} file(s):\n{failure_lines}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Register every fdg_pt to psma_pt space using transforms optimized "
            "from fdg_ct and psma_ct, then write new H5 files."
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
        help="New root directory for output H5 files.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="warped_fdg_pet",
        help="Dataset name used for the registered FDG PET image.",
    )
    parser.add_argument(
        "--type_of_transform",
        type=str,
        default="SyN",
        help="ANTs transform type, for example Rigid, Affine, SyN, or SyNOnly.",
    )
    parser.add_argument(
        "--interpolator",
        type=str,
        default="linear",
        help="ANTs interpolator used for the continuous PET image.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output files that already contain the requested dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    process_all(parse_args())
