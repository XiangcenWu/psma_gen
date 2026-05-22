import argparse
import os
import sys
import tempfile

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import ants
except ImportError as exc:
    raise ImportError(
        "registerANTs.py requires ANTsPy/antspyx. Install it in the environment "
        "used for inference, for example: pip install antspyx"
    ) from exc

from General.data_loader import ReadH5d, create_data_loader
from General.dataset_sample import split_multiple_train_test
from General.segments import SEGMENT_INDEX
from Registration.inferencing import (
    compute_tre_single,
    dice_metric,
    get_binary_mask_with_label,
    save_registration_results,
)


DEFAULT_DATA_DIRS = [
    "/data2/xiangcen/data/pet_gen/processed/batch1_h5_v2",
    "/data2/xiangcen/data/pet_gen/processed/batch2_h5_v2",
]
DEFAULT_TEST_COUNTS = [40, 40]
DEFAULT_RESULT_DIR = "/share/home/xcwu/pet_reg_results"
DEFAULT_OUTPUT_NAME = "ants_fdgct_to_psmact.txt"


def spacing_to_list(spacing):
    return [item.item() if hasattr(item, "item") else float(item) for item in spacing]


def tensor_to_ants_image(tensor, spacing):
    """
    Convert a batched tensor from the H5 loader to an ANTs image.

    The project stores volumes as (B, C, X, Y, Z), matching the spacing order
    saved in the H5 attributes.
    """
    array = tensor.detach().cpu()[0, 0].numpy().astype(np.float32)
    return ants.from_numpy(array, spacing=spacing)


def ants_image_to_tensor(image):
    array = image.numpy().astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def build_test_loader(args):
    if len(args.data_dirs) != len(args.test_counts):
        raise ValueError("--data_dirs and --test_counts must have the same length.")

    transform = ReadH5d()
    _, test_list = split_multiple_train_test(args.data_dirs, args.test_counts)

    return create_data_loader(
        test_list,
        transform,
        batch_size=1,
        shuffle=False,
    )


def make_output_path(args):
    os.makedirs(args.result_dir, exist_ok=True)
    return os.path.join(args.result_dir, args.output_name)


def register_fdg_ct_to_psma_ct(
    fdg_ct,
    psma_ct,
    fdg_mask,
    fdg_spacing,
    psma_spacing,
    type_of_transform,
    interpolator,
):
    moving_ct = tensor_to_ants_image(fdg_ct, fdg_spacing)
    fixed_ct = tensor_to_ants_image(psma_ct, psma_spacing)
    moving_mask = tensor_to_ants_image(fdg_mask, fdg_spacing)

    with tempfile.TemporaryDirectory(prefix="ants_fdg_to_psma_") as tmpdir:
        registration = ants.registration(
            fixed=fixed_ct,
            moving=moving_ct,
            type_of_transform=type_of_transform,
            outprefix=os.path.join(tmpdir, "ants_"),
        )

        warped_mask = ants.apply_transforms(
            fixed=fixed_ct,
            moving=moving_mask,
            transformlist=registration["fwdtransforms"],
            interpolator=interpolator,
        )

    return ants_image_to_tensor(warped_mask)


def inference_ants_batch(
    loader,
    filename,
    masks_names=list(SEGMENT_INDEX.keys()),
    type_of_transform="SyN",
    interpolator="nearestNeighbor",
):
    mask_list = []
    for name in masks_names:
        if name in SEGMENT_INDEX:
            mask_list.append(SEGMENT_INDEX[name])
        else:
            raise ValueError(f"Unknown segment names: {name}")

    num_of_masks = len(masks_names)
    dice_before_lists = [[] for _ in range(num_of_masks)]
    dice_after_lists = [[] for _ in range(num_of_masks)]
    tre_before_lists = [[] for _ in range(num_of_masks)]
    tre_after_lists = [[] for _ in range(num_of_masks)]

    for batch in tqdm(loader, desc="ANTs inferencing", total=len(loader)):
        assert batch["fdg_ct"].shape[0] == 1, (
            f"Expected batch size 1, got {batch['fdg_ct'].shape[0]}"
        )

        fdg_ct = batch["fdg_ct"]
        fdg_mask = batch["fdg_mask"]
        fdg_spacing = spacing_to_list(batch["fdg_spacing"])

        psma_ct = batch["psma_ct"]
        psma_mask = batch["psma_mask"]
        psma_spacing = spacing_to_list(batch["psma_spacing"])

        spacing = (torch.tensor(fdg_spacing) + torch.tensor(psma_spacing)) / 2

        warped_fdg_mask = register_fdg_ct_to_psma_ct(
            fdg_ct=fdg_ct,
            psma_ct=psma_ct,
            fdg_mask=fdg_mask,
            fdg_spacing=fdg_spacing,
            psma_spacing=psma_spacing,
            type_of_transform=type_of_transform,
            interpolator=interpolator,
        )

        for idx, _ in enumerate(masks_names):
            mask_idx = mask_list[idx]
            binary_mask_fdg = get_binary_mask_with_label(fdg_mask, mask_idx)
            binary_mask_psma = get_binary_mask_with_label(psma_mask, mask_idx)
            binary_mask_warped_fdg = get_binary_mask_with_label(warped_fdg_mask, mask_idx)

            dice_before_lists[idx].append(
                dice_metric(binary_mask_fdg, binary_mask_psma).cpu().item()
            )
            dice_after_lists[idx].append(
                dice_metric(binary_mask_warped_fdg, binary_mask_psma).cpu().item()
            )

            tre_before_lists[idx].append(
                compute_tre_single(binary_mask_fdg, binary_mask_psma, spacing).cpu().item()
            )
            tre_after_lists[idx].append(
                compute_tre_single(binary_mask_warped_fdg, binary_mask_psma, spacing)
                .cpu()
                .item()
            )

    save_registration_results(
        filename,
        masks_names,
        dice_before_lists,
        dice_after_lists,
        tre_before_lists,
        tre_after_lists,
    )


def main(args):
    output_path = make_output_path(args)
    if os.path.exists(output_path) and not args.overwrite:
        print(f"Skip existing result: {output_path}")
        return

    test_loader = build_test_loader(args)

    print(">>> ANTs registration: fdg_ct -> psma_ct")
    print(f">>> Transform: {args.type_of_transform}")
    print(f">>> Interpolator for masks: {args.interpolator}")
    print(f">>> Result: {output_path}")

    inference_ants_batch(
        test_loader,
        filename=output_path,
        type_of_transform=args.type_of_transform,
        interpolator=args.interpolator,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ANTs registration on the FDG/PSMA test set."
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=DEFAULT_RESULT_DIR,
        help="Directory where the metric txt file will be saved.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help="Name of the metric txt file.",
    )
    parser.add_argument(
        "--data_dirs",
        type=str,
        nargs="+",
        default=DEFAULT_DATA_DIRS,
        help="H5 data directories used for train/test split.",
    )
    parser.add_argument(
        "--test_counts",
        type=int,
        nargs="+",
        default=DEFAULT_TEST_COUNTS,
        help="Number of test cases to take from each data directory.",
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
        default="nearestNeighbor",
        help="ANTs interpolator used when warping masks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute results when the output txt already exists.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
