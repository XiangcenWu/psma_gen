import argparse
import os
import sys

import torch
from monai.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from General.segments import SEGMENT_INDEX
from Registration.baseline_models import build_baseline_model
from Registration.inferencing import (
    compute_tre_single,
    dice_metric,
    get_binary_mask_with_label,
    save_registration_results,
)
from Registration.training import make_identity_grid_m11, predict_ddf_and_grid
from RegistrationPsmaReg.dataloading import ReadH5PsmaRegd, get_train_test_h5_lists


DEFAULT_ROOT_DIR = "/data2/xiangcen/data/PSMAReg_h5/"
DEFAULT_SPATIAL_SIZE = (160, 160, 288)
DEFAULT_REGISTRATION_INPUT_KEYS = ("moving_pet", "fixed_pet")
CT_REGISTRATION_INPUT_KEYS = (
    "moving_pet",
    "moving_ct",
    "fixed_pet",
    "fixed_ct",
)


DEFAULT_RESULT_DIR = "/share/home/xcwu/psmareg_reg_results"
CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".ptm"}


def ensure_batched_channel_dim(tensor):
    if tensor.dim() == 3:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.dim() == 4:
        return tensor.unsqueeze(1)
    return tensor


def get_psmareg_registration_input_keys(use_ct_input=False):
    if use_ct_input:
        return CT_REGISTRATION_INPUT_KEYS
    return DEFAULT_REGISTRATION_INPUT_KEYS


def make_psmareg_registration_input(batch, input_keys, device):
    return torch.cat(
        [
            ensure_batched_channel_dim(batch[key]).float().to(device)
            for key in input_keys
        ],
        dim=1,
    )


def infer_model_name(weights_path, fallback):
    if fallback != "auto":
        return fallback

    name = os.path.basename(weights_path).lower().replace("-", "_")
    if "transmorph" in name:
        return "transmorph"
    if "voxelmorph" in name or "vxm" in name:
        return "voxelmorph"

    raise ValueError(
        "Cannot infer baseline model from checkpoint name. "
        "Please pass --baseline_model voxelmorph or transmorph."
    )


def load_state_dict(model, weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)


def list_weight_paths(weights_path):
    if not os.path.isdir(weights_path):
        return [weights_path]

    paths = []
    for filename in os.listdir(weights_path):
        path = os.path.join(weights_path, filename)
        _, extension = os.path.splitext(filename)
        if os.path.isfile(path) and extension.lower() in CHECKPOINT_EXTENSIONS:
            paths.append(path)

    return sorted(paths)


def make_output_path(weights_path, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    output_name = os.path.splitext(os.path.basename(weights_path))[0] + ".txt"
    return os.path.join(result_dir, output_name)


def spacing_to_list(spacing):
    if torch.is_tensor(spacing):
        return [float(v) for v in spacing.reshape(-1).tolist()]

    values = []
    for item in spacing:
        if torch.is_tensor(item):
            values.append(float(item.reshape(-1)[0].item()))
        else:
            values.append(float(item))
    return values


def get_mask_labels(mask_names):
    mask_labels = []
    for name in mask_names:
        if name in SEGMENT_INDEX:
            mask_labels.append(SEGMENT_INDEX[name])
            continue

        try:
            mask_labels.append(int(name))
        except ValueError as exc:
            raise ValueError(
                f"Unknown mask name '{name}'. Use a key from SEGMENT_INDEX "
                "or pass a numeric label id."
            ) from exc

    return mask_labels


def single_case_collate(batch):
    if len(batch) != 1:
        raise ValueError("PSMAReg inference expects DataLoader batch_size=1.")
    return batch[0]


def iter_pair_batches(batch):
    if isinstance(batch, dict):
        yield batch
        return

    if isinstance(batch, (list, tuple)):
        for item in batch:
            yield from iter_pair_batches(item)
        return

    raise TypeError(f"Unsupported PSMAReg inference batch type: {type(batch)}")


@torch.no_grad()
def inference_psmareg_batch(
    model,
    loader,
    identity_grid,
    filename,
    masks_names=None,
    input_keys=None,
    mask_key="ct_label",
    device="cuda:0",
):
    if masks_names is None:
        masks_names = list(SEGMENT_INDEX.keys())
    if input_keys is None:
        input_keys = get_psmareg_registration_input_keys()

    mask_labels = get_mask_labels(masks_names)
    moving_mask_key = f"moving_{mask_key}"
    fixed_mask_key = f"fixed_{mask_key}"

    model.eval()
    model.to(device)
    identity_grid = identity_grid.to(device)

    num_masks = len(masks_names)
    dice_before_lists = [[] for _ in range(num_masks)]
    dice_after_lists = [[] for _ in range(num_masks)]
    tre_before_lists = [[] for _ in range(num_masks)]
    tre_after_lists = [[] for _ in range(num_masks)]

    for patient_batch in tqdm(loader, desc="inferencing", total=len(loader)):
        for batch in iter_pair_batches(patient_batch):
            model_input = make_psmareg_registration_input(batch, input_keys, device)
            if model_input.shape[0] != 1:
                raise ValueError(
                    f"Expected batch size 1 for inference, got {model_input.shape[0]}"
                )

            moving_mask = ensure_batched_channel_dim(batch[moving_mask_key]).to(device)
            fixed_mask = ensure_batched_channel_dim(batch[fixed_mask_key]).to(device)

            moving_spacing = spacing_to_list(batch["moving_spacing"])
            fixed_spacing = spacing_to_list(batch["fixed_spacing"])
            spacing = [
                (moving_value + fixed_value) / 2.0
                for moving_value, fixed_value in zip(moving_spacing, fixed_spacing)
            ]

            _, grid = predict_ddf_and_grid(model, model_input, identity_grid)

            for idx, label in enumerate(mask_labels):
                binary_moving_mask = get_binary_mask_with_label(
                    moving_mask,
                    label,
                ).float()
                binary_fixed_mask = get_binary_mask_with_label(
                    fixed_mask,
                    label,
                ).float()

                dice_before_lists[idx].append(
                    dice_metric(binary_moving_mask, binary_fixed_mask).cpu().item()
                )
                tre_before_lists[idx].append(
                    compute_tre_single(
                        binary_moving_mask,
                        binary_fixed_mask,
                        spacing,
                    )
                    .cpu()
                    .item()
                )

                warped_moving_mask = torch.nn.functional.grid_sample(
                    binary_moving_mask,
                    grid,
                    align_corners=True,
                )

                dice_after_lists[idx].append(
                    dice_metric(warped_moving_mask, binary_fixed_mask).cpu().item()
                )
                tre_after_lists[idx].append(
                    compute_tre_single(
                        warped_moving_mask,
                        binary_fixed_mask,
                        spacing,
                    )
                    .cpu()
                    .item()
                )
    print(f"len of dice_before_lists: {[len(lst) for lst in dice_before_lists]}")
    print(f"len of dice_after_lists: {[len(lst) for lst in dice_after_lists]}")

    save_registration_results(
        filename,
        masks_names,
        dice_before_lists,
        dice_after_lists,
        tre_before_lists,
        tre_after_lists,
    )


def build_test_loader(args):
    _, test_list = get_train_test_h5_lists(
        root_dir=args.root_dir,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    transform = ReadH5PsmaRegd(pair_mode=args.pair_mode)

    test_dataset = Dataset(test_list, transform)
    test_loader = DataLoader(
        test_dataset,
        num_workers=args.num_workers,
        batch_size=1,
        collate_fn=single_case_collate,
        shuffle=False,
    )

    return test_loader, test_list


def run_one_checkpoint(args, weights_path, test_loader, identity_grid):
    model_name = infer_model_name(weights_path, args.baseline_model)
    input_keys = get_psmareg_registration_input_keys(args.use_ct_input)

    base_model = build_baseline_model(model_name, in_channels=len(input_keys))
    load_state_dict(base_model, weights_path, args.device)

    output_path = make_output_path(weights_path, args.result_dir)
    if os.path.exists(output_path) and not args.overwrite:
        print(f"Skip existing result: {output_path}")
        return

    print(f">>> Baseline model: {model_name}")
    print(f">>> Model input: {list(input_keys)}")
    print(f">>> Mask key: {args.mask_key}")
    print(f">>> Weights: {weights_path}")
    print(f">>> Result: {output_path}")

    inference_psmareg_batch(
        base_model,
        test_loader,
        identity_grid,
        filename=output_path,
        masks_names=args.masks_names,
        input_keys=input_keys,
        mask_key=args.mask_key,
        device=args.device,
    )


def main(args):
    test_loader, test_list = build_test_loader(args)
    identity_grid = make_identity_grid_m11(
        tuple(args.spatial_size),
        device=args.device,
    )

    weights_paths = list_weight_paths(args.weights_path)
    if not weights_paths:
        raise FileNotFoundError(f"No checkpoint files found in {args.weights_path}")

    print(f">>> PSMAReg root: {args.root_dir}")
    print(f">>> Test cases: {len(test_list)}")
    print(f">>> Pair mode: {args.pair_mode}")
    print(f">>> Example test cases: {test_list[:3]}")

    for weights_path in weights_paths:
        run_one_checkpoint(args, weights_path, test_loader, identity_grid)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline registration checkpoints on PSMAReg H5 data."
    )

    parser.add_argument(
        "--weights_path",
        type=str,
        required=True,
        help="Path to one checkpoint, or a directory containing .pt/.pth/.ptm files.",
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="auto",
        choices=[
            "auto",
            "voxelmorph",
            "transmorph",
        ],
        help="Baseline architecture. Use auto when checkpoint names include the model tag.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=DEFAULT_RESULT_DIR,
        help="Directory where metric txt files will be saved.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=DEFAULT_ROOT_DIR,
        help="Root directory containing PSMAReg .h5 files.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Per-subfolder test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=325,
        help="Random seed used for train/test split.",
    )
    parser.add_argument(
        "--pair_mode",
        type=str,
        default="all_pairs",
        choices=["all_pairs", "all_adjacent", "random_adjacent"],
        help=(
            "Timepoint-pair sampling mode. all_pairs evaluates every "
            "earlier-to-later timepoint pair in each H5 file."
        ),
    )
    parser.add_argument(
        "--mask_key",
        type=str,
        default="ct_label",
        choices=["ct_label", "pet_label", "body_label"],
        help="PSMAReg label key used for Dice/TRE evaluation.",
    )
    parser.add_argument(
        "--masks_names",
        type=str,
        nargs="+",
        default=list(SEGMENT_INDEX.keys()),
        help=(
            "Segment names from SEGMENT_INDEX, or numeric label ids. "
            "Defaults to all SEGMENT_INDEX labels."
        ),
    )
    parser.add_argument(
        "--spatial_size",
        type=int,
        nargs=3,
        default=DEFAULT_SPATIAL_SIZE,
        help="Input volume size used to build the identity grid.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run on, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute results when the output txt already exists.",
    )
    parser.add_argument(
        "--use_ct_input",
        action="store_true",
        help="Use [moving_pet, moving_ct, fixed_pet, fixed_ct] as model input.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
