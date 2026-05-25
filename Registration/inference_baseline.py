import argparse
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from General.data_loader import ReadH5d, create_data_loader
from General.dataset_sample import split_multiple_train_test
from Registration.baseline_models import build_baseline_model
from Registration.diffeomorphic import get_diffeomorphic_tag
from Registration.inferencing import inference_batch
from Registration.training import get_registration_input_keys, make_identity_grid_m11


DEFAULT_DATA_DIRS = [
    "/data2/xiangcen/data/pet_gen/processed/batch1_h5_v2",
    "/data2/xiangcen/data/pet_gen/processed/batch2_h5_v2",
    "/data2/xiangcen/data/pet_gen/processed/batch3_h5_v2",
]
DEFAULT_TEST_COUNTS = [40, 40, 20]
DEFAULT_RESULT_DIR = "/share/home/xcwu/pet_reg_results"
CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".ptm"}


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


def make_output_path(
    weights_path,
    result_dir,
    diffeomorphic=False,
    velocity_scale=0.5,
    int_steps=7,
):
    os.makedirs(result_dir, exist_ok=True)
    output_stem = os.path.splitext(os.path.basename(weights_path))[0]
    diffeomorphic_tag = get_diffeomorphic_tag(
        diffeomorphic,
        velocity_scale,
        int_steps,
    )
    if diffeomorphic_tag and diffeomorphic_tag not in output_stem:
        output_stem += diffeomorphic_tag
    output_name = output_stem + ".txt"
    return os.path.join(result_dir, output_name)


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


def run_one_checkpoint(args, weights_path, test_loader, identity_grid):
    model_name = infer_model_name(weights_path, args.baseline_model)
    input_keys = get_registration_input_keys(args.use_ct_input)

    base_model = build_baseline_model(model_name, in_channels=len(input_keys))
    load_state_dict(base_model, weights_path, args.device)

    output_path = make_output_path(
        weights_path,
        args.result_dir,
        diffeomorphic=args.diffeomorphic,
        velocity_scale=args.velocity_scale,
        int_steps=args.int_steps,
    )
    if os.path.exists(output_path) and not args.overwrite:
        print(f"Skip existing result: {output_path}")
        return

    print(f">>> Baseline model: {model_name}")
    print(f">>> Model input: {list(input_keys)}")
    print(f">>> Diffeomorphic: {args.diffeomorphic}")
    print(f">>> Velocity scale: {args.velocity_scale}")
    print(f">>> Integration steps: {args.int_steps}")
    print(f">>> Weights: {weights_path}")
    print(f">>> Result: {output_path}")

    inference_batch(
        base_model,
        test_loader,
        identity_grid,
        filename=output_path,
        device=args.device,
        input_keys=input_keys,
        diffeomorphic=args.diffeomorphic,
        velocity_scale=args.velocity_scale,
        int_steps=args.int_steps,
    )


def main(args):
    if args.velocity_scale <= 0:
        raise ValueError("--velocity_scale must be > 0.")
    if args.int_steps < 0:
        raise ValueError("--int_steps must be >= 0.")

    test_loader = build_test_loader(args)
    identity_grid = make_identity_grid_m11(args.spatial_size, device=args.device)

    weights_paths = list_weight_paths(args.weights_path)
    if not weights_paths:
        raise FileNotFoundError(f"No checkpoint files found in {args.weights_path}")

    for weights_path in weights_paths:
        run_one_checkpoint(args, weights_path, test_loader, identity_grid)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline registration checkpoints on the test set."
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
        "--spatial_size",
        type=int,
        nargs=3,
        default=(128, 128, 384),
        help="Input volume size used to build the identity grid.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run on, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute results when the output txt already exists.",
    )
    parser.add_argument(
        "--use_ct_input",
        action="store_true",
        help="Use [fdg_pt, fdg_ct, psma_pt, psma_ct] as model input.",
    )
    parser.add_argument(
        "--diffeomorphic",
        action="store_true",
        help="Interpret model output as SVF and integrate it with scaling-and-squaring.",
    )
    parser.add_argument(
        "--velocity_scale",
        type=float,
        default=0.5,
        help="Scale applied to tanh(model_output) before SVF integration.",
    )
    parser.add_argument(
        "--int_steps",
        type=int,
        default=7,
        help="Number of scaling-and-squaring integration steps.",
    )
    parser.add_argument(
        "--ct_smoothness",
        action="store_true",
        help=(
            "Mark that the checkpoint was trained with CT smoothness. "
            "This is recorded for consistency; inference does not recompute loss."
        ),
    )
    parser.add_argument(
        "--ct_smoothness_margin",
        type=float,
        default=3000.0,
        help="Margin value used for CT smoothness during training.",
    )
    parser.add_argument(
        "--ct_smoothness_gamma",
        type=float,
        default=1.0,
        help="Gamma value used for CT smoothness during training.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
