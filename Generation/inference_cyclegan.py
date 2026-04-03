import argparse
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from General.data_loader import ReadH5d, create_data_loader
from General.dataset_sample import split_multiple_train_test
from Generation.CycleGAN_Baseline import CTtoPETCycleGAN, run_inference
from Generation.train_cyclegan import DEFAULT_DATA_DIRS, DEFAULT_VAL_COUNTS


def parse_args():
    parser = argparse.ArgumentParser(description="3D CycleGAN inference")
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        default=DEFAULT_DATA_DIRS,
        help="Directories that contain processed H5 files",
    )
    parser.add_argument(
        "--val-counts",
        nargs="+",
        type=int,
        default=DEFAULT_VAL_COUNTS,
        help="Validation sample counts for each directory",
    )
    parser.add_argument(
        "--input-key",
        type=str,
        default="psma_ct",
        help="Source image key from the H5 loader",
    )
    parser.add_argument(
        "--target-key",
        type=str,
        default="psma_pt",
        help="Target image key from the H5 loader",
    )
    parser.add_argument(
        "--use-fdg-condition",
        action="store_true",
        help="Concatenate FDG PET to CT as a two-channel source domain",
    )
    parser.add_argument(
        "--fdg-key",
        type=str,
        default="fdg_pt",
        help="FDG PET key used when --use-fdg-condition is enabled",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the trained CycleGAN checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory used to save per-case NIfTI outputs",
    )
    parser.add_argument("--seed", type=int, default=325)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4, set to 1 if out-of-memory)",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=32,
        help="Fallback generator base channels if the checkpoint does not store them",
    )
    parser.add_argument(
        "--num-res-blocks",
        type=int,
        default=4,
        help="Fallback number of generator residual blocks if the checkpoint does not store it",
    )
    parser.add_argument(
        "--discriminator-channels",
        type=int,
        default=32,
        help="Fallback discriminator base channels if the checkpoint does not store them",
    )
    parser.add_argument(
        "--discriminator-layers",
        type=int,
        default=3,
        help="Fallback discriminator depth if the checkpoint does not store it",
    )
    return parser.parse_args()


def main(args):
    if len(args.data_dirs) != len(args.val_counts):
        raise ValueError("--data-dirs and --val-counts must have the same length")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    transform = ReadH5d()
    _, test_list = split_multiple_train_test(
        args.data_dirs,
        args.val_counts,
        args.seed,
    )
    print(test_list[:5])

    test_loader = create_data_loader(
        test_list,
        transform,
        batch_size=args.batch_size,
        shuffle=False,
    )

    input_channels = 2 if args.use_fdg_condition else 1
    cyclegan = CTtoPETCycleGAN(
        device=args.device,
        input_channels=input_channels,
        output_channels=1,
        base_channels=args.base_channels,
        num_res_blocks=args.num_res_blocks,
        discriminator_channels=args.discriminator_channels,
        discriminator_layers=args.discriminator_layers,
    )
    cyclegan.load(args.checkpoint_path)

    condition_desc = args.input_key
    if args.use_fdg_condition:
        condition_desc = f"{args.input_key} + {args.fdg_key}"

    print(f">>> Conditioning: {condition_desc} -> {args.target_key}")
    print(f">>> Test samples: {len(test_list)}")
    print(f">>> Checkpoint: {args.checkpoint_path}")
    print(f">>> Output directory: {args.output_dir}")
    print(f">>> Test loader batch size: {args.batch_size}")

    run_inference(cyclegan, test_loader, args)


if __name__ == "__main__":
    main(parse_args())
