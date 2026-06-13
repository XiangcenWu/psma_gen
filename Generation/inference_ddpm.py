import argparse
import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from General.data_loader import ReadH5d, create_data_loader
from General.dataset_sample import split_multiple_train_test
from Generation.DDPM_Baseline import CTtoPETDiffusion, run_inference


FDG_CONDITION_KEYS = {
    "none": None,
    "fdg": "fdg_pt",
    "warped_fdg": "warped_fdg_pet",
}
DEFAULT_DATA_DIRS = [
    "/data2/xiangcen/data/pet_gen/processed/batch1_h5_v2",
    "/data2/xiangcen/data/pet_gen/processed/batch2_h5_v2",
    "/data2/xiangcen/data/pet_gen/processed/batch3_h5_v2",
]
DEFAULT_WARPED_DATA_DIRS = [
    "/data2/xiangcen/data/pet_gen/processed/warped_fdg_pet_h5/batch1_h5_v2",
    "/data2/xiangcen/data/pet_gen/processed/warped_fdg_pet_h5/batch2_h5_v2",
    "/data2/xiangcen/data/pet_gen/processed/warped_fdg_pet_h5/batch3_h5_v2",
]



def parse_args():
    parser = argparse.ArgumentParser(description="3D DDPM inference")
    parser.add_argument(
        "--input-key",
        type=str,
        default="psma_ct",
        help="Condition image key from the H5 loader",
    )
    parser.add_argument(
        "--target-key",
        type=str,
        default="psma_pt",
        help="Target image key from the H5 loader",
    )
    parser.add_argument(
        "--fdg-condition",
        choices=tuple(FDG_CONDITION_KEYS),
        default="none",
        help=(
            "FDG condition to concatenate with CT: none, fdg (fdg_pt), or "
            "warped_fdg (warped_fdg_pet)"
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the trained DDPM checkpoint",
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
        "--num-train-timesteps",
        type=int,
        default=1000,
        help="Fallback scheduler length if the checkpoint does not store it",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of DDPM reverse diffusion steps used at inference time",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4, set to 1 if out-of-memory)",
    )
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        default=None,
        help=(
            "H5 data directories. Defaults to the registered H5 directories for "
            "warped_fdg and the original directories otherwise"
        ),
    )
    parser.add_argument(
        "--val-counts",
        nargs="+",
        type=int,
        default=[40, 40, 20],
        help="Validation sample count for each data directory",
    )
    return parser.parse_args()





def main(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.fdg_key = FDG_CONDITION_KEYS[args.fdg_condition]
    args.use_fdg_condition = args.fdg_key is not None
    data_dirs = args.data_dirs
    if data_dirs is None:
        data_dirs = (
            DEFAULT_WARPED_DATA_DIRS
            if args.fdg_condition == "warped_fdg"
            else DEFAULT_DATA_DIRS
        )
    if len(data_dirs) != len(args.val_counts):
        raise ValueError("--data-dirs and --val-counts must have the same length")

    transform = ReadH5d()
    train_list, test_list = split_multiple_train_test(data_dirs, args.val_counts)
    print(test_list[:5])  # 打印前5个测试样本路径以验证

    test_loader = create_data_loader(
        test_list,
        transform,
        batch_size=args.batch_size,
        shuffle=False,
    )

    condition_channels = 2 if args.use_fdg_condition else 1

    diffusion = CTtoPETDiffusion(
        device=args.device,
        in_channels=condition_channels + 1,
        num_train_timesteps=args.num_train_timesteps,
    )
    diffusion.load(args.checkpoint_path)

    condition_desc = args.input_key
    if args.use_fdg_condition:
        condition_desc = f"{args.input_key} + {args.fdg_key}"

    print(f">>> Conditioning: {condition_desc} -> {args.target_key}")
    print(f">>> Test samples: {len(test_list)}")
    print(f">>> Checkpoint: {args.checkpoint_path}")
    print(f">>> Output directory: {args.output_dir}")
    print(f">>> Test loader batch size: {args.batch_size}")

    run_inference(diffusion, test_loader, args)


if __name__ == "__main__":
    main(parse_args())
