import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from General.data_loader import ReadH5d, create_data_loader
from General.dataset_sample import split_multiple_train_test
from Generation.CycleGAN_Baseline import CTtoPETCycleGAN, ImagePool, train_epoch


DEFAULT_DATA_DIRS = [
    "/data2/xiangcen/data/pet_gen/processed/batch1_h5_v2",
    "/data2/xiangcen/data/pet_gen/processed/batch2_h5_v2",
]
DEFAULT_VAL_COUNTS = [40, 40]


def parse_args():
    parser = argparse.ArgumentParser(description="3D CycleGAN training")
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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=325)
    parser.add_argument("--save-dir", type=str, default="./checkpoints/cyclegan")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--lambda-cycle",
        type=float,
        default=10.0,
        help="Weight for the cycle consistency loss",
    )
    parser.add_argument(
        "--lambda-identity",
        type=float,
        default=5.0,
        help="Weight for identity loss when source and target channels match",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=50,
        help="Replay buffer size used for discriminator updates",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=32,
        help="Base channel width used by the generators",
    )
    parser.add_argument(
        "--num-res-blocks",
        type=int,
        default=4,
        help="Number of residual blocks used by the generators",
    )
    parser.add_argument(
        "--discriminator-channels",
        type=int,
        default=32,
        help="Base channel width used by the discriminators",
    )
    parser.add_argument(
        "--discriminator-layers",
        type=int,
        default=3,
        help="Number of convolutional blocks used by the discriminators",
    )
    return parser.parse_args()


def main(args):
    if len(args.data_dirs) != len(args.val_counts):
        raise ValueError("--data-dirs and --val-counts must have the same length")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_transform = ReadH5d()
    train_list, val_list = split_multiple_train_test(
        args.data_dirs,
        args.val_counts,
        args.seed,
    )

    train_loader = create_data_loader(
        train_list,
        train_transform,
        batch_size=args.batch_size,
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

    optimizer_g = torch.optim.Adam(
        itertools.chain(
            cyclegan.generator_ab.parameters(),
            cyclegan.generator_ba.parameters(),
        ),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    optimizer_d = torch.optim.Adam(
        itertools.chain(
            cyclegan.discriminator_a.parameters(),
            cyclegan.discriminator_b.parameters(),
        ),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_g, T_max=args.epochs
    )
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_d, T_max=args.epochs
    )

    fake_a_pool = ImagePool(args.pool_size)
    fake_b_pool = ImagePool(args.pool_size)

    condition_desc = args.input_key
    if args.use_fdg_condition:
        condition_desc = f"{args.input_key} + {args.fdg_key}"

    print(f">>> Conditioning: {condition_desc} -> {args.target_key}")
    print(f">>> Train samples: {len(train_list)} | Val samples: {len(val_list)}")
    print(f">>> Model will be saved to: {save_dir}")
    if args.lambda_identity > 0 and input_channels != 1:
        print(">>> Identity loss will be disabled because source and target channels do not match.")

    for epoch in range(args.epochs):
        loss_dict = train_epoch(
            cyclegan,
            train_loader,
            optimizer_g,
            optimizer_d,
            args.input_key,
            args.target_key,
            args.device,
            epoch,
            args.epochs,
            args.use_fdg_condition,
            args.fdg_key,
            args.lambda_cycle,
            args.lambda_identity,
            fake_a_pool,
            fake_b_pool,
        )

        scheduler_g.step()
        scheduler_d.step()

        print(
            f"Epoch {epoch:03d} | "
            f"G = {loss_dict['generator']:.6f} | "
            f"D = {loss_dict['discriminator']:.6f} | "
            f"Cycle = {loss_dict['cycle']:.6f} | "
            f"Id = {loss_dict['identity']:.6f}"
        )

        if args.use_fdg_condition:
            save_path = save_dir / f"{args.input_key}_{args.fdg_key}_to_{args.target_key}.pth"
        else:
            save_path = save_dir / f"{args.input_key}_to_{args.target_key}.pth"

        cyclegan.save(save_path)
        print(f">>> Checkpoint saved: {save_path}")


if __name__ == "__main__":
    main(parse_args())
