import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from General.data_loader import ReadH5d, create_data_loader
from General.dataset_sample import split_multiple_train_test
from Generation.DDPM_Baseline import CTtoPETDiffusion, train_epoch


DEFAULT_DATA_DIRS = [
    "/data2/xiangcen/data/pet_gen/processed/batch1_h5_v2",
    "/data2/xiangcen/data/pet_gen/processed/batch2_h5_v2",
]
DEFAULT_VAL_COUNTS = [40, 40]


def parse_args():
    parser = argparse.ArgumentParser(description="3D DDPM training")
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
        help="Condition image key from the H5 loader",
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
        help="Concatenate FDG PET to CT as a two-channel condition",
    )
    parser.add_argument(
        "--fdg-key",
        type=str,
        default="fdg_pt",
        help="FDG PET key used when --use-fdg-condition is enabled",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=325)
    parser.add_argument("--save-dir", type=str, default="./checkpoints/ddpm")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--num-train-timesteps",
        type=int,
        default=1000,
        help="Noise schedule length used by the baseline DDPM",
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
        train_list, train_transform, batch_size=args.batch_size
    )


    condition_channels = 2 if args.use_fdg_condition else 1
    diffusion = CTtoPETDiffusion(
        device=args.device,
        in_channels=condition_channels + 1,
        num_train_timesteps=args.num_train_timesteps,
    )
    optimizer = torch.optim.AdamW(diffusion.model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )


    condition_desc = args.input_key
    if args.use_fdg_condition:
        condition_desc = f"{args.input_key} + {args.fdg_key}"
    print(f">>> Conditioning: {condition_desc} -> {args.target_key}")
    print(f">>> Train samples: {len(train_list)} | Val samples: {len(val_list)}")
    print(f">>> Model will be saved to: {save_dir}")

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            diffusion,
            train_loader,
            optimizer,
            args.input_key,
            args.target_key,
            args.device,
            epoch,
            args.epochs,
            args.use_fdg_condition,
            args.fdg_key,
        )

        scheduler.step()

        print(
            f"Epoch {epoch:03d} | Train Loss = {train_loss:.6f}"
        )

        if args.use_fdg_condition:
            save_path = save_dir / f"{args.input_key}_{args.fdg_key}_to_{args.target_key}.pth"
        else:
            save_path = save_dir / f"{args.input_key}_to_{args.target_key}.pth"


        diffusion.save(save_path)
        print(f'>>> Checkpoint saved: {save_path}')


if __name__ == "__main__":
    main(parse_args())
