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
    parser = argparse.ArgumentParser(description="3D DDPM training")
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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=325)
    parser.add_argument("--save-dir", type=str, default="./checkpoints/ddpm")
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fdg_key = FDG_CONDITION_KEYS[args.fdg_condition]
    use_fdg_condition = fdg_key is not None
    data_dirs = args.data_dirs
    if data_dirs is None:
        data_dirs = (
            DEFAULT_WARPED_DATA_DIRS
            if args.fdg_condition == "warped_fdg"
            else DEFAULT_DATA_DIRS
        )
    if len(data_dirs) != len(args.val_counts):
        raise ValueError("--data-dirs and --val-counts must have the same length")

    train_transform = ReadH5d()
    train_list, test_list = split_multiple_train_test(
        data_dirs,
        args.val_counts,
    )

    train_loader = create_data_loader(
        train_list, train_transform, batch_size=args.batch_size
    )


    condition_channels = 2 if use_fdg_condition else 1
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
    if use_fdg_condition:
        condition_desc = f"{args.input_key} + {fdg_key}"
    print(f">>> Conditioning: {condition_desc} -> {args.target_key}")
    print(f">>> Train samples: {len(train_list)} | test samples: {len(test_list)}")
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
            use_fdg_condition,
            fdg_key,
        )

        scheduler.step()

        print(
            f"Epoch {epoch:03d} | Train Loss = {train_loss:.6f}"
        )

        if use_fdg_condition:
            save_path = save_dir / f"{args.input_key}_{fdg_key}_to_{args.target_key}.pth"
        else:
            save_path = save_dir / f"{args.input_key}_to_{args.target_key}.pth"


    diffusion.save(save_path)
    print(f'>>> Checkpoint saved: {save_path}')


if __name__ == "__main__":
    main(parse_args())
