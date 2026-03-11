import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from General.data_loader import ReadH5d, create_data_loader
from General.dataset_sample import split_multiple_train_test
from Generation.DDPM_Baseline import CTtoPETDiffusion


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




def get_pair(batch, input_key, target_key, device):
    condition = batch[input_key].float().to(device)
    target = batch[target_key].float().to(device)
    return condition, target


def add_noise_3d(diffusion, target, noise, timesteps):
    view_shape = (timesteps.shape[0],) + (1,) * (target.ndim - 1)
    alpha_schedule = diffusion.scheduler.sqrt_alphas_cumprod
    sigma_schedule = diffusion.scheduler.sqrt_one_minus_alphas_cumprod
    schedule_timesteps = timesteps.to(alpha_schedule.device)

    alpha = alpha_schedule[schedule_timesteps].to(target.device).view(view_shape)
    sigma = sigma_schedule[schedule_timesteps].to(target.device).view(view_shape)
    return alpha * target + sigma * noise


def compute_loss(diffusion, condition, target):
    batch_size = target.shape[0]
    timesteps = torch.randint(
        0,
        diffusion.scheduler.num_train_timesteps,
        (batch_size,),
        device=target.device,
        dtype=torch.long,
    )
    noise = torch.randn_like(target)
    noisy_target = add_noise_3d(diffusion, target, noise, timesteps)
    model_input = torch.cat([condition, noisy_target], dim=1)
    noise_pred = diffusion.model(model_input, timesteps)
    return F.mse_loss(noise_pred, noise)


def train_epoch(diffusion, loader, optimizer, input_key, target_key, device, epoch, epochs):
    diffusion.model.train()
    losses = []
    progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

    for batch in progress:
        condition, target = get_pair(batch, input_key, target_key, device)
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(diffusion, condition, target)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)
        progress.set_postfix(loss=f"{loss_value:.4f}")

    return float(np.mean(losses))


@torch.no_grad()
def validate_epoch(diffusion, loader, input_key, target_key, device):
    diffusion.model.eval()
    losses = []

    for batch in tqdm(loader, desc="Validating", leave=False):
        condition, target = get_pair(batch, input_key, target_key, device)
        losses.append(compute_loss(diffusion, condition, target).item())

    return float(np.mean(losses))


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


    diffusion = CTtoPETDiffusion(
        device=args.device,
        num_train_timesteps=args.num_train_timesteps,
    )
    optimizer = torch.optim.AdamW(diffusion.model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val_loss = float("inf")

    print(f">>> Conditioning: {args.input_key} -> {args.target_key}")
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
        )

        scheduler.step()

        print(
            f"Epoch {epoch:03d} | Train Loss = {train_loss:.6f}"
        )

    diffusion.save(save_dir / "final_model.pth")


if __name__ == "__main__":
    main(parse_args())
