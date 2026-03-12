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




def map_zero_one_to_minus_one_one(image):
    return image * 2.0 - 1.0


def get_pair(
    batch,
    input_key,
    target_key,
    device,
    use_fdg_condition=False,
    fdg_key="fdg_pt",
):
    condition = batch[input_key].float().to(device)
    if use_fdg_condition:
        fdg = batch[fdg_key].float().to(device)
        condition = torch.cat([condition, fdg], dim=1)
    target = batch[target_key].float().to(device)
    condition = map_zero_one_to_minus_one_one(condition)
    target = map_zero_one_to_minus_one_one(target)
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


def train_epoch(
    diffusion,
    loader,
    optimizer,
    input_key,
    target_key,
    device,
    epoch,
    epochs,
    use_fdg_condition=False,
    fdg_key="fdg_pt",
):
    diffusion.model.train()
    losses = []
    progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

    for batch in progress:
        condition, target = get_pair(
            batch,
            input_key,
            target_key,
            device,
            use_fdg_condition,
            fdg_key,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(diffusion, condition, target)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)
        progress.set_postfix(loss=f"{loss_value:.4f}")

    return float(np.mean(losses))


@torch.no_grad()
def validate_epoch(
    diffusion,
    loader,
    input_key,
    target_key,
    device,
    use_fdg_condition=False,
    fdg_key="fdg_pt",
):
    diffusion.model.eval()
    losses = []

    for batch in tqdm(loader, desc="Validating", leave=False):
        condition, target = get_pair(
            batch,
            input_key,
            target_key,
            device,
            use_fdg_condition,
            fdg_key,
        )
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
        
        # 策略：每 100 个 epoch 保存一个带版本号的模型，并在最后一个 epoch 强制保存
        # if (epoch + 1) % 100 == 0 or epoch == args.epochs - 1:
        save_path = save_dir / f"model_.pth"
        diffusion.save(save_path)
        print(f'>>> Checkpoint saved: {save_path}')


if __name__ == "__main__":
    main(parse_args())
