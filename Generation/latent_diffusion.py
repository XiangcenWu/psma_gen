import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DiffusionModelUNet
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from General.data_loader import ReadH5d, create_data_loader
from General.dataset_sample import split_multiple_train_test
from Generation.utils import map_zero_one_to_minus_one_one


FDG_CONDITION_KEYS = {
    "none": None,
    "fdg": "fdg_pt",
    "warped_fdg": "warped_fdg_pet",
}
DEFAULT_DATA_DIRS = [
    "/data2/xiangcen/data/pet_gen/processed/warped_fdg_pet_h5/batch1_h5_v2",
    "/data2/xiangcen/data/pet_gen/processed/warped_fdg_pet_h5/batch2_h5_v2",
    "/data2/xiangcen/data/pet_gen/processed/warped_fdg_pet_h5/batch3_h5_v2",
]


def resize_volume(volume, image_size, mode="trilinear"):
    return F.interpolate(
        volume,
        size=tuple(image_size),
        mode=mode,
        align_corners=False,
    )


def get_latent_pair(
    batch,
    input_key,
    target_key,
    device,
    image_size=(128, 128, 384),
    fdg_key="warped_fdg_pet",
):
    """Load and resize the condition and target used by latent diffusion."""
    condition = batch[input_key].float().to(device)
    target = batch[target_key].float().to(device)

    condition = resize_volume(condition, image_size)
    target = resize_volume(target, image_size)

    if fdg_key is not None:
        if fdg_key not in batch:
            raise KeyError(
                f"FDG condition key '{fdg_key}' is missing from the loaded H5 data. "
                f"Available keys: {sorted(batch)}"
            )
        fdg = batch[fdg_key].float().to(device)
        fdg = resize_volume(fdg, image_size)
        condition = torch.cat([condition, fdg], dim=1)

    condition = map_zero_one_to_minus_one_one(condition)
    target = map_zero_one_to_minus_one_one(target)
    return condition, target


def normalization_groups(channels, maximum_groups=8):
    groups = min(maximum_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = normalization_groups(out_channels)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, inputs):
        return self.block(inputs)


class AutoencoderKL3D(nn.Module):
    """Small 3D KL autoencoder for target or condition volumes."""

    def __init__(
        self,
        in_channels=1,
        latent_channels=4,
        channels=(8, 16, 32, 64, 128),
    ):
        super().__init__()
        if len(channels) < 2:
            raise ValueError("Autoencoder channels must contain at least two levels")

        encoder_layers = [ConvBlock3D(in_channels, channels[0])]
        for current_channels, next_channels in zip(channels[:-1], channels[1:]):
            encoder_layers.extend(
                [
                    nn.Conv3d(
                        current_channels,
                        next_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    ConvBlock3D(next_channels, next_channels),
                ]
            )
        self.encoder = nn.Sequential(*encoder_layers)
        self.to_mean = nn.Conv3d(channels[-1], latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv3d(channels[-1], latent_channels, kernel_size=1)

        decoder_layers = [ConvBlock3D(latent_channels, channels[-1])]
        reversed_channels = list(reversed(channels))
        for current_channels, next_channels in zip(
            reversed_channels[:-1], reversed_channels[1:]
        ):
            decoder_layers.extend(
                [
                    nn.ConvTranspose3d(
                        current_channels,
                        next_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    ConvBlock3D(next_channels, next_channels),
                ]
            )
        decoder_layers.extend(
            [
                nn.Conv3d(channels[0], in_channels, kernel_size=3, padding=1),
                nn.Tanh(),
            ]
        )
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs, sample=True):
        features = self.encoder(inputs)
        mean = self.to_mean(features)
        logvar = self.to_logvar(features).clamp(-30.0, 20.0)
        if sample:
            latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        else:
            latent = mean
        return latent, mean, logvar

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, inputs):
        latent, mean, logvar = self.encode(inputs, sample=True)
        reconstruction = self.decode(latent)
        return reconstruction, mean, logvar


class DDPMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )

    def add_noise(self, original, noise, timesteps):
        view_shape = (timesteps.shape[0],) + (1,) * (original.ndim - 1)
        schedule_timesteps = timesteps.to(self.alphas_cumprod.device)
        alpha = self.sqrt_alphas_cumprod[schedule_timesteps]
        sigma = self.sqrt_one_minus_alphas_cumprod[schedule_timesteps]
        return (
            alpha.to(original.device).view(view_shape) * original
            + sigma.to(original.device).view(view_shape) * noise
        )

    def step(self, model_output, timestep, sample):
        timestep = int(timestep)
        beta = self.betas[timestep].to(sample.device)
        mean = self.sqrt_recip_alphas[timestep].to(sample.device) * (
            sample
            - beta
            / self.sqrt_one_minus_alphas_cumprod[timestep].to(sample.device)
            * model_output
        )
        if timestep == 0:
            return mean
        variance = torch.sqrt(self.posterior_variance[timestep].to(sample.device))
        return mean + variance * torch.randn_like(sample)


class LatentDiffusionModel:
    def __init__(
        self,
        latent_channels,
        condition_channels,
        diffusion_channels=(32, 64, 128),
        num_train_timesteps=1000,
        use_flash_attention=False,
        device="cuda",
    ):
        self.device = device
        self.latent_channels = latent_channels
        self.model = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=latent_channels + condition_channels,
            out_channels=latent_channels,
            num_res_blocks=tuple(2 for _ in diffusion_channels),
            channels=tuple(diffusion_channels),
            attention_levels=tuple(
                index > 0 for index in range(len(diffusion_channels))
            ),
            norm_num_groups=8,
            with_conditioning=False,
            resblock_updown=True,
            num_head_channels=8,
            use_flash_attention=use_flash_attention,
        ).to(device)
        self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

    @torch.no_grad()
    def generate_latent(
        self,
        condition,
        latent_size,
        condition_autoencoder=None,
        condition_latent_scale=1.0,
        num_inference_steps=1000,
    ):
        self.model.eval()
        batch_size = condition.shape[0]
        sample = torch.randn(
            (batch_size, self.latent_channels, *latent_size),
            device=self.device,
        )
        latent_condition = encode_condition(
            condition,
            latent_size,
            condition_autoencoder=condition_autoencoder,
            latent_scale=condition_latent_scale,
            sample=False,
        )
        timesteps = torch.linspace(
            self.scheduler.num_train_timesteps - 1,
            0,
            num_inference_steps,
            device=self.device,
        ).long()
        for timestep in tqdm(timesteps, desc="Sampling latent", leave=False):
            timestep_value = int(timestep.item())
            timestep_batch = torch.full(
                (batch_size,),
                timestep_value,
                device=self.device,
                dtype=torch.long,
            )
            model_input = torch.cat([latent_condition, sample], dim=1)
            noise_prediction = self.model(model_input, timestep_batch)
            sample = self.scheduler.step(noise_prediction, timestep_value, sample)
        return sample

    @torch.no_grad()
    def generate(
        self,
        autoencoder,
        condition,
        latent_size,
        target_latent_scale=1.0,
        condition_autoencoder=None,
        condition_latent_scale=1.0,
        num_inference_steps=1000,
    ):
        latent = self.generate_latent(
            condition,
            latent_size,
            condition_autoencoder=condition_autoencoder,
            condition_latent_scale=condition_latent_scale,
            num_inference_steps=num_inference_steps,
        )
        return autoencoder.decode(latent / target_latent_scale)


def encode_condition(
    condition,
    latent_size,
    condition_autoencoder=None,
    latent_scale=1.0,
    sample=True,
):
    if condition_autoencoder is None:
        return F.interpolate(
            condition,
            size=latent_size,
            mode="trilinear",
            align_corners=False,
        )

    condition_latent, _, _ = condition_autoencoder.encode(
        condition,
        sample=sample,
    )
    if condition_latent.shape[2:] != tuple(latent_size):
        condition_latent = F.interpolate(
            condition_latent,
            size=latent_size,
            mode="trilinear",
            align_corners=False,
        )
    return condition_latent * latent_scale


def kl_loss(mean, logvar):
    return -0.5 * torch.mean(1.0 + logvar - mean.square() - logvar.exp())


def train_vae_epoch(
    target_vae,
    condition_vae,
    loader,
    target_optimizer,
    condition_optimizer,
    target_scaler,
    condition_scaler,
    args,
    fdg_key,
    epoch,
):
    target_vae.train()
    condition_vae.train()
    total_losses = []
    target_losses = []
    condition_losses = []
    progress = tqdm(
        loader,
        desc=f"Target + condition VAE {epoch + 1}/{args.vae_epochs}",
        leave=False,
    )
    for batch in progress:
        condition, target = get_latent_pair(
            batch,
            args.input_key,
            args.target_key,
            args.device,
            args.image_size,
            fdg_key,
        )

        target_optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            target_reconstruction, target_mean, target_logvar = target_vae(target)
            target_reconstruction_loss = F.l1_loss(target_reconstruction, target)
            target_kl_loss = kl_loss(target_mean, target_logvar)
            target_loss = (
                target_reconstruction_loss + args.kl_weight * target_kl_loss
            )
        target_scaler.scale(target_loss).backward()
        target_scaler.step(target_optimizer)
        target_scaler.update()

        condition_optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            condition_reconstruction, condition_mean, condition_logvar = condition_vae(
                condition
            )
            condition_reconstruction_loss = F.l1_loss(
                condition_reconstruction,
                condition,
            )
            condition_kl_loss = kl_loss(condition_mean, condition_logvar)
            condition_loss = (
                condition_reconstruction_loss + args.kl_weight * condition_kl_loss
            )
        condition_scaler.scale(condition_loss).backward()
        condition_scaler.step(condition_optimizer)
        condition_scaler.update()

        target_loss_value = target_loss.item()
        condition_loss_value = condition_loss.item()
        total_loss_value = target_loss_value + condition_loss_value
        total_losses.append(total_loss_value)
        target_losses.append(target_loss_value)
        condition_losses.append(condition_loss_value)
        progress.set_postfix(
            total=f"{total_loss_value:.4f}",
            target=f"{target_loss_value:.4f}",
            condition=f"{condition_loss_value:.4f}",
        )
    return {
        "total": float(np.mean(total_losses)),
        "target": float(np.mean(target_losses)),
        "condition": float(np.mean(condition_losses)),
    }


def train_diffusion_epoch(
    diffusion,
    target_vae,
    condition_vae,
    loader,
    optimizer,
    scaler,
    args,
    fdg_key,
    epoch,
):
    diffusion.model.train()
    target_vae.eval()
    if condition_vae is not None:
        condition_vae.eval()
    losses = []
    progress = tqdm(
        loader,
        desc=f"Latent diffusion {epoch + 1}/{args.diffusion_epochs}",
        leave=False,
    )
    for batch in progress:
        condition, target = get_latent_pair(
            batch,
            args.input_key,
            args.target_key,
            args.device,
            args.image_size,
            fdg_key,
        )
        with torch.no_grad():
            latent, _, _ = target_vae.encode(target, sample=True)
            latent = latent * args.target_latent_scale
            latent_condition = encode_condition(
                condition,
                latent.shape[2:],
                condition_autoencoder=condition_vae,
                latent_scale=args.condition_latent_scale,
                sample=True,
            )

        timesteps = torch.randint(
            0,
            diffusion.scheduler.num_train_timesteps,
            (latent.shape[0],),
            device=latent.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(latent)
        noisy_latent = diffusion.scheduler.add_noise(latent, noise, timesteps)
        model_input = torch.cat([latent_condition, noisy_latent], dim=1)
        with torch.cuda.amp.autocast(enabled=args.amp):
            noise_prediction = diffusion.model(model_input, timesteps)
            loss = F.mse_loss(noise_prediction, noise)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return float(np.mean(losses))


def save_vae(
    path,
    vae,
    optimizer,
    epoch,
    args,
    role,
    input_channels,
    latent_channels,
):
    torch.save(
        {
            "model_state_dict": vae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "role": role,
            "input_channels": input_channels,
            "image_size": tuple(args.image_size),
            "latent_channels": latent_channels,
            "vae_channels": tuple(args.vae_channels),
        },
        path,
    )


def load_vae(path, vae, device, role):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    vae.load_state_dict(state_dict)
    print(f">>> Loaded {role} VAE: {path}")


def save_latent_diffusion(
    path,
    diffusion,
    target_vae,
    condition_vae,
    optimizer,
    epoch,
    args,
):
    torch.save(
        {
            "diffusion_state_dict": diffusion.model.state_dict(),
            "target_vae_state_dict": target_vae.state_dict(),
            "condition_vae_state_dict": (
                condition_vae.state_dict() if condition_vae is not None else None
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": {
                "image_size": tuple(args.image_size),
                "target_latent_channels": args.target_latent_channels,
                "condition_latent_channels": (
                    args.condition_latent_channels
                    if condition_vae is not None
                    else None
                ),
                "target_latent_scale": args.target_latent_scale,
                "condition_latent_scale": args.condition_latent_scale,
                "vae_channels": tuple(args.vae_channels),
                "diffusion_channels": tuple(args.diffusion_channels),
                "num_train_timesteps": args.num_train_timesteps,
                "fdg_condition": args.fdg_condition,
                "uses_condition_vae": condition_vae is not None,
            },
        },
        path,
    )


def optional_checkpoint(value):
    if value is None or str(value).strip().lower() == "none":
        return None
    return Path(value).expanduser()


def validate_args(args):
    downsample_factor = 2 ** (len(args.vae_channels) - 1)
    diffusion_downsample_factor = 2 ** (len(args.diffusion_channels) - 1)
    if any(size <= 0 for size in args.image_size):
        raise ValueError("--image-size values must be positive")
    if any(size % downsample_factor != 0 for size in args.image_size):
        raise ValueError(
            f"--image-size must be divisible by the autoencoder downsample factor "
            f"{downsample_factor}, got {tuple(args.image_size)}"
        )
    latent_size = tuple(size // downsample_factor for size in args.image_size)
    if any(size < diffusion_downsample_factor for size in latent_size):
        raise ValueError(
            f"Latent size {latent_size} is too small for diffusion channels "
            f"{tuple(args.diffusion_channels)}"
        )
    if len(args.data_dirs) != len(args.val_counts):
        raise ValueError("--data-dirs and --val-counts must have the same length")
    if args.target_latent_scale <= 0 or args.condition_latent_scale <= 0:
        raise ValueError("Latent scale values must be positive")
    if args.target_latent_channels <= 0 or args.condition_latent_channels <= 0:
        raise ValueError("Latent channel values must be positive")
    if any(channels <= 0 for channels in args.vae_channels):
        raise ValueError("--vae-channels values must be positive")
    if any(channels <= 0 for channels in args.diffusion_channels):
        raise ValueError("--diffusion-channels values must be positive")
    if any(channels % 8 != 0 for channels in args.diffusion_channels):
        raise ValueError("--diffusion-channels values must be divisible by 8")


def parse_args():
    parser = argparse.ArgumentParser(description="3D conditional latent diffusion")
    parser.add_argument(
        "--stage",
        choices=("vae", "diffusion", "all"),
        default="vae",
        help="Train both VAEs, train diffusion, or run both stages in sequence",
    )
    parser.add_argument("--input-key", default="psma_ct")
    parser.add_argument("--target-key", default="psma_pt")
    parser.add_argument(
        "--fdg-condition",
        choices=tuple(FDG_CONDITION_KEYS),
        default="warped_fdg",
    )
    parser.add_argument(
        "--image-size",
        nargs=3,
        type=int,
        default=[128, 128, 384],
        metavar=("X", "Y", "Z"),
        help="Volume size used by get_latent_pair, for example: 128 128 384",
    )
    parser.add_argument("--target-latent-channels", type=int, default=4)
    parser.add_argument("--condition-latent-channels", type=int, default=4)
    parser.add_argument("--target-latent-scale", type=float, default=1.0)
    parser.add_argument("--condition-latent-scale", type=float, default=1.0)
    parser.add_argument(
        "--vae-channels",
        nargs="+",
        type=int,
        default=[8, 16, 32, 64, 128],
    )
    parser.add_argument(
        "--diffusion-channels",
        nargs="+",
        type=int,
        default=[32, 64, 128],
    )
    parser.add_argument("--vae-epochs", type=int, default=100)
    parser.add_argument("--diffusion-epochs", type=int, default=500)
    parser.add_argument("--vae-lr", type=float, default=1e-4)
    parser.add_argument("--diffusion-lr", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=325)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA mixed precision to reduce training memory",
    )
    parser.add_argument("--use-flash-attention", action="store_true")
    parser.add_argument(
        "--target-vae-checkpoint",
        default=None,
        help="Target PSMA VAE checkpoint used for diffusion training",
    )
    parser.add_argument(
        "--condition-vae-checkpoint",
        default="none",
        help=(
            "Condition VAE checkpoint. Pass 'none' to downsample the condition "
            "with trilinear interpolation instead"
        ),
    )
    parser.add_argument(
        "--save-dir",
        default="./checkpoints/latent_diffusion",
    )
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        default=DEFAULT_DATA_DIRS,
    )
    parser.add_argument(
        "--val-counts",
        nargs="+",
        type=int,
        default=[40, 40, 20],
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main(args):
    validate_args(args)
    args.amp = args.amp and str(args.device).startswith("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fdg_key = FDG_CONDITION_KEYS[args.fdg_condition]
    condition_input_channels = 1 + int(fdg_key is not None)
    condition_name = args.input_key
    if fdg_key is not None:
        condition_name = f"{condition_name}_{fdg_key}"
    target_vae_path = save_dir / f"target_vae_{args.target_key}.pth"
    condition_vae_path = save_dir / f"condition_vae_{condition_name}.pth"
    diffusion_path = save_dir / (
        f"latent_{condition_name}_to_{args.target_key}.pth"
    )

    train_list, test_list = split_multiple_train_test(
        args.data_dirs,
        args.val_counts,
        seed=args.seed,
    )
    train_loader = create_data_loader(
        train_list,
        ReadH5d(),
        batch_size=args.batch_size,
    )

    target_vae = AutoencoderKL3D(
        in_channels=1,
        latent_channels=args.target_latent_channels,
        channels=tuple(args.vae_channels),
    ).to(args.device)
    condition_checkpoint = (
        optional_checkpoint(args.condition_vae_checkpoint)
        if args.stage == "diffusion"
        else None
    )
    use_condition_vae = args.stage in ("vae", "all") or condition_checkpoint is not None
    condition_vae = None
    if use_condition_vae:
        condition_vae = AutoencoderKL3D(
            in_channels=condition_input_channels,
            latent_channels=args.condition_latent_channels,
            channels=tuple(args.vae_channels),
        ).to(args.device)

    print(f">>> Stage: {args.stage}")
    print(f">>> Image size: {tuple(args.image_size)}")
    downsample_factor = 2 ** (len(args.vae_channels) - 1)
    latent_size = tuple(size // downsample_factor for size in args.image_size)
    print(f">>> VAE downsample factor: {downsample_factor}")
    print(f">>> Latent size: {latent_size}")
    print(f">>> Mixed precision: {args.amp}")
    print(f">>> Condition: {condition_name} -> {args.target_key}")
    print(f">>> Train samples: {len(train_list)} | test samples: {len(test_list)}")
    print(
        f">>> Target VAE parameters: "
        f"{sum(parameter.numel() for parameter in target_vae.parameters()):,}"
    )
    if condition_vae is not None:
        print(
            f">>> Condition VAE parameters: "
            f"{sum(parameter.numel() for parameter in condition_vae.parameters()):,}"
        )
    else:
        print(">>> Condition VAE: none (using trilinear downsampling)")

    if args.stage in ("vae", "all"):
        target_optimizer = torch.optim.AdamW(
            target_vae.parameters(),
            lr=args.vae_lr,
        )
        condition_optimizer = torch.optim.AdamW(
            condition_vae.parameters(),
            lr=args.vae_lr,
        )
        target_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            target_optimizer,
            T_max=max(args.vae_epochs, 1),
        )
        condition_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            condition_optimizer,
            T_max=max(args.vae_epochs, 1),
        )
        target_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        condition_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        for epoch in range(args.vae_epochs):
            losses = train_vae_epoch(
                target_vae,
                condition_vae,
                train_loader,
                target_optimizer,
                condition_optimizer,
                target_scaler,
                condition_scaler,
                args,
                fdg_key,
                epoch,
            )
            target_scheduler.step()
            condition_scheduler.step()
            save_vae(
                target_vae_path,
                target_vae,
                target_optimizer,
                epoch,
                args,
                role="target",
                input_channels=1,
                latent_channels=args.target_latent_channels,
            )
            save_vae(
                condition_vae_path,
                condition_vae,
                condition_optimizer,
                epoch,
                args,
                role="condition",
                input_channels=condition_input_channels,
                latent_channels=args.condition_latent_channels,
            )
            print(
                f"VAE epoch {epoch:03d} | Total = {losses['total']:.6f} | "
                f"Target = {losses['target']:.6f} | "
                f"Condition = {losses['condition']:.6f}"
            )
        print(f">>> Target VAE saved: {target_vae_path}")
        print(f">>> Condition VAE saved: {condition_vae_path}")

    if args.stage in ("diffusion", "all"):
        if args.stage == "diffusion":
            target_checkpoint = optional_checkpoint(args.target_vae_checkpoint)
            if target_checkpoint is None:
                target_checkpoint = target_vae_path
            if not target_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Target VAE checkpoint does not exist: {target_checkpoint}"
                )
            load_vae(target_checkpoint, target_vae, args.device, role="target")

            if condition_checkpoint is None:
                print(">>> Condition VAE: none (using trilinear downsampling)")
            else:
                if not condition_checkpoint.is_file():
                    raise FileNotFoundError(
                        "Condition VAE checkpoint does not exist: "
                        f"{condition_checkpoint}"
                    )
                load_vae(
                    condition_checkpoint,
                    condition_vae,
                    args.device,
                    role="condition",
                )

        for parameter in target_vae.parameters():
            parameter.requires_grad_(False)
        target_vae.eval()
        if condition_vae is not None:
            for parameter in condition_vae.parameters():
                parameter.requires_grad_(False)
            condition_vae.eval()

        diffusion_condition_channels = (
            args.condition_latent_channels
            if condition_vae is not None
            else condition_input_channels
        )

        diffusion = LatentDiffusionModel(
            latent_channels=args.target_latent_channels,
            condition_channels=diffusion_condition_channels,
            diffusion_channels=tuple(args.diffusion_channels),
            num_train_timesteps=args.num_train_timesteps,
            use_flash_attention=args.use_flash_attention,
            device=args.device,
        )
        diffusion_optimizer = torch.optim.AdamW(
            diffusion.model.parameters(),
            lr=args.diffusion_lr,
        )
        diffusion_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            diffusion_optimizer,
            T_max=max(args.diffusion_epochs, 1),
        )
        diffusion_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        for epoch in range(args.diffusion_epochs):
            loss = train_diffusion_epoch(
                diffusion,
                target_vae,
                condition_vae,
                train_loader,
                diffusion_optimizer,
                diffusion_scaler,
                args,
                fdg_key,
                epoch,
            )
            diffusion_scheduler.step()
            save_latent_diffusion(
                diffusion_path,
                diffusion,
                target_vae,
                condition_vae,
                diffusion_optimizer,
                epoch,
                args,
            )
            print(f"Latent diffusion epoch {epoch:03d} | Loss = {loss:.6f}")
        print(f">>> Latent diffusion saved: {diffusion_path}")


if __name__ == "__main__":
    main(parse_args())
