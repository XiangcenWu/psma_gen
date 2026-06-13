import argparse
import os
import sys
from pathlib import Path

import SimpleITK as sitk
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from General.data_loader import ReadH5d, create_data_loader
from General.dataset_sample import split_multiple_train_test
from General.save_itk import tensor_to_itk
from Generation.latent_diffusion import (
    DEFAULT_DATA_DIRS,
    FDG_CONDITION_KEYS,
    AutoencoderKL3D,
    LatentDiffusionModel,
    get_latent_pair,
    optional_checkpoint,
)
from Generation.utils import map_minus_one_one_to_zero_one


def parse_args():
    parser = argparse.ArgumentParser(description="3D latent diffusion inference")
    parser.add_argument("--input-key", default="psma_ct")
    parser.add_argument("--target-key", default="psma_pt")
    parser.add_argument(
        "--fdg-condition",
        choices=tuple(FDG_CONDITION_KEYS),
        default="warped_fdg",
    )
    parser.add_argument("--target-vae-checkpoint", required=True)
    parser.add_argument(
        "--condition-vae-checkpoint",
        default="none",
        help="Condition VAE checkpoint, or 'none' for trilinear downsampling",
    )
    parser.add_argument("--diffusion-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--image-size",
        nargs=3,
        type=int,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Expected training image size; checked against the checkpoint config",
    )
    parser.add_argument("--num-inference-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=325)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-flash-attention", action="store_true")
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


def load_checkpoint(path, device, name):
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{name} checkpoint does not exist: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def model_state(checkpoint, key="model_state_dict"):
    if key in checkpoint:
        return checkpoint[key]
    return checkpoint


def write_volume(tensor, path):
    sitk.WriteImage(tensor_to_itk(tensor), str(path))


@torch.no_grad()
def run_inference(
    loader,
    diffusion,
    target_vae,
    condition_vae,
    args,
    config,
):
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    fdg_key = FDG_CONDITION_KEYS[args.fdg_condition]
    image_size = tuple(config["image_size"])
    downsample_factor = 2 ** (len(config["vae_channels"]) - 1)
    latent_size = tuple(size // downsample_factor for size in image_size)
    target_latent_scale = float(config.get("target_latent_scale", 1.0))
    condition_latent_scale = float(config.get("condition_latent_scale", 1.0))

    target_vae.eval()
    if condition_vae is not None:
        condition_vae.eval()
    diffusion.model.eval()

    sample_index = 0
    progress = tqdm(loader, desc="Latent diffusion inference")
    for batch in progress:
        condition, target = get_latent_pair(
            batch,
            args.input_key,
            args.target_key,
            args.device,
            image_size,
            fdg_key,
        )
        with torch.cuda.amp.autocast(enabled=args.amp):
            prediction = diffusion.generate(
                target_vae,
                condition,
                latent_size,
                target_latent_scale=target_latent_scale,
                condition_autoencoder=condition_vae,
                condition_latent_scale=condition_latent_scale,
                num_inference_steps=args.num_inference_steps,
            )

        prediction = map_minus_one_one_to_zero_one(
            prediction.float(),
            clamp_output=True,
        )
        target = map_minus_one_one_to_zero_one(target.float(), clamp_output=True)

        for batch_index in range(prediction.shape[0]):
            case_dir = output_dir / f"sample_{sample_index:04d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            write_volume(
                prediction[batch_index].unsqueeze(0),
                case_dir / "psma_prediction.nii.gz",
            )
            write_volume(
                target[batch_index].unsqueeze(0),
                case_dir / "psma_gt.nii.gz",
            )
            sample_index += 1
        progress.set_postfix(saved=sample_index)

    print(f">>> Saved {sample_index} samples to: {output_dir}")


def main(args):
    if len(args.data_dirs) != len(args.val_counts):
        raise ValueError("--data-dirs and --val-counts must have the same length")
    if args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive")

    args.amp = args.amp and str(args.device).startswith("cuda")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    diffusion_checkpoint = load_checkpoint(
        args.diffusion_checkpoint,
        args.device,
        "Diffusion",
    )
    if "config" not in diffusion_checkpoint:
        raise KeyError("Diffusion checkpoint is missing its 'config' section")
    config = diffusion_checkpoint["config"]
    checkpoint_image_size = tuple(int(size) for size in config["image_size"])
    if args.image_size is not None:
        requested_image_size = tuple(args.image_size)
        if requested_image_size != checkpoint_image_size:
            raise ValueError(
                "Inference image size does not match the diffusion checkpoint: "
                f"trained={checkpoint_image_size}, requested={requested_image_size}"
            )
    num_train_timesteps = int(config["num_train_timesteps"])
    if args.num_inference_steps > num_train_timesteps:
        raise ValueError(
            "--num-inference-steps cannot exceed the trained scheduler length "
            f"{num_train_timesteps}"
        )

    trained_fdg_condition = config.get("fdg_condition")
    if trained_fdg_condition != args.fdg_condition:
        raise ValueError(
            "Inference condition does not match the diffusion checkpoint: "
            f"trained={trained_fdg_condition}, requested={args.fdg_condition}"
        )

    target_checkpoint = load_checkpoint(
        args.target_vae_checkpoint,
        args.device,
        "Target VAE",
    )
    target_vae = AutoencoderKL3D(
        in_channels=1,
        latent_channels=int(config["target_latent_channels"]),
        channels=tuple(config["vae_channels"]),
    ).to(args.device)
    target_vae.load_state_dict(model_state(target_checkpoint))

    fdg_key = FDG_CONDITION_KEYS[args.fdg_condition]
    condition_input_channels = 1 + int(fdg_key is not None)
    condition_checkpoint_path = optional_checkpoint(args.condition_vae_checkpoint)
    condition_vae = None
    if condition_checkpoint_path is not None:
        if not config.get("uses_condition_vae", False):
            raise ValueError(
                "A condition VAE was provided, but the diffusion checkpoint was "
                "trained with downsampled conditions"
            )
        condition_checkpoint = load_checkpoint(
            condition_checkpoint_path,
            args.device,
            "Condition VAE",
        )
        condition_vae = AutoencoderKL3D(
            in_channels=condition_input_channels,
            latent_channels=int(config["condition_latent_channels"]),
            channels=tuple(config["vae_channels"]),
        ).to(args.device)
        condition_vae.load_state_dict(model_state(condition_checkpoint))
    elif config.get("uses_condition_vae", False):
        raise ValueError(
            "The diffusion checkpoint was trained with a condition VAE, so "
            "--condition-vae-checkpoint cannot be 'none'"
        )

    diffusion_condition_channels = (
        int(config["condition_latent_channels"])
        if condition_vae is not None
        else condition_input_channels
    )
    diffusion = LatentDiffusionModel(
        latent_channels=int(config["target_latent_channels"]),
        condition_channels=diffusion_condition_channels,
        diffusion_channels=tuple(config["diffusion_channels"]),
        num_train_timesteps=num_train_timesteps,
        use_flash_attention=args.use_flash_attention,
        device=args.device,
    )
    diffusion.model.load_state_dict(
        model_state(diffusion_checkpoint, key="diffusion_state_dict")
    )

    _, test_list = split_multiple_train_test(
        args.data_dirs,
        args.val_counts,
        seed=args.seed,
    )
    test_loader = create_data_loader(
        test_list,
        ReadH5d(),
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(f">>> Image size: {tuple(config['image_size'])}")
    print(f">>> Condition VAE: {condition_checkpoint_path or 'none'}")
    print(f">>> Test samples: {len(test_list)}")
    print(f">>> Inference steps: {args.num_inference_steps}")
    print(f">>> Mixed precision: {args.amp}")
    run_inference(
        test_loader,
        diffusion,
        target_vae,
        condition_vae,
        args,
        config,
    )


if __name__ == "__main__":
    main(parse_args())
