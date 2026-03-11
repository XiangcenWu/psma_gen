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
from Generation.DDPM_Baseline import CTtoPETDiffusion


DEFAULT_DATA_DIRS = [
    "/data2/xiangcen/data/pet_gen/processed/batch1_h5_v2",
    "/data2/xiangcen/data/pet_gen/processed/batch2_h5_v2",
]
DEFAULT_VAL_COUNTS = [40, 40]


def parse_args():
    parser = argparse.ArgumentParser(description="3D DDPM inference")
    parser.add_argument("--weights-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
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
        help="Reference image key from the H5 loader",
    )
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
        "--split",
        choices=["train", "val", "all"],
        default="val",
        help="Which split to run inference on",
    )
    parser.add_argument("--seed", type=int, default=325)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def spacing_key_for(image_key):
    return f"{image_key.split('_', 1)[0]}_spacing"


def spacing_to_list(spacing):
    if isinstance(spacing, torch.Tensor):
        return [float(value) for value in spacing.flatten().tolist()[:3]]

    values = []
    for value in spacing:
        if isinstance(value, torch.Tensor):
            values.append(float(value.item()))
        else:
            values.append(float(value))
    return values[:3]


def tensor_to_itk(tensor, spacing):
    array = tensor.detach().cpu()[0, 0].numpy().transpose(2, 1, 0)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    return image


def get_file_list(args):
    train_list, val_list = split_multiple_train_test(
        args.data_dirs,
        args.val_counts,
        args.seed,
    )
    if args.split == "train":
        return train_list
    if args.split == "val":
        return val_list
    return train_list + val_list


def load_diffusion(weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device)
    num_train_timesteps = checkpoint.get("scheduler_config", {}).get(
        "num_train_timesteps", 1000
    )
    diffusion = CTtoPETDiffusion(
        device=device,
        num_train_timesteps=num_train_timesteps,
    )
    diffusion.model.load_state_dict(checkpoint["model_state_dict"])
    diffusion.model.eval()
    return diffusion


def save_case(sample_dir, batch, generated, input_key, target_key):
    input_spacing = spacing_to_list(batch[spacing_key_for(input_key)])
    target_spacing_key = spacing_key_for(target_key)
    target_spacing = (
        spacing_to_list(batch[target_spacing_key])
        if target_spacing_key in batch
        else input_spacing
    )

    sitk.WriteImage(
        tensor_to_itk(batch[input_key].float(), input_spacing),
        os.path.join(sample_dir, f"{input_key}.nii.gz"),
    )
    sitk.WriteImage(
        tensor_to_itk(generated.float(), target_spacing),
        os.path.join(sample_dir, f"generated_{target_key}.nii.gz"),
    )
    sitk.WriteImage(
        tensor_to_itk(batch[target_key].float(), target_spacing),
        os.path.join(sample_dir, f"{target_key}.nii.gz"),
    )


def main(args):
    if len(args.data_dirs) != len(args.val_counts):
        raise ValueError("--data-dirs and --val-counts must have the same length")

    file_list = get_file_list(args)
    if args.max_samples is not None:
        file_list = file_list[: args.max_samples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_loader = create_data_loader(
        file_list,
        ReadH5d(),
        batch_size=1,
        shuffle=False,
    )
    diffusion = load_diffusion(args.weights_path, args.device)

    print(f">>> Conditioning: {args.input_key} -> {args.target_key}")
    print(f">>> Samples: {len(file_list)}")
    print(f">>> Outputs will be saved to: {output_dir}")

    for sample_path, batch in tqdm(
        zip(file_list, test_loader),
        total=len(file_list),
        desc="Generating",
    ):
        sample_name = Path(sample_path).stem
        sample_dir = output_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        condition = batch[args.input_key].float().to(args.device)
        generated = diffusion.generate(
            condition,
            num_inference_steps=args.num_inference_steps,
        ).clamp_(0.0, 1.0)

        save_case(sample_dir, batch, generated, args.input_key, args.target_key)


if __name__ == "__main__":
    main(parse_args())
