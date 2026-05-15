import argparse
import os
import sys

import torch
from monai.transforms import Compose, Resized

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from General.data_loader import ReadH5d, create_data_loader
from General.dataset_sample import split_multiple_train_test
from llm_Registration.modernbert_registration_adapter import ModernBERTSwinUNETRRegistrationModel
from Registration.training import make_identity_grid_m11, train_batch_llm


SPATIAL_SIZE = (64, 64, 192)
IMAGE_KEYS = ("fdg_ct", "fdg_pt", "psma_ct", "psma_pt")
MASK_KEYS = ("fdg_mask", "psma_mask")


def ensure_parent_dir(file_path: str) -> None:
    parent_dir = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(parent_dir, exist_ok=True)


def main(args: argparse.Namespace) -> None:
    device = args.device

    model = ModernBERTSwinUNETRRegistrationModel(
        model_dir=args.hf_model_dir,
        spatial_size=SPATIAL_SIZE,
        image_channels=4,
        freeze_text_encoder=True,
    ).to(device)

    train_transform = Compose(
        [
            ReadH5d(),
            Resized(
                keys=IMAGE_KEYS + MASK_KEYS,
                spatial_size=SPATIAL_SIZE,
                mode=("trilinear", "trilinear", "trilinear", "trilinear", "nearest", "nearest"),
                align_corners=(False, False, False, False, None, None),
            ),
        ]
    )

    train_list, test_list = split_multiple_train_test(
        args.data_dirs,
        args.num_validations,
    )

    train_loader = create_data_loader(
        train_list,
        train_transform,
        batch_size=args.batch_size,
    )
    print(test_list[:3])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    identity_grid = make_identity_grid_m11(SPATIAL_SIZE, device=device)

    print(f">>> Spatial size = {SPATIAL_SIZE}")
    print(f">>> Max prompt organs = {args.max_prompt_organs}")
    print(f">>> Hugging Face model dir = {args.hf_model_dir}")
    if args.save_path:
        print(f">>> Model will be saved to: {args.save_path}")
        ensure_parent_dir(args.save_path)
        
    for epoch in range(3):
        loss_batch = train_batch_llm(
            model,
            train_loader,
            optimizer,
            identity_grid,
            max_prompt_organs=args.max_prompt_organs,
            device=device,
            zero_ddf=True,  # zero out DDF for the first few epochs to warm up the model
        )
        print(f"Epoch {epoch:03d} | Loss = {loss_batch:.6f}")

    for epoch in range(args.epochs):
        loss_batch = train_batch_llm(
            model,
            train_loader,
            optimizer,
            identity_grid,
            max_prompt_organs=args.max_prompt_organs,
            device=device,
        )
        print(f"Epoch {epoch:03d} | Loss = {loss_batch:.6f}")
        if args.save_path:
            torch.save(model.state_dict(), args.save_path)
            print(f"model saved at {args.save_path}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy text-guided 3D registration training")
    parser.add_argument(
        "--data_dirs",
        nargs="+",
        default=[
            "/data2/xiangcen/data/pet_gen/processed/batch1_h5_v2",
            "/data2/xiangcen/data/pet_gen/processed/batch2_h5_v2",
        ],
        help="One or more H5 data directories.",
    )
    parser.add_argument(
        "--num_validations",
        nargs="+",
        type=int,
        default=[40, 40],
        help="Validation count for each data directory.",
    )
    parser.add_argument(
        "--hf_model_dir",
        type=str,
        required=True,
        help="Local directory containing the ModernBERT tokenizer and model weights.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="Optional path for saving the trained model state_dict.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=350,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--max_prompt_organs",
        type=int,
        default=5,
        help="Maximum number of organ labels sampled into each prompt.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Training device.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
