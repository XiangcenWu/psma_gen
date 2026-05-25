import argparse
import os
import sys

import torch
from monai.networks.nets import SwinUNETR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from General.data_loader import create_data_loader
from Registration.baseline_models import build_baseline_model
from Registration.smoothness_losses import l2_gradient
from Registration.train_baseline import get_mask_loss_inputs
from Registration.training import (
    get_ct_lambda,
    get_save_path,
    loss_function_dice,
    make_identity_grid_m11,
    predict_ddf_and_grid,
)
from RegistrationPsmaReg.dataloading import ReadH5PsmaRegd, get_train_test_h5_lists


DEFAULT_ROOT_DIR = "PSMAReg_h5"
DEFAULT_SPATIAL_SIZE = (128, 128, 288)
DEFAULT_REGISTRATION_INPUT_KEYS = ("moving_pet", "fixed_pet")
CT_REGISTRATION_INPUT_KEYS = (
    "moving_pet",
    "moving_ct",
    "fixed_pet",
    "fixed_ct",
)


def ensure_batched_channel_dim(tensor):
    if tensor.dim() == 3:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.dim() == 4:
        return tensor.unsqueeze(1)
    return tensor


def get_psmareg_registration_input_keys(use_ct_input=False):
    if use_ct_input:
        return CT_REGISTRATION_INPUT_KEYS
    return DEFAULT_REGISTRATION_INPUT_KEYS


def make_psmareg_registration_input(batch, input_keys, device):
    return torch.cat(
        [
            ensure_batched_channel_dim(batch[key]).float().to(device)
            for key in input_keys
        ],
        dim=1,
    )


def build_registration_model(model_name, in_channels=2):
    model_name = model_name.lower().replace("-", "_")
    if model_name in ("swinunetr", "swin_transformer", "swin"):
        return SwinUNETR(
            in_channels=in_channels,
            out_channels=3,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            downsample="mergingv2",
            use_v2=True,
        )

    if model_name == "vxm":
        model_name = "voxelmorph"

    return build_baseline_model(model_name, in_channels=in_channels)


def get_psmareg_model_save_path(args):
    save_path = get_save_path(args)
    save_dir = os.path.dirname(save_path)
    save_name = os.path.basename(save_path)

    model_name = args.registration_model.lower().replace("-", "_")
    if model_name == "vxm":
        model_name = "voxelmorph"

    if model_name in ("swinunetr", "swin_transformer", "swin"):
        return os.path.join(save_dir, f"psmareg_{save_name}")

    return os.path.join(save_dir, f"psmareg_{model_name}_{save_name}")


def train_psmareg_batch(
    model,
    loader,
    optimizer,
    identity_grid,
    smoothness_lambda=1000,
    ct_smoothness=False,
    ct_smoothness_margin=3000,
    ct_smoothness_gamma=1,
    num_masks=50,
    input_keys=None,
    mask_key="ct_label",
    device="cuda:0",
):
    if input_keys is None:
        input_keys = get_psmareg_registration_input_keys()

    model.train()
    model.to(device)
    identity_grid = identity_grid.to(device)

    moving_mask_key = f"moving_{mask_key}"
    fixed_mask_key = f"fixed_{mask_key}"

    step = 0.0
    loss_a = 0.0

    for batch in loader:
        moving_ct = ensure_batched_channel_dim(batch["moving_ct"]).float().to(device)
        fixed_ct = ensure_batched_channel_dim(batch["fixed_ct"]).float().to(device)

        moving_mask = ensure_batched_channel_dim(batch[moving_mask_key]).to(device)
        fixed_mask = ensure_batched_channel_dim(batch[fixed_mask_key]).to(device)

        model_input = make_psmareg_registration_input(batch, input_keys, device)

        ddf, grid = predict_ddf_and_grid(model, model_input, identity_grid)

        if ct_smoothness:
            tensor_weights = get_ct_lambda(
                moving_ct,
                ct_smoothness_margin,
                smoothness_lambda,
                ct_smoothness_gamma,
            )
            smoothness_loss = l2_gradient(ddf, tensor_weights)
        else:
            smoothness_loss = smoothness_lambda * l2_gradient(ddf)

        moving_masks, fixed_masks = get_mask_loss_inputs(
            moving_mask,
            fixed_mask,
            num_masks,
            device,
        )

        warped_moving_masks = torch.nn.functional.grid_sample(moving_masks, grid)
        warped_moving_ct = torch.nn.functional.grid_sample(moving_ct, grid)

        loss = (
            loss_function_dice(fixed_masks, warped_moving_masks)
            + loss_function_dice(warped_moving_ct, fixed_ct)
            + smoothness_loss
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_a += loss.item()
        step += 1.0

    return loss_a / step


def build_train_loader(args):
    train_list, test_list = get_train_test_h5_lists(
        root_dir=args.root_dir,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    train_transform = ReadH5PsmaRegd()

    train_loader = create_data_loader(
        train_list,
        train_transform,
        batch_size=args.batch_size,
    )

    return train_loader, train_list, test_list


def main(args):
    device = args.device
    input_keys = get_psmareg_registration_input_keys(args.use_ct_input)

    model = build_registration_model(
        args.registration_model,
        in_channels=len(input_keys),
    ).to(device)

    train_loader, train_list, test_list = build_train_loader(args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    identity_grid = make_identity_grid_m11(
        tuple(args.spatial_size),
        device=device,
    )

    save_path = get_psmareg_model_save_path(args)

    print(f">>> PSMAReg root = {args.root_dir}")
    print(f">>> Train cases = {len(train_list)}")
    print(f">>> Test cases = {len(test_list)}")
    print(f">>> Example test cases = {test_list[:3]}")
    print(f">>> Registration model = {args.registration_model}")
    print(f">>> Model input = {list(input_keys)}")
    print(f">>> Mask key = {args.mask_key}")
    print(f">>> Smoothness lambda = {args.smoothness}")
    print(f">>> CT smoothness = {args.ct_smoothness}")
    print(f">>> CT smoothness margin = {args.ct_smoothness_margin}")
    print(f">>> CT smoothness gamma = {args.ct_smoothness_gamma}")
    print(f">>> Model will be saved to: {save_path}")

    for epoch in range(args.epochs):
        loss_batch = train_psmareg_batch(
            model,
            train_loader,
            optimizer,
            identity_grid,
            smoothness_lambda=args.smoothness,
            ct_smoothness=args.ct_smoothness,
            ct_smoothness_margin=args.ct_smoothness_margin,
            ct_smoothness_gamma=args.ct_smoothness_gamma,
            num_masks=args.num_masks,
            input_keys=input_keys,
            mask_key=args.mask_key,
            device=device,
        )

        print(f"Epoch {epoch:03d} | Loss = {loss_batch:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"model saved at {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="3D Registration Training on PSMAReg H5 data."
    )

    parser.add_argument(
        "--registration_model",
        type=str,
        default="swinunetr",
        choices=[
            "swinunetr",
            "swin_transformer",
            "swin",
            "transmorph",
            "voxelmorph",
            "vxm",
        ],
        help="Registration backbone to train.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=DEFAULT_ROOT_DIR,
        help="Root directory containing PSMAReg .h5 files.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Per-subfolder test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=325,
        help="Random seed used for train/test split.",
    )
    parser.add_argument(
        "--mask_key",
        type=str,
        default="ct_label",
        choices=["ct_label", "pet_label", "body_label"],
        help="PSMAReg label key used as the weak supervision mask.",
    )
    parser.add_argument(
        "--smoothness",
        type=float,
        default=8000,
        help="Smoothness regularization weight (lambda).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=350,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--num_masks",
        type=int,
        default=5,
        help="Number of sampled masks for weak supervision (0 = use all shared labels).",
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size.",
    )
    parser.add_argument(
        "--spatial_size",
        type=int,
        nargs=3,
        default=DEFAULT_SPATIAL_SIZE,
        help="Input volume size used to build the identity grid.",
    )
    parser.add_argument(
        "--use_ct_input",
        action="store_true",
        help="Use [moving_pet, moving_ct, fixed_pet, fixed_ct] as model input.",
    )
    parser.add_argument(
        "--ct_smoothness",
        action="store_true",
        help="Enable CT as DDF smoothness regularization.",
    )
    parser.add_argument(
        "--ct_smoothness_margin",
        type=float,
        default=3000.0,
        help="Margin value for CT smoothness regularization.",
    )
    parser.add_argument(
        "--ct_smoothness_gamma",
        type=float,
        default=1.0,
        help="Gamma value for CT smoothness regularization.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
