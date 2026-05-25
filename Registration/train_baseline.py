import argparse
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from General.data_loader import create_data_loader, ReadH5d
from General.dataset_sample import split_multiple_train_test
from Registration.baseline_models import build_baseline_model
from Registration.diffeomorphic import (
    get_diffeomorphic_tag,
    predict_diffeomorphic_ddf_and_grid,
)
from Registration.mask import sample_labels_to_binary, sample_shared_binary_masks
from Registration.smoothness_losses import l2_gradient
from Registration.training import (
    DEFAULT_REGISTRATION_INPUT_KEYS,
    get_ct_lambda,
    get_registration_input_keys,
    make_identity_grid_m11,
    make_registration_input,
    predict_ddf_and_grid,
    loss_function_dice,
)


DEFAULT_SAVE_DIR = "/share/home/xcwu/registration_v3"


def format_tag_value(value):
    return f"{float(value):g}"


def get_baseline_save_path(args):
    mask_tag = "" if args.num_masks == 0 else f"_k{args.num_masks}"
    input_tag = "_ctinput" if args.use_ct_input else ""
    model_tag = args.baseline_model.lower().replace("-", "_")
    ct_smoothness_tag = ""
    if getattr(args, "ct_smoothness", False):
        ct_smoothness_tag = (
            f"_ctsmoothness"
            f"_mar{format_tag_value(getattr(args, 'ct_smoothness_margin', 3000.0))}"
            f"_gam{getattr(args, 'ct_smoothness_gamma', 1.0):g}"
        )
    diffeomorphic_tag = get_diffeomorphic_tag(
        getattr(args, "diffeomorphic", False),
        getattr(args, "velocity_scale", 0.5),
        getattr(args, "int_steps", 7),
    )
    save_name = (
        f"{model_tag}_l{format_tag_value(args.smoothness)}"
        f"{mask_tag}{input_tag}{ct_smoothness_tag}{diffeomorphic_tag}.ptm"
    )
    return os.path.join(getattr(args, "save_dir", DEFAULT_SAVE_DIR), save_name)


def get_mask_loss_inputs(fdg_mask, psma_mask, num_masks, device):
    if num_masks != 0:
        return sample_shared_binary_masks(
            moving_mask=fdg_mask,
            fixed_mask=psma_mask,
            num_samples=num_masks,
            device=device,
        )

    return (
        sample_labels_to_binary(fdg_mask),
        sample_labels_to_binary(psma_mask),
    )

def train_baseline_batch(
    model,
    loader,
    optimizer,
    identity_grid,
    smoothness_lambda=1000,
    ct_smoothness=False,
    ct_smoothness_margin=3000,
    ct_smoothness_gamma=1,
    num_masks=50,
    input_keys=DEFAULT_REGISTRATION_INPUT_KEYS,
    diffeomorphic=False,
    velocity_scale=0.5,
    int_steps=7,
    device="cuda:0",
):
    model.train()
    model.to(device)
    identity_grid = identity_grid.to(device)

    step = 0.0
    loss_a = 0.0

    for batch in loader:
        fdg_ct = batch["fdg_ct"].to(device)
        psma_ct = batch["psma_ct"].to(device)

        fdg_pt = batch["fdg_pt"].to(device)
        fdg_mask = batch["fdg_mask"].to(device)

        psma_pt = batch["psma_pt"].to(device)
        psma_mask = batch["psma_mask"].to(device)

        model_input = make_registration_input(batch, input_keys, device)

        if diffeomorphic:
            ddf, grid = predict_diffeomorphic_ddf_and_grid(
                model,
                model_input,
                identity_grid,
                velocity_scale=velocity_scale,
                int_steps=int_steps,
            )
        else:
            ddf, grid = predict_ddf_and_grid(model, model_input, identity_grid)

        if ct_smoothness:
            tensor_weights = get_ct_lambda(
                fdg_ct,
                ct_smoothness_margin,
                smoothness_lambda,
                ct_smoothness_gamma,
            )
            smoothness_loss = l2_gradient(ddf, tensor_weights)
        else:
            smoothness_loss = smoothness_lambda * l2_gradient(ddf)

        fdg_masks, psma_masks = get_mask_loss_inputs(
            fdg_mask,
            psma_mask,
            num_masks,
            device,
        )

        warped_moving_masks = torch.nn.functional.grid_sample(
            fdg_masks,
            grid,
            align_corners=True,
        )
        warped_moving_ct = torch.nn.functional.grid_sample(
            fdg_ct,
            grid,
            align_corners=True,
        )

        loss = (
            loss_function_dice(psma_masks, warped_moving_masks)
            + loss_function_dice(warped_moving_ct, psma_ct)
            + smoothness_loss
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_a += loss.item()
        step += 1.0

    return loss_a / step


def main(args):
    device = args.device
    input_keys = get_registration_input_keys(args.use_ct_input)
    if args.velocity_scale <= 0:
        raise ValueError("--velocity_scale must be > 0.")
    if args.int_steps < 0:
        raise ValueError("--int_steps must be >= 0.")

    model = build_baseline_model(
        args.baseline_model,
        in_channels=len(input_keys),
    ).to(device)

    train_transform = ReadH5d()

    train_list, test_list = split_multiple_train_test(
        [
            "/data2/xiangcen/data/pet_gen/processed/batch1_h5_v2",
            "/data2/xiangcen/data/pet_gen/processed/batch2_h5_v2",
            "/data2/xiangcen/data/pet_gen/processed/batch3_h5_v2",
        ],
        [40, 40, 20],
    )

    train_loader = create_data_loader(
        train_list,
        train_transform,
        batch_size=2,
    )
    print(test_list[:3])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    identity_grid = make_identity_grid_m11(
        (128, 128, 384),
        device=device,
    )

    save_path = get_baseline_save_path(args)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f">>> Baseline model = {args.baseline_model}")
    print(f">>> Model input = {list(input_keys)}")
    print(f">>> Smoothness lambda = {args.smoothness}")
    print(f">>> CT smoothness = {args.ct_smoothness}")
    print(f">>> CT smoothness margin = {args.ct_smoothness_margin}")
    print(f">>> CT smoothness gamma = {args.ct_smoothness_gamma}")
    print(f">>> Diffeomorphic = {args.diffeomorphic}")
    print(f">>> Velocity scale = {args.velocity_scale}")
    print(f">>> Integration steps = {args.int_steps}")
    print(f">>> Model will be saved to: {save_path}")

    for epoch in range(args.epochs):
        loss_batch = train_baseline_batch(
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
            diffeomorphic=args.diffeomorphic,
            velocity_scale=args.velocity_scale,
            int_steps=args.int_steps,
            device=device,
        )

        print(f"Epoch {epoch:03d} | Loss = {loss_batch:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"model saved at {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Baseline Registration Training")

    parser.add_argument(
        "--baseline_model",
        type=str,
        default="voxelmorph",
        choices=[
            "voxelmorph",
            "transmorph",
        ],
        help="Baseline model architecture to train.",
    )

    parser.add_argument(
        "--smoothness",
        type=float,
        default=8000,
        help="Smoothness regularization weight (lambda)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=350,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--num_masks",
        type=int,
        default=5,
        help="Number of sampled masks for weak supervision (0 = use all shared labels)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Training device",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=DEFAULT_SAVE_DIR,
        help="Directory where trained checkpoint will be saved.",
    )

    parser.add_argument(
        "--use_ct_input",
        action="store_true",
        help="Use [fdg_pt, fdg_ct, psma_pt, psma_ct] as model input.",
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
    parser.add_argument(
        "--diffeomorphic",
        action="store_true",
        help="Interpret model output as SVF and integrate it with scaling-and-squaring.",
    )
    parser.add_argument(
        "--velocity_scale",
        type=float,
        default=0.5,
        help="Scale applied to tanh(model_output) before SVF integration.",
    )
    parser.add_argument(
        "--int_steps",
        type=int,
        default=7,
        help="Number of scaling-and-squaring integration steps.",
    )

    args = parser.parse_args()
    main(args)
