import argparse
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from General.data_loader import create_data_loader, ReadH5d
from General.dataset_sample import split_multiple_train_test
from Registration.baseline_models import build_baseline_model
from Registration.mask import sample_labels_to_binary, sample_shared_binary_masks
from Registration.smoothness_losses import (
    BetaPriorLoss,
    l2_gradient,
    spatially_weighted_l2_gradient,
)
from Registration.training import make_identity_grid_m11, loss_function_dice


SPATIALLY_VARYING_MODELS = {
    "svr_diff",
    "spatially_varying_regularization",
}


def is_spatially_varying_model(model_name):
    return model_name.lower().replace("-", "_") in SPATIALLY_VARYING_MODELS


def get_baseline_save_path(args):
    mask_tag = "" if args.num_masks == 0 else f"_k{args.num_masks}"
    model_tag = args.baseline_model.lower().replace("-", "_")
    beta_tag = ""
    if model_tag in SPATIALLY_VARYING_MODELS:
        beta_tag = (
            f"_mar{int(args.smoothness_margin)}"
            f"_beta{args.beta_lambda:g}"
            f"_a{args.beta_alpha:g}"
            f"_b{args.beta_beta:g}"
            f"_{args.beta_prior_mode}"
        )
    return (
        "/data1/xiangcen/models/registration_v2/"
        f"{model_tag}_l{int(args.smoothness)}{mask_tag}{beta_tag}.ptm"
    )


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
    smoothness_margin=3000,
    beta_lambda=1.0,
    beta_prior_loss=None,
    num_masks=50,
    spatially_varying_regularization=False,
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

        model_input = torch.cat([fdg_pt, psma_pt], dim=1)

        if spatially_varying_regularization:
            outputs = model(fdg_pt, psma_pt)
            ddf = outputs["ddf"]
            regularization_map = outputs["regularization_map"]

            lambda_min = smoothness_lambda - smoothness_margin
            lambda_max = smoothness_lambda + smoothness_margin
            if lambda_min <= 0:
                raise ValueError(
                    "smoothness - smoothness_margin must be positive for "
                    "spatially varying regularization."
                )
            regularization_weight_map = (
                lambda_min
                + regularization_map * (lambda_max - lambda_min)
            )

            smoothness_loss = spatially_weighted_l2_gradient(
                ddf,
                regularization_weight_map,
            )
            beta_loss = beta_lambda * beta_prior_loss(regularization_map)
        else:
            ddf = torch.tanh(model(model_input))
            smoothness_loss = smoothness_lambda * l2_gradient(ddf)
            beta_loss = torch.zeros((), device=device)

        grid = identity_grid + ddf
        grid = grid.permute(0, 2, 3, 4, 1)

        fdg_masks, psma_masks = get_mask_loss_inputs(
            fdg_mask,
            psma_mask,
            num_masks,
            device,
        )

        warped_moving_masks = torch.nn.functional.grid_sample(fdg_masks, grid)
        warped_moving_ct = torch.nn.functional.grid_sample(fdg_ct, grid)

        loss = (
            loss_function_dice(psma_masks, warped_moving_masks)
            + loss_function_dice(warped_moving_ct, psma_ct)
            + smoothness_loss
            + beta_loss
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_a += loss.item()
        step += 1.0

    return loss_a / step


def main(args):
    device = args.device

    model = build_baseline_model(args.baseline_model).to(device)

    train_transform = ReadH5d()

    train_list, test_list = split_multiple_train_test(
        [
            "/data2/xiangcen/data/pet_gen/processed/batch1_h5_v2",
            "/data2/xiangcen/data/pet_gen/processed/batch2_h5_v2",
        ],
        [40, 40],
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
    spatially_varying_regularization = is_spatially_varying_model(
        args.baseline_model,
    )
    beta_prior_loss = BetaPriorLoss(
        alpha=args.beta_alpha,
        beta=args.beta_beta,
        mode=args.beta_prior_mode,
    )

    print(f">>> Baseline model = {args.baseline_model}")
    print(f">>> Smoothness lambda = {args.smoothness}")
    print(f">>> Smoothness margin = {args.smoothness_margin}")
    print(f">>> Beta lambda = {args.beta_lambda}")
    print(f">>> Beta prior = Beta({args.beta_alpha}, {args.beta_beta}), mode={args.beta_prior_mode}")
    print(f">>> Spatially varying regularization = {spatially_varying_regularization}")
    print(f">>> Model will be saved to: {save_path}")

    for epoch in range(args.epochs):
        loss_batch = train_baseline_batch(
            model,
            train_loader,
            optimizer,
            identity_grid,
            smoothness_lambda=args.smoothness,
            smoothness_margin=args.smoothness_margin,
            beta_lambda=args.beta_lambda,
            beta_prior_loss=beta_prior_loss,
            num_masks=args.num_masks,
            spatially_varying_regularization=spatially_varying_regularization,
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
            "vxm",
            "transmorph",
            "svr_diff",
            "spatially_varying_regularization",
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
        "--smoothness_margin",
        type=float,
        default=3000,
        help="Margin for SVR regularization weights: [lambda-margin, lambda+margin].",
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
        "--beta_lambda",
        type=float,
        default=1.0,
        help="Weight for Beta prior regularization on spatial weights.",
    )

    parser.add_argument(
        "--beta_alpha",
        type=float,
        default=1.1,
        help="Alpha parameter of the Beta prior.",
    )

    parser.add_argument(
        "--beta_beta",
        type=float,
        default=1.0,
        help="Beta parameter of the Beta prior.",
    )

    parser.add_argument(
        "--beta_prior_mode",
        type=str,
        default="full_beta",
        choices=["full_beta", "repo_logbeta"],
        help="Beta prior loss mode for spatial regularization weights.",
    )

    args = parser.parse_args()
    main(args)
