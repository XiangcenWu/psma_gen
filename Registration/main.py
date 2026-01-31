import sys
import os
import argparse
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monai.networks.nets import SwinUNETR
from General.data_loader import create_data_loader, ReadH5d
from monai.losses import DiceLoss
import torch.nn as nn

from General.dataset_sample import split_train_test
from Registration.training import train_batch, make_identity_grid_m11


def main(args):
    device = args.device

    model = SwinUNETR(
        in_channels=2,
        out_channels=3,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        downsample="mergingv2",
        use_v2=True,
    ).to(device)

    train_transform = ReadH5d()

    train_list, test_list = split_train_test(
        '/data1/xiangcen/data/pet_gen/processed/batch1_h5'
    )

    train_loader = create_data_loader(
        train_list, train_transform, batch_size=2
    )
    test_loader = create_data_loader(
        test_list, train_transform
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    identity_grid = make_identity_grid_m11(
        (128, 128, 384), device=device
    )

    if args.cross_modality_loss:
        save_path=f'/data1/xiangcen/models/registration/baseline_l{args.smoothness}_cml.ptm'
    else:
        save_path = f'/data1/xiangcen/models/registration/baseline_l{args.smoothness}.ptm'

    print(f'>>> Smoothness lambda = {args.smoothness}')
    print(f'>>> Model will be saved to: {save_path}')

    for epoch in range(args.epochs):
        loss_batch = train_batch(
            model,
            train_loader,
            optimizer,
            identity_grid,
            smoothness_lambda=args.smoothness,
            mask_per_iteration=args.mask_per_iteration,
            cross_modality_loss=args.cross_modality_loss
        )

        print(f'Epoch {epoch:03d} | Loss = {loss_batch:.6f}')

        torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Registration Training")

    parser.add_argument(
        "--smoothness",
        type=float,
        default=10.0,
        help="Smoothness regularization weight (lambda)"
    )

    parser.add_argument(
        "--mask_per_iteration",
        type=int,
        default=50,
        help="Number of masks sampled per iteration"
    )

    parser.add_argument(
        "--cross_modality_loss",
        action="store_true",
        help="Enable cross-modality loss"
    )


    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Training device"
    )

    args = parser.parse_args()
    main(args)
