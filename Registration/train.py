import sys
import os
import argparse
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monai.networks.nets import SwinUNETR
from General.data_loader import create_data_loader, ReadH5d


from General.dataset_sample import split_multiple_train_test
from Registration.training import train_batch, make_identity_grid_m11, get_save_path


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

    train_list, test_list = split_multiple_train_test(
        ['/data2/xiangcen/data/pet_gen/processed/batch1_h5_v2','/data2/xiangcen/data/pet_gen/processed/batch2_h5_v2'],
        [35, 35]
    )

    train_loader = create_data_loader(
        train_list, train_transform, batch_size=2
    )
    print(test_list[:3]) 

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    identity_grid = make_identity_grid_m11(
        (128, 128, 384), device=device
    )

    save_path = get_save_path(args)


    print(f'>>> Smoothness lambda = {args.smoothness}')
    print(f'>>> Model will be saved to: {save_path}')

    for epoch in range(args.epochs):
        loss_batch = train_batch(
            model,
            train_loader,
            optimizer,
            identity_grid,
            smoothness_lambda=args.smoothness,
            ct_smoothness = args.ct_smoothness,
            ct_smoothness_margin = args.ct_smoothness_margin,
            ct_smoothness_gamma = args.ct_smoothness_gamma,
            num_masks=args.num_masks,
        )

        print(f'Epoch {epoch:03d} | Loss = {loss_batch:.6f}')

    torch.save(model.state_dict(), save_path)
    print(f'model saved at {save_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Registration Training")

    parser.add_argument(
        "--smoothness",
        type=float,
        default=8000,
        help="Smoothness regularization weight (lambda)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=350,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--num_masks",
        type=int,
        default=5,
        help="Number of sampled masks for weak supervision (0 = use all shared labels)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Training device"
    )
    
    parser.add_argument(
        "--ct_smoothness",
        action="store_true",
        help="Enable CT as ddf smoothness regularization (default: False)"
    )

    parser.add_argument(
        "--ct_smoothness_margin",
        type=float,
        default=3000.0,
        help="Margin value for CT smoothness regularization (default: 3000)"
    )
    parser.add_argument(
        "--ct_smoothness_gamma",
        type=float,
        default=1.0,
        help="gamma value for CT smoothness regularization (default: 3000)"
    )


    args = parser.parse_args()
    main(args)
