import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from Registration.mask import sample_labels_to_binary
from monai.networks.nets import SwinUNETR
from General.data_loader import create_data_loader, ReadH5d
from General.dataset_sample import split_train_test
from Registration.training import train_batch, make_identity_grid_m11

import SimpleITK as sitk
import argparse
from tqdm import tqdm


def tensor_to_itk(tensor, spacing=None):
    """
    tensor: (1, 1, X, Y, Z) 
    """

    tensor = tensor.cpu()[0, 0]

    array = tensor.numpy().transpose(2, 1, 0)
    itk_img = sitk.GetImageFromArray(array)

    if spacing is not None:
        itk_img.SetSpacing(spacing)


    return itk_img


@torch.no_grad()
def generate_warp(
        model, 
        loader,
        identity_grid,
        result_dir,
        device="cuda:0"
    ):


    
    model.eval()
    model.to(device)
    identity_grid.to(device)

    os.makedirs(result_dir, exist_ok=True)


    for i, batch in enumerate(tqdm(loader, desc="Generating warps", total=len(loader))):
        assert batch["fdg_pt"].shape[0] == 1, \
                f"Expected batch size 1, got {batch['fdg_pt'].shape[0]}"

        fdg_pt = batch['fdg_pt'].to(device)
        fdg_mask = batch['fdg_mask'].to(device)
        fdg_spacing = batch['fdg_spacing']

        psma_pt = batch['psma_pt'].to(device)
        psma_mask = batch['psma_mask'].to(device)
        psma_spacing = batch['psma_spacing']
        

        input = torch.cat([fdg_pt, psma_pt], dim=1)

        # sample mask to be used to train loss
        fdg_mask = sample_labels_to_binary(fdg_mask)
        psma_mask = sample_labels_to_binary(psma_mask)



        ddf = model(input)
        ddf = torch.tanh(ddf)
        grid = identity_grid + ddf
        grid = grid.permute(0, 2, 3, 4, 1)

        # -----------------------------
        # warp moving images
        # -----------------------------
        warped_fdg_pt = torch.nn.functional.grid_sample(fdg_pt, grid)
        warped_fdg_mask = torch.nn.functional.grid_sample(fdg_mask, grid)


        sample_dir = os.path.join(result_dir, f"sample_{i:04d}")
        os.makedirs(sample_dir, exist_ok=True)


        psma_pt_itk = tensor_to_itk(psma_pt, [t.item() for t in psma_spacing])
        psma_mask_itk = tensor_to_itk(psma_mask, [t.item() for t in psma_spacing])
        fdg_pt_warped_itk = tensor_to_itk(warped_fdg_pt, [t.item() for t in fdg_spacing])
        fdg_mask_warped_itk = tensor_to_itk(warped_fdg_mask, [t.item() for t in fdg_spacing])


        sitk.WriteImage(psma_pt_itk, os.path.join(sample_dir, "psma_pt.nii.gz"))
        sitk.WriteImage(psma_mask_itk, os.path.join(sample_dir, "psma_ct_mask.nii.gz"))
        sitk.WriteImage(fdg_pt_warped_itk, os.path.join(sample_dir, "fdg_pt_warped.nii.gz"))
        sitk.WriteImage(fdg_mask_warped_itk, os.path.join(sample_dir, "fdg_mask_warped.nii.gz"))


def main(args):
    device = args.device

    model = SwinUNETR(
        in_channels=2,
        out_channels=3,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        downsample="mergingv2",
        use_v2=True,
    )

    model.load_state_dict(torch.load(args.weights_path, map_location=device))

    train_transform = ReadH5d()

    train_list, test_list = split_train_test(
        '/data1/xiangcen/data/pet_gen/processed/batch1_h5'
    )


    test_loader = create_data_loader(
        test_list, train_transform, shuffle=False
    )

    identity_grid = make_identity_grid_m11(
        (128, 128, 384), device=device
    )


    generate_warp(model,
                  test_loader,
                  identity_grid,
                  args.result_dir,
                  device
                  )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate deformation fields using SwinUNETR"
    )

    parser.add_argument(
        "--weights_path",
        type=str,
        required=True,
        help="Path to the trained SwinUNETR model weights (.pth)"
    )

    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Directory to save generated warp results"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run on (e.g., cuda, cuda:0, cpu)"
    )



    args = parser.parse_args()

    main(args)
