import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Registration.mask import sample_labels_to_binary, sample_shared_binary_masks
from Registration.smoothness_losses import l2_gradient

from monai.losses import DiceLoss

def make_identity_grid_m11(spatial_size, device=None, dtype=torch.float32):
    """
    Create an identity grid normalized to [-1, 1] for grid_sample
    (align_corners=True).

    Args:
        spatial_size: tuple like (D, H, W) or (H, W)

    Returns:
        grid: shape (1, ndim, *spatial_size)
              order: (x, y, z, ...)
    """
    coords = torch.meshgrid(
        *[
            torch.linspace(-1.0, 1.0, s, device=device, dtype=dtype)
            if s > 1 else torch.zeros(1, device=device, dtype=dtype)
            for s in spatial_size
        ],
        indexing="ij"
    )

    # (z, y, x) → (x, y, z)
    grid = torch.stack(coords[::-1], dim=0)

    return grid.unsqueeze(0)



loss_function_dice = DiceLoss(
    to_onehot_y=False,
    softmax=False,
    include_background=True
)
loss_function_mse = torch.nn.MSELoss()

def train_batch(
        model, 
        loader,
        optimizer,
        identity_grid,
        smoothness_lambda=1000,
        ct_smoothness = False,
        ct_smoothness_margin = 3000,
        cross_modality_loss='mse',
        num_masks=50,
        device="cuda:0"
    ):
    
    model.train()
    model.to(device)
    identity_grid.to(device)

    step = 0.
    loss_a = 0.


    for batch in loader:
        # Option A: Manual unpacking to specific variable names
        if cross_modality_loss or ct_smoothness:
            fdg_ct = batch['fdg_ct'].to(device)
            psma_ct = batch['psma_ct'].to(device)


        fdg_pt = batch['fdg_pt'].to(device)
        fdg_mask = batch['fdg_mask'].to(device)

        
        
        psma_pt = batch['psma_pt'].to(device)
        psma_mask = batch['psma_mask'].to(device)


        input = torch.cat([fdg_pt, psma_pt], dim=1)





        ddf = model(input)
        ddf = torch.tanh(ddf)

        #claculate smoothness loss first hand 
        if ct_smoothness:
            smoothness_loss = get_ct_lambda(fdg_ct, ct_smoothness_margin, smoothness_lambda)*l2_gradient(ddf)
        else:
            smoothness_loss = smoothness_lambda*l2_gradient(ddf)

        grid = identity_grid + ddf
        grid = grid.permute(0, 2, 3, 4, 1)

        if num_masks != 0:
            fdg_masks, psma_masks = sample_shared_binary_masks(
                moving_mask = fdg_mask,
                fixed_mask=psma_mask,
                num_samples=num_masks
            )
        warped_moving_masks = torch.nn.functional.grid_sample(fdg_masks, grid)

        if cross_modality_loss == 'mse':
            warped_moving_ct = torch.nn.functional.grid_sample(fdg_ct, grid)
            loss = loss_function_dice(psma_masks, warped_moving_masks) + \
                loss_function_mse(warped_moving_ct, psma_ct) + smoothness_loss
        elif cross_modality_loss == 'dice':
            warped_moving_ct = torch.nn.functional.grid_sample(fdg_ct, grid)
            loss = loss_function_dice(psma_masks, warped_moving_masks) + \
                loss_function_dice(warped_moving_ct, psma_ct) + smoothness_loss
        elif cross_modality_loss == None and ct_smoothness:
            warped_moving_ct = torch.nn.functional.grid_sample(fdg_ct, grid)
            loss = loss_function_dice(psma_masks, warped_moving_masks) + \
                loss_function_dice(warped_moving_ct, psma_ct) + smoothness_loss
        else:
            loss = loss_function_dice(psma_masks, warped_moving_masks) + smoothness_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_a += loss.item()
        step += 1.
    loss_of_this_epoch = loss_a / step

    return loss_of_this_epoch



def get_save_path(args) -> str:
    mask_tag = "" if args.num_masks == 0 else f"_k{args.num_masks}"

    if args.cross_modality_loss == 'dice':
        save_path=f'/data1/xiangcen/models/registration/baseline_l{int(args.smoothness)}{mask_tag}_cmldice.ptm'
        if args.ct_smoothness:
            save_path=\
                f'/data1/xiangcen/models/registration/\
                    ctsmoothness_l{int(args.smoothness)}{mask_tag}_cmldice_{args.smoothness_margin}.ptm'

    elif args.cross_modality_loss == 'mse':
        save_path=f'/data1/xiangcen/models/registration/baseline_l{int(args.smoothness)}{mask_tag}_cmlmse.ptm'
        if args.ct_smoothness:
            save_path=\
                f'/data1/xiangcen/models/registration/\
                    ctsmoothness_l{int(args.smoothness)}{mask_tag}_cmlmse_{args.smoothness_margin}.ptm'

    else:
        save_path = f'/data1/xiangcen/models/registration/baseline_l{int(args.smoothness)}{mask_tag}.ptm'
    return save_path


def get_ct_lambda(ct_img, margin, smoothness):
    _min, _max = smoothness-margin, smoothness+margin
    return _min + ct_img * (_max - _min)

