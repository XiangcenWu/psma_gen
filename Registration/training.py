import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Registration.mask import (
    prompt_labels_to_binary_masks,
    sample_labels_to_binary,
    sample_shared_binary_masks,
)
from Registration.smoothness_losses import l2_gradient
from llm_Registration.prompt.read_basic_prompt import read_basic_prompt

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


def train_batch(
        model, 
        loader,
        optimizer,
        identity_grid,
        smoothness_lambda=1000,
        ct_smoothness = False,
        ct_smoothness_margin = 3000,
        ct_smoothness_gamma = 1,
        num_masks=50,
        device="cuda:0"
    ):
    
    model.train()
    model.to(device)
    identity_grid.to(device)

    step = 0.
    loss_a = 0.


    for batch in loader:

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
            tensor_weights = get_ct_lambda(fdg_ct, ct_smoothness_margin, smoothness_lambda, ct_smoothness_gamma)
            smoothness_loss = l2_gradient(ddf, tensor_weights)
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


        warped_moving_ct = torch.nn.functional.grid_sample(fdg_ct, grid)
        loss = loss_function_dice(psma_masks, warped_moving_masks) + \
            loss_function_dice(warped_moving_ct, psma_ct) + smoothness_loss


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_a += loss.item()
        step += 1.
    loss_of_this_epoch = loss_a / step

    return loss_of_this_epoch



def get_save_path(args) -> str:
    mask_tag = "" if args.num_masks == 0 else f"k{args.num_masks}"

    save_path=f'/data1/xiangcen/models/registration_v2/baseline_l{int(args.smoothness)}{mask_tag}.ptm'
    if args.ct_smoothness:
        save_path=f'/data1/xiangcen/models/registration_v2/ctsmoothness_l{int(args.smoothness)}_{mask_tag}_mar{int(args.ct_smoothness_margin)}_gam{str(args.ct_smoothness_gamma)}.ptm'
    else:
        save_path=f'/data1/xiangcen/models/registration_v2/baseline_l{int(args.smoothness)}_{mask_tag}.ptm'


    return save_path


def get_ct_lambda(ct_img, margin, smoothness, gamma):


    ct_img = ct_img[:, :, 1:-1, 1:-1, 1:-1] # slice to match the gradient tensor
    ct_img = ct_img ** gamma
    _min, _max = smoothness-margin, smoothness+margin
    return _min + ct_img * (_max - _min)



def train_batch_llm(
    model,
    loader,
    optimizer,
    identity_grid,
    max_prompt_organs=5,
    device="cuda:0",
    zero_ddf = False,
    log_loss_in_fdg_masks=True,
    fixed_prompt_pairs=None,
):
    model.train()
    identity_grid = identity_grid.to(device)

    step = 0.0
    loss_a = 0.0

    

    for batch in loader:
        # data loading
        fdg_ct = batch["fdg_ct"].to(device)
        psma_ct = batch["psma_ct"].to(device)

        fdg_pt = batch["fdg_pt"].to(device)
        fdg_mask = batch["fdg_mask"].to(device)

        psma_pt = batch["psma_pt"].to(device)
        psma_mask = batch["psma_mask"].to(device)

        
        if fixed_prompt_pairs:
            prompt_indices = torch.randint(0, len(fixed_prompt_pairs), (fdg_pt.shape[0],)).tolist()
            prompt_pairs = [fixed_prompt_pairs[idx] for idx in prompt_indices]
        else:
            prompt_pairs = [
                read_basic_prompt(organs=torch.randint(1, max_prompt_organs + 1, (1,)).item())
                for _ in range(fdg_pt.shape[0])
            ]
        prompts = [prompt for prompt, _ in prompt_pairs]
        labels_from_prompts = [labels for _, labels in prompt_pairs]

        # input of the model
        moving_input = torch.cat([fdg_pt, fdg_ct], dim=1)
        fixed_input = torch.cat([psma_pt, psma_ct], dim=1)

        # generate ddf from model
        model_outputs = model(
            moving=moving_input,
            fixed=fixed_input,
            texts=prompts,
        )
        ddf = model_outputs["ddf"]
        ddf = torch.tanh(ddf)
        if zero_ddf:
            ddf_loss = ddf.pow(2).mean()
            return 0.
        else:
            grid = identity_grid + ddf
            grid = grid.permute(0, 2, 3, 4, 1)


            # later
            spatial_regularization_map = model_outputs["spatial_regularization_map"]
            # get masks from labels_from_prompts
            fdg_masks, psma_masks = prompt_labels_to_binary_masks(
                moving_mask = fdg_mask,
                fixed_mask=psma_mask,
                labels_from_prompts=labels_from_prompts,
            )
            log_loss_map = -torch.log(spatial_regularization_map.clamp(min=1e-4, max=1.0))
            if log_loss_in_fdg_masks:
                fdg_log_mask = ((fdg_masks > 0).any(dim=1, keepdim=True) | (psma_masks > 0).any(dim=1, keepdim=True)).to(
                    dtype=spatial_regularization_map.dtype,
                    device=spatial_regularization_map.device,
                )
                log_loss = 0.8 * (log_loss_map * fdg_log_mask).sum() / fdg_log_mask.sum().clamp_min(1.0) / 9.210340371976184
            else:
                log_loss = 0.8 * log_loss_map.mean() / 9.210340371976184
            # get smoothness loss
            spatial_regularization_map = spatial_regularization_map[:, :, 1:-1, 1:-1, 1:-1]
            smoothness_loss = 8000*l2_gradient(ddf, spatial_regularization_map)
            # warp masks from labels_from_prompts
            warped_moving_masks = torch.nn.functional.grid_sample(fdg_masks, grid)
            # calculate the final loss
            loss = loss_function_dice(psma_masks, warped_moving_masks) + smoothness_loss + log_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_a += loss.item()
            step += 1.0

        return loss_a / step
