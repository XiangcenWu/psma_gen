import torch


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Registration.mask import sample_labels_to_binary
from Registration.smoothness_losses import l2_gradient


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


def train_batch(
        model, 
        loader,
        optimizer,
        loss_function,
        identity_grid,
        smoothness_lambda=1000,
        mask_per_iteration=30,
        cross_modality_loss=False,
        device="cuda:0"
    ):
    
    model.train()
    model.to(device)
    identity_grid.to(device)

    step = 0.
    loss_a = 0.


    for batch in dataloader:
        # Option A: Manual unpacking to specific variable names
        if cross_modality_loss:
            fdg_ct = batch['fdg_ct'].to(device)
            psma_ct = batch['psma_ct'].to(device)


        fdg_pt = batch['fdg_pt'].to(device)
        fdg_mask = batch['fdg_mask'].to(device)

        
        
        psma_pt = batch['psma_pt'].to(device)
        psma_mask = batch['psma_mask'].to(device)


        input = torch.cat([fdg_pt, psma_pt], dim=1)

        # sample mask to be used to train loss
        fdg_mask = sample_labels_to_binary(fdg_mask, mask_per_iteration)
        psma_mask = sample_labels_to_binary(psma_mask, mask_per_iteration)



        ddf = model(input)
        ddf = torch.tanh(ddf)
        grid = identity_grid + ddf

        warped_moving = torch.nn.functional.grid_sample(fdg_mask, grid)

        
        loss = loss_function(psma_mask, fdg_mask) + smoothness_lambda*l2_gradient(ddf)


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_a += loss.item()
        step += 1.
    loss_of_this_epoch = loss_a / step

    return loss_of_this_epoch