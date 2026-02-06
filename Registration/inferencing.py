import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np



from tqdm import tqdm
from General.segments import SEGMENT_INDEX

from monai.losses import GlobalMutualInformationLoss

def mutual_information(fixed, moving, num_bins=64):
    """
    Compute Mutual Information using MONAI.
    Inputs: tensors of same shape, any dimension (2D/3D), no batch dim needed.
    """
    # Add batch and channel dims if missing
    if fixed.ndim == 2:
        fixed = fixed.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
        moving = moving.unsqueeze(0).unsqueeze(0)
    elif fixed.ndim == 3:
        fixed = fixed.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W, D)
        moving = moving.unsqueeze(0).unsqueeze(0)
    elif fixed.ndim == 4:
        # Assume (H, W, D, C) → permute? Better to ensure (B, C, H, W, D)
        raise ValueError("Handle 4D as needed")

    mi_loss = GlobalMutualInformationLoss(
        num_bins=num_bins,
        kernel_type="gaussian",
        
    )
    neg_mi = mi_loss(moving, fixed)
    return -neg_mi

def get_binary_mask_with_label(mask: torch.tensor, label: int) -> torch.tensor:

    mask = (mask == label).to(mask.dtype)

    return mask


def summarize_statistics(list1, list2, percentile=95):
    """
    Compute mean, std, and percentile for two lists of numbers.

    Args:
        list1 (list or array): First list of values (e.g., before warp).
        list2 (list or array): Second list of values (e.g., after warp).
        percentile (float): Percentile to calculate (default 95).

    Returns:
        dict: A dictionary containing statistics for both lists.
    """
    stats = {}

    for name, lst in zip(["before", "after"], [list1, list2]):
        arr = np.array(lst)
        stats[name] = {
            "mean": np.mean(arr),
            "std": np.std(arr, ddof=1),  # use ddof=1 for sample std
            f"{percentile}th_percentile": np.percentile(arr, percentile)
        }

    return stats

@torch.no_grad()
def inference_batch(
        model,
        loader,
        identity_grid,
        masks_names, # list of names
        device="cuda:0"
    ):

    mask_list = []
    for name in masks_names:
        if name in SEGMENT_INDEX:
            mask_list.append(SEGMENT_INDEX[name])
        else:
            raise ValueError(f"Unknown segment names: {name}")


    
    model.eval()
    model.to(device)
    identity_grid.to(device)


    mi_before = []
    mi_after = []


    num_of_masks = len(masks_names)
    dice_before_lists = [[] for _ in range(num_of_masks)]
    dice_after_lists = [[] for _ in range(num_of_masks)]




    for i, batch in enumerate(tqdm(loader, desc="inferencing", total=len(loader))):
        assert batch["fdg_pt"].shape[0] == 1, \
                f"Expected batch size 1, got {batch['fdg_pt'].shape[0]}"

        fdg_pt = batch['fdg_pt'].to(device)
        fdg_mask = batch['fdg_mask'].to(device)
        fdg_spacing = batch['fdg_spacing']

        psma_pt = batch['psma_pt'].to(device)
        psma_mask = batch['psma_mask'].to(device)
        psma_spacing = batch['psma_spacing']

        mi_before.append(mutual_information(fdg_pt, psma_pt).item())
        

        input = torch.cat([fdg_pt, psma_pt], dim=1)

        # sample mask to be used to train loss
        # fdg_mask = sample_labels_to_binary(fdg_mask)
        # psma_mask = sample_labels_to_binary(psma_mask)



        ddf = model(input)
        ddf = torch.tanh(ddf)
        grid = identity_grid + ddf
        grid = grid.permute(0, 2, 3, 4, 1)

        # -----------------------------
        # get pet image's MI 
        # -----------------------------
        warped_fdg_pt = torch.nn.functional.grid_sample(fdg_pt, grid)

        mi_after.append(mutual_information(warped_fdg_pt, psma_pt).item())

        stats = summarize_statistics(mi_before, mi_after)
        print('This is stats for MI')
        print(stats)




