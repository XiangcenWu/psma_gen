import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np



from tqdm import tqdm
from General.segments import SEGMENT_INDEX

from monai.losses import GlobalMutualInformationLoss


def dice_metric(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred, target: same shape, binary (0/1) tensors
    """
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    intersection = (pred * target).sum()
    return (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)


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
            "mean": np.mean(arr).item(),
            "std": np.std(arr, ddof=1).item(),  # use ddof=1 for sample std
            f"{percentile}th_percentile": np.percentile(arr, percentile).item()
        }

    return stats



def dice_for_organs(moving: torch.Tensor, fixed: torch.Tensor, organ_names: list, eps: float = 1e-6) -> list:
    """
    Calculate Dice scores for a list of organs using dice_metric.

    Parameters:
        moving: tensor of moving segmentation (H,W,D) or (B,H,W,D)
        fixed: tensor of fixed segmentation (same shape as moving)
        organ_names: list of strings, organ names in SEGMENT_INDEX
        eps: small value to avoid division by zero

    Returns:
        List of Dice scores (float) in the same order as organ_names
    """
    dice_scores = []
    
    for organ_name in organ_names:
        if organ_name not in SEGMENT_INDEX:
            raise ValueError(f"Organ '{organ_name}' not in SEGMENT_INDEX")
        
        organ_label = SEGMENT_INDEX[organ_name]

        # Create binary masks for the organ
        moving_mask = (moving == organ_label).float()
        fixed_mask = (fixed == organ_label).float()

        # Compute Dice using dice_metric
        dice_score = dice_metric(moving_mask, fixed_mask, eps)
        dice_scores.append(dice_score.item())
    
    return dice_scores


@torch.no_grad()
def inference_batch(
        model,
        loader,
        identity_grid,
        masks_names=list(SEGMENT_INDEX.keys()), # list of names
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




        ddf = model(input)
        ddf = torch.tanh(ddf)
        grid = identity_grid + ddf
        grid = grid.permute(0, 2, 3, 4, 1)

        # -----------------------------
        # get pet image's MI 
        # -----------------------------
        warped_fdg_pt = torch.nn.functional.grid_sample(fdg_pt, grid)

        mi_after.append(mutual_information(warped_fdg_pt, psma_pt).cpu().item())
        
        for idx, names in enumerate(masks_names):
            mask_idx = mask_list[idx]
            binary_mask_fdg = get_binary_mask_with_label(fdg_mask, mask_idx)
            binary_mask_psma = get_binary_mask_with_label(psma_mask, mask_idx)

            dice_before_lists[idx].append(dice_metric(binary_mask_fdg, binary_mask_psma).cpu().item())
            warpped_fdg_mask = torch.nn.functional.grid_sample(binary_mask_fdg, grid)
            dice_after_lists[idx].append(dice_metric(warpped_fdg_mask, binary_mask_psma).cpu().item())


    stats = summarize_statistics(mi_before, mi_after)
    print('This is stats for MI')
    print(stats)

    for idx, names in enumerate(masks_names):
        print(names)
        stats = summarize_statistics(dice_before_lists[idx], dice_after_lists[idx])
        print(stats)




