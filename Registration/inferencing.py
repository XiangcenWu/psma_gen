import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
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



import torch
import torch.nn.functional as F


import torch
import torch.nn.functional as F


def tre_surface_from_masks(fixed_mask, moving_mask, spacing, threshold=0.5):
    """
    Compute symmetric TRE between two 3D masks using surface distances.

    Args:
        fixed_mask (torch.Tensor): (B, N, D,H,W) binary or probabilistic
        moving_mask (torch.Tensor): (B, N, D,H,W) binary or probabilistic
        spacing (sequence): voxel spacing (z,y,x)
        threshold (float): binarization threshold

    Returns:
        float: TRE in physical units
    """
    fixed_mask = fixed_mask.squeeze(0).squeeze(0)
    moving_mask = moving_mask.squeeze(0).squeeze(0)
    
    device = fixed_mask.device
    spacing = torch.as_tensor(spacing, device=device, dtype=torch.float32)

    fixed_bin = (fixed_mask >= threshold).float()
    moving_bin = (moving_mask >= threshold).float()

    def extract_surface(mask):
        kernel = torch.ones((3, 3, 3), device=device)
        kernel[1, 1, 1] = 0

        neigh = F.conv3d(mask[None, None], kernel[None, None], padding=1)
        surface = mask * (neigh < 26)

        # keep only z,y,x
        coords = surface.nonzero(as_tuple=False)[:, 2:].float()
        return coords

    fixed_surf = extract_surface(fixed_bin)
    moving_surf = extract_surface(moving_bin)

    if fixed_surf.numel() == 0 or moving_surf.numel() == 0:
        return float("nan")

    # voxel -> physical
    fixed_phys = fixed_surf * spacing
    moving_phys = moving_surf * spacing

    # nearest neighbor distance
    diff = fixed_phys[:, None, :] - moving_phys[None, :, :]
    dist = torch.norm(diff, dim=2)
    min_dist = dist.min(dim=1)[0]

    return min_dist.mean().item()






# def dice_for_organs(moving: torch.Tensor, fixed: torch.Tensor, organ_names: list, eps: float = 1e-6) -> list:
#     """
#     Calculate Dice scores for a list of organs using dice_metric.

#     Parameters:
#         moving: tensor of moving segmentation (H,W,D) or (B,H,W,D)
#         fixed: tensor of fixed segmentation (same shape as moving)
#         organ_names: list of strings, organ names in SEGMENT_INDEX
#         eps: small value to avoid division by zero

#     Returns:
#         List of Dice scores (float) in the same order as organ_names
#     """
#     dice_scores = []
    
#     for organ_name in organ_names:
#         if organ_name not in SEGMENT_INDEX:
#             raise ValueError(f"Organ '{organ_name}' not in SEGMENT_INDEX")
        
#         organ_label = SEGMENT_INDEX[organ_name]

#         # Create binary masks for the organ
#         moving_mask = (moving == organ_label).float()
#         fixed_mask = (fixed == organ_label).float()

#         # Compute Dice using dice_metric
#         dice_score = dice_metric(moving_mask, fixed_mask, eps)
#         dice_scores.append(dice_score.item())
    
#     return dice_scores


def save_registration_results(
    filename, masks_names, dice_before_lists, dice_after_lists, tre_before_lists, tre_after_lists
):
    """
    Save registration results to a text file.

    Parameters:
    - filename: str, path to save the text file
    - masks_names: list of mask names
    - dice_before_lists: list of lists of dice scores before registration
    - dice_after_lists: list of lists of dice scores after registration
    - tre_before_lists: list of lists of TRE before registration
    - tre_after_lists: list of lists of TRE after registration
    """
    
    # Helper function to convert list of values to a string joined by ";"
    def list_to_string(lst):
        return ";".join(str(x) for x in lst)
    
    with open(filename, "w") as f:
        # Write header (mask names)
        f.write(list_to_string(masks_names) + "\n")
        
        # Write dice before
        f.write(list_to_string(dice_before_lists) + "\n")
        
        # Write dice after
        f.write(list_to_string(dice_after_lists) + "\n")
        
        # Write tre before
        f.write(list_to_string(tre_before_lists) + "\n")
        
        # Write tre after
        f.write(list_to_string(tre_after_lists) + "\n")


@torch.no_grad()
def inference_batch(
        model,
        loader,
        identity_grid,
        filename,
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

    # mi_before = []
    # mi_after = []

    num_of_masks = len(masks_names)
    dice_before_lists = [[] for _ in range(num_of_masks)]
    dice_after_lists = [[] for _ in range(num_of_masks)]

    tre_before_lists = [[] for _ in range(num_of_masks)]
    tre_after_lists = [[] for _ in range(num_of_masks)]

    for i, batch in enumerate(tqdm(loader, desc="inferencing", total=len(loader))):
        assert batch["fdg_pt"].shape[0] == 1, \
                f"Expected batch size 1, got {batch['fdg_pt'].shape[0]}"

        fdg_pt = batch['fdg_pt'].to(device)
        fdg_mask = batch['fdg_mask'].to(device)
        fdg_spacing = batch['fdg_spacing']
        fdg_spacing = [t.item() for t in fdg_spacing]

        psma_pt = batch['psma_pt'].to(device)
        psma_mask = batch['psma_mask'].to(device)
        psma_spacing = batch['psma_spacing']
        psma_spacing = [t.item() for t in psma_spacing]

        spacing = (torch.tensor(fdg_spacing) + torch.tensor(psma_spacing)) / 2

        # mi_before.append(mutual_information(fdg_pt, psma_pt).item())
        

        _input = torch.cat([fdg_pt, psma_pt], dim=1)
        ddf = model(_input)
        ddf = torch.tanh(ddf)
        grid = identity_grid + ddf
        grid = grid.permute(0, 2, 3, 4, 1)

        # warped_fdg_pt = torch.nn.functional.grid_sample(fdg_pt, grid)
        # mi_after.append(mutual_information(warped_fdg_pt, psma_pt).cpu().item())
        
        for idx, names in enumerate(masks_names):
            mask_idx = mask_list[idx]
            binary_mask_fdg = get_binary_mask_with_label(fdg_mask, mask_idx)
            binary_mask_psma = get_binary_mask_with_label(psma_mask, mask_idx)

            dice_before_lists[idx].append(dice_metric(binary_mask_fdg, binary_mask_psma).cpu().item())
            tre_before_lists[idx].append(tre_surface_from_masks(binary_mask_fdg, binary_mask_psma, spacing).cpu().item())

            warpped_fdg_mask = torch.nn.functional.grid_sample(binary_mask_fdg, grid)

            dice_after_lists[idx].append(dice_metric(warpped_fdg_mask, binary_mask_psma).cpu().item())
            tre_after_lists[idx].append(tre_surface_from_masks(warpped_fdg_mask, binary_mask_psma, spacing).cpu().item())


    save_registration_results(filename, masks_names, dice_before_lists, dice_after_lists, tre_before_lists, tre_after_lists)
    # stats = summarize_statistics(mi_before, mi_after)
    # print('This is stats for MI')
    # print(stats)
