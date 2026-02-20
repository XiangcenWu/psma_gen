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



def compute_tre_single(moving, fixed, voxel_spacing, eps=1e-8):
    """
    Compute TRE between one moving and one fixed 3D segmentation.
 
    Args:
        moving: (1, 1, X, Y, Z) probabilistic segmentation (0–1)
        fixed:  (1, 1, X, Y, Z) binary segmentation (0 or 1)
        voxel_spacing: list or tensor [sx, sy, sz]
        eps: numerical stability
 
    Returns:
        tre: scalar (physical units, e.g., mm)
             returns NaN if fixed segmentation is empty
    """
 
    device = moving.device
    _, _, X, Y, Z = moving.shape
 
    # Remove batch and channel dims → (X, Y, Z)
    moving = moving[0, 0]
    fixed = fixed[0, 0]
 
    # ---- Check empty fixed segmentation ----
    fixed_sum = fixed.sum()
    moving_sum = moving.sum()
    if fixed_sum < 1 or moving_sum < 1:
        return torch.tensor(float(0), device=device)
 
    # Create coordinate grid
    xs = torch.arange(X, device=device)
    ys = torch.arange(Y, device=device)
    zs = torch.arange(Z, device=device)
 
    grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
 
    # ---- Moving centroid (soft) ----
    moving_sum = moving.sum() + eps
 
    moving_centroid_x = (moving * grid_x).sum() / moving_sum
    moving_centroid_y = (moving * grid_y).sum() / moving_sum
    moving_centroid_z = (moving * grid_z).sum() / moving_sum
 
    moving_centroid = torch.stack(
        [moving_centroid_x, moving_centroid_y, moving_centroid_z]
    )
 
    # ---- Fixed centroid (binary) ----
    fixed_centroid_x = (fixed * grid_x).sum() / (fixed_sum + eps)
    fixed_centroid_y = (fixed * grid_y).sum() / (fixed_sum + eps)
    fixed_centroid_z = (fixed * grid_z).sum() / (fixed_sum + eps)
 
    fixed_centroid = torch.stack(
        [fixed_centroid_x, fixed_centroid_y, fixed_centroid_z]
    )
 
    # ---- Convert to physical space ----
    voxel_spacing = torch.tensor(voxel_spacing, device=device).float()
    diff = (moving_centroid - fixed_centroid) * voxel_spacing
 
    tre = torch.norm(diff, p=2)
 
    return tre

def tre_surface_from_masks(fixed_mask, moving_mask, spacing):
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


    def extract_surface(mask):
        kernel = torch.ones((3, 3, 3), device=device)
        kernel[1, 1, 1] = 0

        neigh = F.conv3d(mask[None, None], kernel[None, None], padding=1)
        surface = mask * (neigh < 26)

        # keep only z,y,x
        coords = surface.nonzero(as_tuple=False)[:, 2:].float()
        return coords

    fixed_surf = extract_surface(fixed_mask)
    moving_surf = extract_surface(moving_mask)

    if fixed_surf.numel() == 0 or moving_surf.numel() == 0:
        return torch.tensor(float(0))

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
            tre_before_lists[idx].append(compute_tre_single(binary_mask_fdg, binary_mask_psma, spacing).cpu().item())

            warpped_fdg_mask = torch.nn.functional.grid_sample(binary_mask_fdg, grid)

            dice_after_lists[idx].append(dice_metric(warpped_fdg_mask, binary_mask_psma).cpu().item())
            tre_after_lists[idx].append(compute_tre_single(warpped_fdg_mask, binary_mask_psma, spacing).cpu().item())


    save_registration_results(filename, masks_names, dice_before_lists, dice_after_lists, tre_before_lists, tre_after_lists)
    # stats = summarize_statistics(mi_before, mi_after)
    # print('This is stats for MI')
    # print(stats)
