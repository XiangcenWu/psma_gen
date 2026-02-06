import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np



from tqdm import tqdm
from General.segments import SEGMENT_INDEX

def mutual_information(img_fixed, img_moving, num_bins=64):
    """
    Compute Mutual Information (MI) between two images using PyTorch.
    """
    # Ensure inputs are tensors and flattened
    img_fixed = torch.as_tensor(img_fixed).flatten().float()
    img_moving = torch.as_tensor(img_moving).flatten().float()
    # 1. Calculate the min/max for binning
    # Note: For medical imaging, you might want to use fixed ranges (e.g., 0-255)
    f_min, f_max = img_fixed.min(), img_fixed.max()
    m_min, m_max = img_moving.min(), img_moving.max()
    # 2. Compute Joint Histogram
    # We stack them to create a (N, 2) input for histogramdd
    sample = torch.stack([img_fixed, img_moving], dim=1)
    # Define the range for both dimensions
    hist_range = [f_min, f_max, m_min, m_max]
    joint_hist = torch.histogramdd(
        sample, 
        bins=num_bins, 
        range=hist_range
    ).hist
    # 3. Convert to probability distribution
    joint_prob = joint_hist / joint_hist.sum()
    # 4. Marginal probabilities
    prob_fixed = joint_prob.sum(dim=1)
    prob_moving = joint_prob.sum(dim=0)
    # 5. Mutual Information calculation
    # We use a small epsilon to avoid log(0) and division by zero
    eps = torch.finfo(joint_prob.dtype).eps
    # Outer product of marginals: p(x) * p(y)
    marginals_prod = torch.outer(prob_fixed, prob_moving)
    # MI = sum( P(x,y) * log( P(x,y) / (P(x)*P(y)) ) )
    # Only calculate where joint_prob > 0
    mask = joint_prob > 0
    mi = torch.sum(
        joint_prob[mask] * torch.log(joint_prob[mask] / (marginals_prod[mask] + eps))
    )
    return mi

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
        maks_names, # list of names
        device="cuda:0"
    ):

    mask_list = []
    for name in maks_names:
        if name in SEGMENT_INDEX:
            mask_list.append(SEGMENT_INDEX[name])
        else:
            raise ValueError(f"Unknown segment names: {name}")


    
    model.eval()
    model.to(device)
    identity_grid.to(device)


    mi_before = []
    mi_after = []


    num_of_masks = len(names)
    dice_before_lists = [[] for _ in range(num_masks)]
    dice_after_lists = [[] for _ in range(num_masks)]




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




