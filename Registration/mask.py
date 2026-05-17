import torch
from collections.abc import Sequence


def sample_labels_to_binary(mask):
    
    binary_mask = (mask != 0).to(mask.dtype)

    return binary_mask


def sample_shared_binary_masks(
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        num_samples: int,
        ignore_label: int = 0,
        device='cuda:0'
    ):
    """
    Args:
        moving_mask: Tensor of shape (N, B, X, Y, Z) or (N, X, Y, Z)
        fixed_mask:  Tensor of same shape as moving_mask
        num_samples: Number of shared classes to sample
        ignore_label: Label to ignore (usually background = 0)

    Returns:
        moving_bin: (N, K, X, Y, Z)
        fixed_bin:  (N, K, X, Y, Z)
    """

    assert moving_mask.shape == fixed_mask.shape, "Masks must have same shape"

    # If there's a channel dim B, squeeze it out
    if moving_mask.dim() == 5:
        moving_mask = moving_mask.squeeze(1)
        fixed_mask = fixed_mask.squeeze(1)

    # Get unique labels
    moving_labels = torch.unique(moving_mask)
    fixed_labels = torch.unique(fixed_mask)

    # Shared labels
    shared_labels = torch.tensor(
        list(set(moving_labels.tolist()) & set(fixed_labels.tolist())),
        device=moving_mask.device,
    )

    # Remove ignore label
    if ignore_label is not None:
        shared_labels = shared_labels[shared_labels != ignore_label]

    if len(shared_labels) == 0:
        raise ValueError("No shared labels found between moving and fixed masks.")

    # Sample labels
    num_samples = min(num_samples, len(shared_labels))
    perm = torch.randperm(len(shared_labels), device=moving_mask.device)
    sampled_labels = shared_labels[perm[:num_samples]]

    # Create binary masks
    moving_bin = []
    fixed_bin = []

    for label in sampled_labels:
        moving_bin.append((moving_mask == label).float())
        fixed_bin.append((fixed_mask == label).float())

    # Stack into channels
    moving_bin = torch.stack(moving_bin, dim=1)  # (N, K, X, Y, Z)
    fixed_bin = torch.stack(fixed_bin, dim=1)

    return moving_bin.to(device), fixed_bin.to(device)


def labels_to_binary_masks(
        mask: torch.Tensor,
        labels_from_prompts: Sequence[Sequence[int]],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
    """
    Convert prompt-selected labels into a multi-channel binary mask tensor.

    Args:
        mask: Tensor of shape (B, 1, H, W, D) or (B, H, W, D).
        labels_from_prompts: List of length B. Each item contains the labels
            sampled for that batch item.
        dtype: Output tensor dtype.

    Returns:
        Tensor of shape (B, K, H, W, D), where K is the maximum number of
        labels in the batch. Items with fewer than K labels are zero-padded.
    """
    if mask.dim() == 5:
        if mask.shape[1] != 1:
            raise ValueError("mask channel dimension must be 1 when mask is 5D.")
        mask = mask.squeeze(1)
    elif mask.dim() != 4:
        raise ValueError("mask must have shape (B, 1, H, W, D) or (B, H, W, D).")

    batch_size = mask.shape[0]
    if len(labels_from_prompts) != batch_size:
        raise ValueError("labels_from_prompts length must match mask batch size.")

    max_labels = max((len(labels) for labels in labels_from_prompts), default=0)
    if max_labels < 1:
        raise ValueError("labels_from_prompts must contain at least one label.")

    binary_masks = torch.zeros(
        (batch_size, max_labels, *mask.shape[1:]),
        device=mask.device,
        dtype=dtype,
    )

    for batch_idx, labels in enumerate(labels_from_prompts):
        for label_idx, label in enumerate(labels):
            binary_masks[batch_idx, label_idx] = (mask[batch_idx] == label).to(dtype)

    return binary_masks


if __name__ == '__main__':

    mask = torch.randint(0, 8, (2, 1, 16, 16, 16))
    binary_mask = labels_to_binary_masks(mask, labels_from_prompts=[[1, 2, 3], [4, 5, 6]])

    print("Binary mask shape:", binary_mask.shape)
    print("Binary mask unique values:", torch.unique(binary_mask))
