import torch


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


if __name__ == '__main__':

    mask = torch.randint(0, 128, (128, 128 , 128))
    binary_mask, chosen = sample_labels_to_binary(mask, num_samples=5)

    print("Chosen labels:", chosen)
    print("Binary mask unique values:", torch.unique(binary_mask))
