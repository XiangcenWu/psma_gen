import torch


def sample_labels_to_binary(mask, num_samples, background=0):
    """
    Convert a multi-label mask into a binary mask
    by randomly sampling labels.

    Args:
        mask: tensor (H,W) or (D,H,W), integer labels
        num_samples: number of labels to sample
        background: background label value

    Returns:
        binary_mask: same shape as mask, values {0,1}
        selected_labels: tensor of sampled labels
    """
    labels = torch.unique(mask)
    labels = labels[labels != background]

    if labels.numel() == 0:
        return torch.zeros_like(mask), labels

    num_samples = min(num_samples, labels.numel())
    perm = torch.randperm(labels.numel(), device=labels.device)
    selected_labels = labels[perm[:num_samples]]

    binary_mask = torch.isin(mask, selected_labels)
    binary_mask = binary_mask.to(mask.dtype)

    return binary_mask



if __name__ == '__main__':

    mask = torch.randint(0, 128, (128, 128 , 128))
    binary_mask, chosen = sample_labels_to_binary(mask, num_samples=5)

    print("Chosen labels:", chosen)
    print("Binary mask unique values:", torch.unique(binary_mask))
