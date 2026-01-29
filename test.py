import torch

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


print(make_identity_grid_m11((2, 3, 5))[0, 0])