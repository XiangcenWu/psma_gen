import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Registration.mask import (
    labels_to_binary_masks,
    sample_labels_to_binary,
    sample_shared_binary_masks,
)
from Registration.smoothness_losses import l2_gradient


from monai.losses import DiceLoss

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



loss_function_dice = DiceLoss(
    to_onehot_y=False,
    softmax=False,
    include_background=True
)

DEFAULT_REGISTRATION_INPUT_KEYS = ("fdg_pt", "psma_pt")
CT_REGISTRATION_INPUT_KEYS = ("fdg_pt", "fdg_ct", "psma_pt", "psma_ct")


def get_registration_input_keys(use_ct_input=False):
    if use_ct_input:
        return CT_REGISTRATION_INPUT_KEYS
    return DEFAULT_REGISTRATION_INPUT_KEYS


def make_registration_input(batch, input_keys=DEFAULT_REGISTRATION_INPUT_KEYS, device="cuda:0"):
    return torch.cat([batch[key].to(device) for key in input_keys], dim=1)


def predict_ddf_and_grid(model, model_input, identity_grid, apply_tanh=True):
    """
    Run a registration model and build the grid_sample sampling grid.

    Returns:
        ddf: shape (B, ndim, *spatial_size), in normalized grid coordinates.
        grid: shape (B, *spatial_size, ndim), ready for torch.nn.functional.grid_sample.
    """
    ddf = model(model_input)
    if apply_tanh:
        ddf = torch.tanh(ddf)

    grid = identity_grid.to(device=ddf.device, dtype=ddf.dtype) + ddf
    grid = torch.movedim(grid, 1, -1)
    return ddf, grid




def get_save_path(args) -> str:
    mask_tag = "" if args.num_masks == 0 else f"k{args.num_masks}"
    input_tag = "_ctinput" if getattr(args, "use_ct_input", False) else ""

    save_path=f'/share/home/xcwu/registration_v3/baseline_l{int(args.smoothness)}{mask_tag}.ptm'
    if args.ct_smoothness:
        save_path=f'/share/home/xcwu/registration_v3/ctsmoothness_l{int(args.smoothness)}_{mask_tag}{input_tag}_mar{int(args.ct_smoothness_margin)}_gam{str(args.ct_smoothness_gamma)}.ptm'
    else:
        save_path=f'/share/home/xcwu/registration_v3/baseline_l{int(args.smoothness)}_{mask_tag}{input_tag}.ptm'


    return save_path


def get_ct_lambda(ct_img, margin, smoothness, gamma):


    ct_img = ct_img[:, :, 1:-1, 1:-1, 1:-1] # slice to match the gradient tensor
    ct_img = ct_img ** gamma
    _min, _max = smoothness-margin, smoothness+margin
    return _min + ct_img * (_max - _min)

