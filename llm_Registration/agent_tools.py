import os
import sys
from typing import Dict, List, Optional, Sequence

import torch
from monai.networks.nets import SwinUNETR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from General.data_loader import ReadH5d
from Registration.mask import labels_to_binary_masks
from Registration.smoothness_losses import l2_gradient
from Registration.training import (
    loss_function_dice,
    make_identity_grid_m11,
    predict_ddf_and_grid,
)
from llm_Registration.config import DEVICE, REGISTRATION_WEIGHTS_PATH, SPATIAL_SIZE


def build_registration_model():
    return SwinUNETR(
        in_channels=2,
        out_channels=3,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        downsample="mergingv2",
        use_v2=True,
    )


def load_registration_weights(model, weights_path: str, device: str):
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if isinstance(checkpoint, dict):
        checkpoint = {
            key.removeprefix("module."): value
            for key, value in checkpoint.items()
        }

    model.load_state_dict(checkpoint)


def load_single_patient_batch(patient_path: str) -> Dict:
    batch = ReadH5d()(patient_path)
    return {
        key: value.unsqueeze(0) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def validate_mask_labels(mask_labels: Sequence[int]) -> List[int]:
    labels = [int(label) for label in mask_labels]
    if not labels:
        raise ValueError("mask_labels must contain at least one label.")
    for label in labels:
        if label < 1 or label > 128:
            raise ValueError(f"mask label must be in [1, 128], got {label}.")
    return labels


def finetune_registration_model_on_roi(
    patient_path: str,
    mask_labels: Sequence[int],
    model: torch.nn.Module = None,
    weights_path: str = REGISTRATION_WEIGHTS_PATH,
    epochs: int = 1,
    lr: float = 1e-6,
    smoothness_lambda: float = 4500.0,
    save_model_path: Optional[str] = None,
    device: str = DEVICE,
) -> torch.nn.Module:
    """
    Agent tool: fine-tune one registration model on one patient and selected mask labels.

    The moving image is FDG and the fixed image is PSMA.
    Mask loss is computed only on mask_labels.
    CT loss is computed on the whole CT volume.
    Smoothness uses the original l2_gradient regularization.
    """
    labels = validate_mask_labels(mask_labels)
    epochs = max(1, int(epochs))

    if model is None:
        model = build_registration_model().to(device)
        load_registration_weights(model, weights_path, device)
    else:
        model = model.to(device)
    model.train()

    batch = load_single_patient_batch(patient_path)

    fdg_ct = batch["fdg_ct"].to(device)
    psma_ct = batch["psma_ct"].to(device)
    fdg_pt = batch["fdg_pt"].to(device)
    psma_pt = batch["psma_pt"].to(device)
    fdg_mask = batch["fdg_mask"].to(device)
    psma_mask = batch["psma_mask"].to(device)

    if fdg_pt.shape[0] != 1:
        raise ValueError(f"Agent tool expects batch size 1, got {fdg_pt.shape[0]}.")

    identity_grid = make_identity_grid_m11(SPATIAL_SIZE, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    labels_for_case = [labels]
    fdg_roi_masks = labels_to_binary_masks(fdg_mask, labels_for_case)
    psma_roi_masks = labels_to_binary_masks(psma_mask, labels_for_case)

    for epoch in range(epochs):
        model_input = torch.cat([fdg_pt, psma_pt], dim=1)
        ddf, grid = predict_ddf_and_grid(model, model_input, identity_grid)

        smoothness_loss = smoothness_lambda * l2_gradient(ddf)

        warped_fdg_roi_masks = torch.nn.functional.grid_sample(fdg_roi_masks, grid)
        warped_fdg_ct = torch.nn.functional.grid_sample(fdg_ct, grid)

        mask_loss = loss_function_dice(psma_roi_masks, warped_fdg_roi_masks)
        ct_loss = loss_function_dice(warped_fdg_ct, psma_ct)
        total_loss = mask_loss + ct_loss + smoothness_loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if save_model_path is not None:
        save_dir = os.path.dirname(save_model_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), save_model_path)

    return model
