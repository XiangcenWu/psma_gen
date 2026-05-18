import os
import sys
from typing import Dict, Optional

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from General.segments import SEGMENT_INDEX
from Registration.inferencing import get_binary_mask_with_label
from Registration.smoothness_losses import l2_gradient
from Registration.training import make_identity_grid_m11
from llm_Registration.inference_single_case import (
    DEFAULT_DEVICE,
    DEFAULT_SPATIAL_SIZE,
    DEFAULT_WEIGHTS_PATH,
    build_registration_model,
    load_single_case_batch,
    load_model_weights,
    make_case_json_from_model,
)


BLADDER_NAME = "urinary_bladder"
BLADDER_LABEL = SEGMENT_INDEX[BLADDER_NAME]


def _dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    score = (2.0 * intersection + eps) / (pred.sum(dim=1) + target.sum(dim=1) + eps)
    return 1.0 - score.mean()


def finetune_bladder_registration(
    patient_path: str,
    weights_path: str = DEFAULT_WEIGHTS_PATH,
    max_steps: int = 20,
    lr: float = 1e-6,
    smoothness_lambda: float = 1000.0,
    image_loss_lambda: float = 0.05,
    save_model_path: Optional[str] = None,
    device: str = DEFAULT_DEVICE,
) -> Dict:
    """
    Test-time fine-tune the registration model for one case using the bladder mask.

    The fixed target is the PSMA bladder mask and the moving source is the FDG
    bladder mask. The returned JSON compares the base model and the tuned model.
    """
    max_steps = min(max(int(max_steps), 1), 300)
    lr = min(max(float(lr), 1e-7), 1e-4)

    batch = load_single_case_batch(patient_path)
    model = build_registration_model()
    load_model_weights(model, weights_path, device)
    model.train()
    model.to(device)

    identity_grid = make_identity_grid_m11(DEFAULT_SPATIAL_SIZE, device=device)
    before_json = make_case_json_from_model(
        model=model,
        batch=batch,
        patient_path=patient_path,
        identity_grid=identity_grid,
        device=device,
    )
    model.train()

    fdg_pt = batch["fdg_pt"].to(device)
    fdg_mask = batch["fdg_mask"].to(device)
    psma_pt = batch["psma_pt"].to(device)
    psma_mask = batch["psma_mask"].to(device)

    moving_bladder = get_binary_mask_with_label(fdg_mask, BLADDER_LABEL)
    fixed_bladder = get_binary_mask_with_label(psma_mask, BLADDER_LABEL)

    if moving_bladder.sum().item() < 1 or fixed_bladder.sum().item() < 1:
        return {
            "tool": "finetune_bladder_registration",
            "status": "skipped",
            "reason": "Bladder mask is empty in moving or fixed image.",
            "patient_path": patient_path,
            "before": before_json,
            "after": before_json,
        }

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_history = []

    for _ in range(max_steps):
        model_input = torch.cat([fdg_pt, psma_pt], dim=1)
        ddf = torch.tanh(model(model_input))
        grid = identity_grid + ddf
        grid = grid.permute(0, 2, 3, 4, 1)

        warped_bladder = torch.nn.functional.grid_sample(moving_bladder, grid)
        warped_fdg_pt = torch.nn.functional.grid_sample(fdg_pt, grid)

        roi_loss = _dice_loss(warped_bladder, fixed_bladder)
        image_loss = torch.mean((warped_fdg_pt - psma_pt) ** 2)
        smoothness_loss = l2_gradient(ddf)
        loss = roi_loss + image_loss_lambda * image_loss + smoothness_lambda * smoothness_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(
            {
                "total_loss": loss.detach().cpu().item(),
                "roi_dice_loss": roi_loss.detach().cpu().item(),
                "image_loss": image_loss.detach().cpu().item(),
                "smoothness_loss": smoothness_loss.detach().cpu().item(),
            }
        )

    after_json = make_case_json_from_model(
        model=model,
        batch=batch,
        patient_path=patient_path,
        identity_grid=identity_grid,
        device=device,
    )

    if save_model_path is not None:
        save_dir = os.path.dirname(save_model_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), save_model_path)

    before_bladder = before_json["organs"][BLADDER_NAME]
    after_bladder = after_json["organs"][BLADDER_NAME]

    return {
        "tool": "finetune_bladder_registration",
        "status": "success",
        "patient_path": patient_path,
        "organ_name": BLADDER_NAME,
        "label": BLADDER_LABEL,
        "parameters": {
            "weights_path": weights_path,
            "max_steps": max_steps,
            "lr": lr,
            "smoothness_lambda": smoothness_lambda,
            "image_loss_lambda": image_loss_lambda,
            "save_model_path": save_model_path,
        },
        "bladder_before": before_bladder,
        "bladder_after": after_bladder,
        "bladder_improvement": {
            "dice_delta": after_bladder["dice_after"] - before_bladder["dice_after"],
            "tre_delta": after_bladder["tre_after"] - before_bladder["tre_after"],
        },
        "global_before": before_json["metrics"],
        "global_after": after_json["metrics"],
        "loss_history": loss_history,
        "before": before_json,
        "after": after_json,
    }
