import argparse
import json
import os
import sys

import torch
from monai.networks.nets import SwinUNETR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from General.data_loader import ReadH5d
from General.segments import SEGMENT_INDEX
from llm_Registration.config import (
    DEVICE,
    MASK_NAMES,
    REGISTRATION_WEIGHTS_PATH,
    SINGLE_CASE_OUTPUT_DIR,
    SPATIAL_SIZE,
)
from Registration.inferencing import (
    compute_tre_single,
    dice_metric,
    get_binary_mask_with_label,
    mutual_information,
    normalized_cross_correlation,
)
from Registration.training import make_identity_grid_m11, predict_ddf_and_grid


DEFAULT_WEIGHTS_PATH = REGISTRATION_WEIGHTS_PATH
DEFAULT_OUTPUT_DIR = SINGLE_CASE_OUTPUT_DIR
DEFAULT_SPATIAL_SIZE = SPATIAL_SIZE
DEFAULT_DEVICE = DEVICE
DEFAULT_MASK_NAMES = MASK_NAMES



def build_registration_model():
    return SwinUNETR(
        in_channels=2,
        out_channels=3,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        downsample="mergingv2",
        use_v2=True,
    )


def load_model_weights(model, weights_path, device):
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


def spacing_to_list(spacing):
    if torch.is_tensor(spacing):
        return [float(item) for item in spacing.reshape(-1).tolist()]
    return [item.item() if hasattr(item, "item") else float(item) for item in spacing]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def make_case_json_name(patient_path):
    case_id = os.path.splitext(os.path.basename(patient_path))[0]
    source_dir = os.path.basename(os.path.dirname(patient_path))
    return f"{source_dir}-{case_id}.json"


def load_single_case_batch(patient_path):
    batch = ReadH5d()(patient_path)
    return {
        key: value.unsqueeze(0) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def prepare_case_variables(batch, device=DEFAULT_DEVICE):
    fdg_ct = batch["fdg_ct"].to(device)
    fdg_pt = batch["fdg_pt"].to(device)
    fdg_mask = batch["fdg_mask"].to(device)
    fdg_spacing = spacing_to_list(batch["fdg_spacing"])

    psma_ct = batch["psma_ct"].to(device)
    psma_pt = batch["psma_pt"].to(device)
    psma_mask = batch["psma_mask"].to(device)
    psma_spacing = spacing_to_list(batch["psma_spacing"])

    spacing = (torch.tensor(fdg_spacing) + torch.tensor(psma_spacing)) / 2

    return {
        "fdg_ct": fdg_ct,
        "fdg_pt": fdg_pt,
        "fdg_mask": fdg_mask,
        "psma_ct": psma_ct,
        "psma_pt": psma_pt,
        "psma_mask": psma_mask,
        "spacing": spacing,
    }


@torch.no_grad()
def make_case_json_from_h5(
    model,
    batch,
    patient_path,
    masks_names=DEFAULT_MASK_NAMES,
    device=DEFAULT_DEVICE,
):
    model.eval()
    model.to(device)


    identity_grid = make_identity_grid_m11(
        DEFAULT_SPATIAL_SIZE,
        device=device,
    )

    case_vars = prepare_case_variables(batch, device=device)
    fdg_ct = case_vars["fdg_ct"]
    fdg_pt = case_vars["fdg_pt"]
    fdg_mask = case_vars["fdg_mask"]
    psma_ct = case_vars["psma_ct"]
    psma_pt = case_vars["psma_pt"]
    psma_mask = case_vars["psma_mask"]
    spacing = case_vars["spacing"]

    model_input = torch.cat([fdg_pt, psma_pt], dim=1)
    _, grid = predict_ddf_and_grid(model, model_input, identity_grid)

    warped_fdg_pt = torch.nn.functional.grid_sample(fdg_pt, grid)
    warped_fdg_ct = torch.nn.functional.grid_sample(fdg_ct, grid)

    organ_metrics = {}
    case_dice_before = []
    case_dice_after = []
    case_tre_before = []
    case_tre_after = []

    for name in masks_names:
        if name not in SEGMENT_INDEX:
            raise ValueError(f"Unknown segment name: {name}")

        mask_label = SEGMENT_INDEX[name]
        binary_fdg_mask = get_binary_mask_with_label(fdg_mask, mask_label)
        binary_psma_mask = get_binary_mask_with_label(psma_mask, mask_label)
        warped_fdg_mask = torch.nn.functional.grid_sample(binary_fdg_mask, grid)

        dice_before = dice_metric(binary_fdg_mask, binary_psma_mask).cpu().item()
        dice_after = dice_metric(warped_fdg_mask, binary_psma_mask).cpu().item()
        tre_before = compute_tre_single(
            binary_fdg_mask,
            binary_psma_mask,
            spacing,
        ).cpu().item()
        tre_after = compute_tre_single(
            warped_fdg_mask,
            binary_psma_mask,
            spacing,
        ).cpu().item()

        case_dice_before.append(dice_before)
        case_dice_after.append(dice_after)
        case_tre_before.append(tre_before)
        case_tre_after.append(tre_after)

        organ_metrics[name] = {
            "label": mask_label,
            "dice_before": dice_before,
            "dice_after": dice_after,
            "dice_delta": dice_after - dice_before,
            "tre_before": tre_before,
            "tre_after": tre_after,
            "tre_delta": tre_after - tre_before,
        }

    pet_mi_before = mutual_information(fdg_pt, psma_pt).cpu().item()
    pet_mi_after = mutual_information(warped_fdg_pt, psma_pt).cpu().item()
    pet_ncc_before = normalized_cross_correlation(fdg_pt, psma_pt).cpu().item()
    pet_ncc_after = normalized_cross_correlation(warped_fdg_pt, psma_pt).cpu().item()
    ct_mi_before = mutual_information(fdg_ct, psma_ct).cpu().item()
    ct_mi_after = mutual_information(warped_fdg_ct, psma_ct).cpu().item()
    ct_ncc_before = normalized_cross_correlation(fdg_ct, psma_ct).cpu().item()
    ct_ncc_after = normalized_cross_correlation(warped_fdg_ct, psma_ct).cpu().item()

    case_id = os.path.splitext(os.path.basename(patient_path))[0]
    case_json = {
        "case_id": case_id,
        "data_path": patient_path,
        "metrics": {
            "mean_dice_before": mean(case_dice_before),
            "mean_dice_after": mean(case_dice_after),
            "mean_dice_delta": mean(case_dice_after) - mean(case_dice_before),
            "mean_tre_before": mean(case_tre_before),
            "mean_tre_after": mean(case_tre_after),
            "mean_tre_delta": mean(case_tre_after) - mean(case_tre_before),
            "pet_mi_before": pet_mi_before,
            "pet_mi_after": pet_mi_after,
            "pet_mi_delta": pet_mi_after - pet_mi_before,
            "pet_ncc_before": pet_ncc_before,
            "pet_ncc_after": pet_ncc_after,
            "pet_ncc_delta": pet_ncc_after - pet_ncc_before,
            "ct_mi_before": ct_mi_before,
            "ct_mi_after": ct_mi_after,
            "ct_mi_delta": ct_mi_after - ct_mi_before,
            "ct_ncc_before": ct_ncc_before,
            "ct_ncc_after": ct_ncc_after,
            "ct_ncc_delta": ct_ncc_after - ct_ncc_before,
        },
        "organs": organ_metrics,
    }

    return case_json


@torch.no_grad()
def inference_single_case_json(
    patient_path,
    output_dir=DEFAULT_OUTPUT_DIR,
    weights_path=DEFAULT_WEIGHTS_PATH,
    masks_names=DEFAULT_MASK_NAMES,
    device=DEFAULT_DEVICE,
):
    os.makedirs(output_dir, exist_ok=True)

    batch = load_single_case_batch(patient_path)
    model = build_registration_model()
    load_model_weights(model, weights_path, device)

    return make_case_json_from_h5(
        model=model,
        batch=batch,
        patient_path=patient_path,
        masks_names=masks_names,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run one patient registration inference and print one JSON object."
    )
    parser.add_argument(
        "patient_path",
        type=str,
        help="Path to one patient .h5 file.",
    )

    args = parser.parse_args()
    result_json = inference_single_case_json(args.patient_path)
    print(json.dumps(result_json, indent=2, ensure_ascii=False))
