import argparse
import json
import os
import sys

import torch
from monai.networks.nets import SwinUNETR
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from General.data_loader import ReadH5d, create_data_loader
from General.dataset_sample import split_multiple_train_test
from General.segments import SEGMENT_INDEX
from Registration.inferencing import (
    compute_tre_single,
    dice_metric,
    get_binary_mask_with_label,
    mutual_information,
    normalized_cross_correlation,
    save_registration_results,
)
from Registration.training import make_identity_grid_m11


DEFAULT_DATA_DIRS = [
    "/data2/xiangcen/data/pet_gen/processed/batch1_h5_v2",
    "/data2/xiangcen/data/pet_gen/processed/batch2_h5_v2",
]
DEFAULT_NUM_VALIDATIONS = [40, 40]
DEFAULT_BATCH3_DIR = "/data2/xiangcen/data/pet_gen/processed/batch3_h5_v2"
DEFAULT_SPATIAL_SIZE = (128, 128, 384)
DEFAULT_OUTPUT_DIR = "/share/home/xcwu/pet_reg_results_llm"
DEFAULT_SEED = 325
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_MASK_NAMES = list(SEGMENT_INDEX.keys())


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


def collect_weight_paths(weights_path):
    if os.path.isdir(weights_path):
        paths = [
            os.path.join(weights_path, name)
            for name in os.listdir(weights_path)
            if os.path.isfile(os.path.join(weights_path, name))
        ]
        return sorted(paths)
    return [weights_path]


def spacing_to_list(spacing):
    if torch.is_tensor(spacing):
        return [float(item) for item in spacing.reshape(-1).tolist()]
    return [item.item() if hasattr(item, "item") else float(item) for item in spacing]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


@torch.no_grad()
def inference_batch_with_case_json(
    model,
    loader,
    data_paths,
    identity_grid,
    txt_filename,
    json_output_dir,
    masks_names=DEFAULT_MASK_NAMES,
    device=DEFAULT_DEVICE,
):
    mask_labels = []
    for name in masks_names:
        if name not in SEGMENT_INDEX:
            raise ValueError(f"Unknown segment name: {name}")
        mask_labels.append(SEGMENT_INDEX[name])

    os.makedirs(json_output_dir, exist_ok=True)

    model.eval()
    model.to(device)
    identity_grid = identity_grid.to(device)

    num_masks = len(masks_names)
    dice_before_lists = [[] for _ in range(num_masks)]
    dice_after_lists = [[] for _ in range(num_masks)]
    tre_before_lists = [[] for _ in range(num_masks)]
    tre_after_lists = [[] for _ in range(num_masks)]

    for batch, data_path in tqdm(
        zip(loader, data_paths),
        desc="inferencing",
        total=len(data_paths),
    ):
        assert batch["fdg_pt"].shape[0] == 1, (
            f"Expected batch size 1, got {batch['fdg_pt'].shape[0]}"
        )

        fdg_ct = batch["fdg_ct"].to(device)
        fdg_pt = batch["fdg_pt"].to(device)
        fdg_mask = batch["fdg_mask"].to(device)
        fdg_spacing = spacing_to_list(batch["fdg_spacing"])

        psma_ct = batch["psma_ct"].to(device)
        psma_pt = batch["psma_pt"].to(device)
        psma_mask = batch["psma_mask"].to(device)
        psma_spacing = spacing_to_list(batch["psma_spacing"])

        spacing = (torch.tensor(fdg_spacing) + torch.tensor(psma_spacing)) / 2

        model_input = torch.cat([fdg_pt, psma_pt], dim=1)
        ddf = torch.tanh(model(model_input))
        grid = identity_grid + ddf
        grid = grid.permute(0, 2, 3, 4, 1)

        warped_fdg_pt = torch.nn.functional.grid_sample(fdg_pt, grid)
        warped_fdg_ct = torch.nn.functional.grid_sample(fdg_ct, grid)

        organ_metrics = {}
        case_dice_before = []
        case_dice_after = []
        case_tre_before = []
        case_tre_after = []

        for idx, name in enumerate(masks_names):
            mask_label = mask_labels[idx]
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

            dice_before_lists[idx].append(dice_before)
            dice_after_lists[idx].append(dice_after)
            tre_before_lists[idx].append(tre_before)
            tre_after_lists[idx].append(tre_after)

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

        case_id = os.path.splitext(os.path.basename(data_path))[0]
        source_dir = os.path.basename(os.path.dirname(data_path))
        json_name = f"{source_dir}-{case_id}.json"
        case_json = {
            "case_id": case_id,
            "data_path": data_path,
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
        save_json(case_json, os.path.join(json_output_dir, json_name))

    save_registration_results(
        txt_filename,
        masks_names,
        dice_before_lists,
        dice_after_lists,
        tre_before_lists,
        tre_after_lists,
    )


def main(weights_path):
    device = DEFAULT_DEVICE
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

    _, test_list = split_multiple_train_test(
        DEFAULT_DATA_DIRS,
        DEFAULT_NUM_VALIDATIONS,
        seed=DEFAULT_SEED,
    )
    test_list += [
        os.path.join(DEFAULT_BATCH3_DIR, f)
        for f in os.listdir(DEFAULT_BATCH3_DIR)
        if f.endswith(".h5")
    ]

    test_loader = create_data_loader(
        test_list,
        ReadH5d(),
        batch_size=1,
        shuffle=False,
    )

    identity_grid = make_identity_grid_m11(
        DEFAULT_SPATIAL_SIZE,
        device=device,
    )

    model = build_registration_model()

    for current_weights_path in collect_weight_paths(weights_path):
        weight_name = os.path.splitext(os.path.basename(current_weights_path))[0]
        result_name = weight_name + ".txt"
        result_path = os.path.join(DEFAULT_OUTPUT_DIR, result_name)
        json_output_dir = os.path.join(DEFAULT_OUTPUT_DIR, weight_name)

        print(f"Running test inference: {current_weights_path}")
        print(f"Saving metrics to: {result_path}")
        print(f"Saving case JSON files to: {json_output_dir}")

        load_model_weights(model, current_weights_path, device)
        inference_batch_with_case_json(
            model=model,
            loader=test_loader,
            data_paths=test_list,
            identity_grid=identity_grid,
            txt_filename=result_path,
            json_output_dir=json_output_dir,
            device=device,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run llm_Registration test-set inference."
    )
    parser.add_argument(
        "weights_path",
        type=str,
        help="Path to one trained model weight file, or a directory of weight files.",
    )

    args = parser.parse_args()
    main(args.weights_path)
