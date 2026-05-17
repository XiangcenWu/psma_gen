import argparse
import os
import sys

import torch
from monai.networks.nets import SwinUNETR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from General.data_loader import ReadH5d, create_data_loader
from General.dataset_sample import split_multiple_train_test
from Registration.inferencing import inference_batch
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
        result_name = os.path.splitext(os.path.basename(current_weights_path))[0] + ".txt"
        result_path = os.path.join(DEFAULT_OUTPUT_DIR, result_name)

        if os.path.exists(result_path):
            print(f"Skip existing result: {result_path}")
            continue

        print(f"Running test inference: {current_weights_path}")
        print(f"Saving metrics to: {result_path}")

        load_model_weights(model, current_weights_path, device)
        inference_batch(
            model=model,
            loader=test_loader,
            identity_grid=identity_grid,
            filename=result_path,
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
