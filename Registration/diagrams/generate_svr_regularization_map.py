import os
import sys

import SimpleITK as sitk
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from General.data_loader import ReadH5d
from General.save_itk import tensor_to_itk
from Registration.baseline_models import build_baseline_model


H5_PATH = "/data2/xiangcen/data/pet_gen/processed/batch3_h5_v2/patient_0067.h5"
WEIGHTS_PATH = "/share/home/xcwu/registration_v3/svr_diff_l4500_k10_mar3000_beta1_a1.1_b1_repo_logbeta.ptm"
CT_KEY = "fdg_ct"
OUTPUT_DIR = "Registration/diagrams"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_state_dict(model, weights_path, device):
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


def add_batch_dim_if_needed(tensor):
    if tensor.dim() == 4:
        return tensor.unsqueeze(0)
    if tensor.dim() == 5:
        return tensor
    raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(tensor.shape)}")


@torch.no_grad()
def infer_regularization_map(h5_path, weights_path, device):
    batch = ReadH5d()(h5_path)

    fdg_pt = add_batch_dim_if_needed(batch["fdg_pt"]).float().to(device)
    psma_pt = add_batch_dim_if_needed(batch["psma_pt"]).float().to(device)

    model = build_baseline_model("svr_diff").to(device)
    load_state_dict(model, weights_path, device)
    model.eval()

    outputs = model(fdg_pt, psma_pt)
    return batch, outputs["regularization_map"]


def spacing_for_ct_key(batch, ct_key):
    if ct_key == "fdg_ct":
        return [float(x) for x in batch["fdg_spacing"]]
    if ct_key == "psma_ct":
        return [float(x) for x in batch["psma_spacing"]]
    raise ValueError("ct_key must be either 'fdg_ct' or 'psma_ct'.")


def save_ct_and_regularization_nii(
    ct_tensor,
    regularization_map,
    spacing,
    output_dir=OUTPUT_DIR,
    prefix="svr_regularization",
):
    os.makedirs(output_dir, exist_ok=True)

    ct_path = os.path.join(output_dir, f"{prefix}_ct.nii.gz")
    regularization_path = os.path.join(
        output_dir,
        f"{prefix}_regularization_map.nii.gz",
    )

    ct_itk = tensor_to_itk(ct_tensor.detach().cpu(), spacing)
    regularization_itk = tensor_to_itk(
        regularization_map.detach().cpu().float(),
        spacing,
    )

    sitk.WriteImage(ct_itk, ct_path)
    sitk.WriteImage(regularization_itk, regularization_path)

    print(f"Saved CT to {ct_path}")
    print(f"Saved regularization map to {regularization_path}")
    return ct_path, regularization_path


def main(
    h5_path=H5_PATH,
    weights_path=WEIGHTS_PATH,
    ct_key=CT_KEY,
    output_dir=OUTPUT_DIR,
    device=DEVICE,
):
    if ct_key not in {"fdg_ct", "psma_ct"}:
        raise ValueError("ct_key must be either 'fdg_ct' or 'psma_ct'.")

    batch, regularization_map = infer_regularization_map(
        h5_path=h5_path,
        weights_path=weights_path,
        device=device,
    )

    ct_tensor = add_batch_dim_if_needed(batch[ct_key]).float()
    regularization_map = regularization_map.float()

    if ct_tensor.shape != regularization_map.shape:
        raise ValueError(
            "CT and regularization map shapes do not match: "
            f"{tuple(ct_tensor.shape)} vs {tuple(regularization_map.shape)}"
        )

    patient_name = os.path.splitext(os.path.basename(h5_path))[0]
    weights_name = os.path.splitext(os.path.basename(weights_path))[0]
    prefix = f"{patient_name}_{weights_name}_{ct_key}"
    spacing = spacing_for_ct_key(batch, ct_key)

    return save_ct_and_regularization_nii(
        ct_tensor=ct_tensor,
        regularization_map=regularization_map,
        spacing=spacing,
        output_dir=output_dir,
        prefix=prefix,
    )


if __name__ == "__main__":
    main()
