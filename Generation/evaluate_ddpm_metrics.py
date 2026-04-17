import csv
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F


BASE_DIR = Path("/results/generation_ddpm/psma_ct_to_psma_pt")
GT_NAME = "psma_gt.nii.gz"
PRED_NAME = "psma_prediction.nii.gz"

# If your saved NIfTI is already in SUV, keep 1.0.
# If it is normalized to [0, 1], set this to the SUV scale used in preprocessing.
SUV_SCALE = 1.0

# If None, PSNR uses each case's intensity range.
# If your data is normalized, you can set this to 1.0.
PSNR_DATA_RANGE = None

SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5
SAVE_CSV = False
CSV_NAME = "metrics_summary.csv"
EPS = 1e-8


def load_nifti_array(path):
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    return array


def get_valid_case_dirs(base_dir):
    case_dirs = []
    for item in sorted(base_dir.iterdir()):
        if not item.is_dir():
            continue
        gt_path = item / GT_NAME
        pred_path = item / PRED_NAME
        if gt_path.exists() and pred_path.exists():
            case_dirs.append(item)
    return case_dirs


def get_psnr_data_range(gt, pred):
    if PSNR_DATA_RANGE is not None:
        return float(PSNR_DATA_RANGE)

    value_min = float(min(np.min(gt), np.min(pred)))
    value_max = float(max(np.max(gt), np.max(pred)))
    data_range = value_max - value_min

    if data_range <= EPS:
        data_range = max(abs(value_max), 1.0)

    return float(data_range)


def make_gaussian_kernel_3d(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel_3d = (
        kernel_1d[:, None, None]
        * kernel_1d[None, :, None]
        * kernel_1d[None, None, :]
    )
    kernel_3d = kernel_3d / kernel_3d.sum()
    return kernel_3d.view(1, 1, window_size, window_size, window_size)


def compute_ssim_3d(gt, pred, data_range):
    gt_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
    pred_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)

    min_dim = min(gt.shape)
    window_size = min(SSIM_WINDOW_SIZE, min_dim)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 1:
        window_size = 1

    kernel = make_gaussian_kernel_3d(
        window_size,
        SSIM_SIGMA,
        gt_tensor.device,
        gt_tensor.dtype,
    )
    padding = window_size // 2

    mu_gt = F.conv3d(gt_tensor, kernel, padding=padding)
    mu_pred = F.conv3d(pred_tensor, kernel, padding=padding)

    mu_gt_sq = mu_gt * mu_gt
    mu_pred_sq = mu_pred * mu_pred
    mu_gt_pred = mu_gt * mu_pred

    sigma_gt_sq = F.conv3d(gt_tensor * gt_tensor, kernel, padding=padding) - mu_gt_sq
    sigma_pred_sq = F.conv3d(pred_tensor * pred_tensor, kernel, padding=padding) - mu_pred_sq
    sigma_gt_pred = F.conv3d(gt_tensor * pred_tensor, kernel, padding=padding) - mu_gt_pred

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2.0 * mu_gt_pred + c1) * (2.0 * sigma_gt_pred + c2)
    denominator = (mu_gt_sq + mu_pred_sq + c1) * (sigma_gt_sq + sigma_pred_sq + c2)
    ssim_map = numerator / (denominator + EPS)

    return float(ssim_map.mean().item())


def compute_case_metrics(gt, pred):
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt {gt.shape}, pred {pred.shape}")

    gt_volume = gt.astype(np.float32) * SUV_SCALE
    pred_volume = pred.astype(np.float32) * SUV_SCALE

    valid_mask = np.isfinite(gt_volume) & np.isfinite(pred_volume)
    if not np.any(valid_mask):
        raise ValueError("No finite voxels found.")

    gt_valid = gt_volume[valid_mask]
    pred_valid = pred_volume[valid_mask]

    diff = pred_valid - gt_valid
    abs_diff = np.abs(diff)
    squared_diff = diff ** 2

    mae = float(np.mean(abs_diff))
    mse = float(np.mean(squared_diff))
    rmse = float(math.sqrt(mse))
    nmse = float(np.sum(squared_diff) / (np.sum(gt_valid ** 2) + EPS))

    data_range = get_psnr_data_range(gt_valid, pred_valid)
    if mse <= EPS:
        psnr = float("inf")
    else:
        psnr = float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse))

    gt_volume = np.where(valid_mask, gt_volume, 0.0)
    pred_volume = np.where(valid_mask, pred_volume, 0.0)
    ssim = compute_ssim_3d(
        gt_volume,
        pred_volume,
        data_range,
    )

    return {
        "mae_suv": mae,
        "rmse_suv": rmse,
        "nmse": nmse,
        "psnr_db": psnr,
        "ssim": ssim,
    }


def safe_mean(values):
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


def save_csv(rows, csv_path):
    fieldnames = [
        "case",
        "mae_suv",
        "rmse_suv",
        "nmse",
        "psnr_db",
        "ssim",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    base_dir = Path(BASE_DIR)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir does not exist: {base_dir}")

    case_dirs = get_valid_case_dirs(base_dir)
    if not case_dirs:
        raise FileNotFoundError(
            f"No valid case folder found under {base_dir}. "
            f"Each case folder must contain {GT_NAME} and {PRED_NAME}."
        )

    rows = []

    print(f"Base dir: {base_dir}")
    print(f"Cases found: {len(case_dirs)}")
    print("-" * 88)
    print(
        f"{'case':<20}"
        f"{'MAE(SUV)':>12}"
        f"{'RMSE(SUV)':>12}"
        f"{'NMSE':>12}"
        f"{'PSNR(dB)':>12}"
        f"{'SSIM':>10}"
    )
    print("-" * 88)

    for case_dir in case_dirs:
        gt = load_nifti_array(case_dir / GT_NAME)
        pred = load_nifti_array(case_dir / PRED_NAME)
        metrics = compute_case_metrics(gt, pred)

        row = {"case": case_dir.name, **metrics}
        rows.append(row)

        print(
            f"{case_dir.name:<20}"
            f"{metrics['mae_suv']:>12.4f}"
            f"{metrics['rmse_suv']:>12.4f}"
            f"{metrics['nmse']:>12.6f}"
            f"{metrics['psnr_db']:>12.4f}"
            f"{metrics['ssim']:>10.4f}"
        )

    mean_row = {
        "case": "mean",
        "mae_suv": safe_mean([row["mae_suv"] for row in rows]),
        "rmse_suv": safe_mean([row["rmse_suv"] for row in rows]),
        "nmse": safe_mean([row["nmse"] for row in rows]),
        "psnr_db": safe_mean([row["psnr_db"] for row in rows]),
        "ssim": safe_mean([row["ssim"] for row in rows]),
    }

    print("-" * 88)
    print(
        f"{mean_row['case']:<20}"
        f"{mean_row['mae_suv']:>12.4f}"
        f"{mean_row['rmse_suv']:>12.4f}"
        f"{mean_row['nmse']:>12.6f}"
        f"{mean_row['psnr_db']:>12.4f}"
        f"{mean_row['ssim']:>10.4f}"
    )

    if SAVE_CSV:
        csv_path = base_dir / CSV_NAME
        save_csv(rows + [mean_row], csv_path)
        print(f"\nSaved csv: {csv_path}")


if __name__ == "__main__":
    main()
