import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Registration.baseline_models import build_baseline_model
from Registration.diffeomorphic import predict_diffeomorphic_ddf_and_grid
from Registration.inference_baseline import (
    DEFAULT_DATA_DIRS,
    DEFAULT_TEST_COUNTS,
    infer_model_name,
    list_weight_paths,
    load_state_dict,
)
from General.data_loader import ReadH5d, create_data_loader
from General.dataset_sample import split_multiple_train_test
from Registration.training import (
    get_registration_input_keys,
    make_identity_grid_m11,
    make_registration_input,
    predict_ddf_and_grid,
)


DEFAULT_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "images")
MASK_CONTOUR_COLORS = [
    "cyan",
    "yellow",
    "lime",
    "magenta",
    "orange",
    "deepskyblue",
    "red",
    "white",
    "dodgerblue",
    "springgreen",
    "violet",
    "gold",
]


def build_visualization_test_loader(args):
    if len(args.data_dirs) != len(args.test_counts):
        raise ValueError("--data_dirs and --test_counts must have the same length.")

    _, test_list = split_multiple_train_test(args.data_dirs, args.test_counts)
    test_loader = create_data_loader(
        test_list,
        ReadH5d(),
        batch_size=1,
        shuffle=False,
    )
    return test_loader, test_list


def get_batch_by_index(loader, case_index):
    if case_index < 0:
        raise ValueError("--case_index must be >= 0.")

    for idx, batch in enumerate(loader):
        if idx == case_index:
            return batch

    raise IndexError(
        f"--case_index {case_index} is outside the test loader with "
        f"{len(loader)} batches."
    )


def tensor_to_volume(tensor):
    return tensor.detach().float().cpu().squeeze(0).squeeze(0).numpy()


def tensor_to_field(tensor):
    return tensor.detach().float().cpu().squeeze(0).numpy()


def robust_limits(*volumes, lower=1.0, upper=99.0):
    values = []
    for volume in volumes:
        finite_values = volume[np.isfinite(volume)]
        if finite_values.size:
            values.append(finite_values.reshape(-1))

    if not values:
        return 0.0, 1.0

    values = np.concatenate(values)
    vmin, vmax = np.percentile(values, [lower, upper])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if vmin == vmax:
            vmax = vmin + 1.0

    return float(vmin), float(vmax)


def select_slice_indices(*mask_volumes, num_slices=12, view="coronal"):
    slice_axis = 2 if view == "axial" else 1
    axis_size = mask_volumes[0].shape[slice_axis]
    nonzero_slices = []
    for mask_volume in mask_volumes:
        nonzero = np.where(mask_volume > 0)[slice_axis]
        if nonzero.size:
            nonzero_slices.append(nonzero)

    if nonzero_slices:
        nonzero_slices = np.concatenate(nonzero_slices)
        start = int(nonzero_slices.min())
        stop = int(nonzero_slices.max())
    else:
        start = 0
        stop = axis_size - 1

    indices = np.linspace(start, stop, num_slices).round().astype(int)
    return np.clip(indices, 0, axis_size - 1).tolist()


def plot_slice(volume, slice_index, view="coronal"):
    if view == "axial":
        return np.rot90(volume[:, :, slice_index])
    return np.rot90(volume[:, slice_index, :])


def get_ddf_rgb_scale(ddf_field):
    scale = np.percentile(np.abs(ddf_field[np.isfinite(ddf_field)]), 99.0)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.max(np.abs(ddf_field))) if ddf_field.size else 1.0
    if scale <= 0:
        scale = 1.0
    return scale


def plot_ddf_rgb_slice(ddf_field, slice_index, scale, view="coronal"):
    channels = [
        plot_slice(ddf_field[channel], slice_index, view=view)
        for channel in range(3)
    ]
    rgb = np.stack(channels, axis=-1)
    return np.clip(rgb / scale * 0.5 + 0.5, 0.0, 1.0)


def draw_deformation_grid(
    ax,
    ddf_field,
    slice_index,
    view="coronal",
    grid_step=8,
    linewidth=0.18,
):
    _, depth, height, width = ddf_field.shape
    if view == "axial":
        y_coords = np.linspace(-1.0, 1.0, height)
        z_coords = np.linspace(-1.0, 1.0, depth)
        z_grid, y_grid = np.meshgrid(z_coords, y_coords, indexing="ij")

        # Axial slices are volume[:, :, z], so the in-plane coordinates are z/y.
        y_warped = y_grid + ddf_field[1, :, :, slice_index]
        z_warped = z_grid + ddf_field[2, :, :, slice_index]

        h_warped = (y_warped + 1.0) * 0.5 * (height - 1)
        d_warped = (z_warped + 1.0) * 0.5 * (depth - 1)

        plot_x = d_warped
        plot_y = (height - 1) - h_warped
        x_limit = depth - 1
        y_limit = height - 1
    else:
        x_coords = np.linspace(-1.0, 1.0, width)
        z_coords = np.linspace(-1.0, 1.0, depth)
        z_grid, x_grid = np.meshgrid(z_coords, x_coords, indexing="ij")

        x_warped = x_grid + ddf_field[0, :, slice_index, :]
        z_warped = z_grid + ddf_field[2, :, slice_index, :]

        w_warped = (x_warped + 1.0) * 0.5 * (width - 1)
        d_warped = (z_warped + 1.0) * 0.5 * (depth - 1)

        plot_x = d_warped
        plot_y = (width - 1) - w_warped
        x_limit = depth - 1
        y_limit = width - 1

    ax.set_facecolor("black")
    ax.set_xlim(0, x_limit)
    ax.set_ylim(y_limit, 0)
    ax.set_aspect("equal")

    for row_idx in range(0, plot_x.shape[0], grid_step):
        ax.plot(
            plot_x[row_idx, :],
            plot_y[row_idx, :],
            color="white",
            linewidth=linewidth,
            alpha=0.9,
        )
    for col_idx in range(0, plot_x.shape[1], grid_step):
        ax.plot(
            plot_x[:, col_idx],
            plot_y[:, col_idx],
            color="white",
            linewidth=linewidth,
            alpha=0.9,
        )


def get_mask_labels(mask_volume):
    labels = np.unique(np.rint(mask_volume).astype(np.int64))
    return [label for label in labels.tolist() if label != 0]


def draw_mask_contours(ax, mask_volume, slice_index, view="coronal"):
    mask_slice = np.rint(plot_slice(mask_volume, slice_index, view=view)).astype(np.int64)
    for label in get_mask_labels(mask_slice):
        binary_slice = mask_slice == label
        if not binary_slice.any():
            continue
        ax.contour(
            binary_slice.astype(float),
            levels=[0.5],
            colors=MASK_CONTOUR_COLORS[(label - 1) % len(MASK_CONTOUR_COLORS)],
            linewidths=0.35,
        )


def save_figure(
    fixed_pet,
    warped_moving_pet,
    fixed_ct,
    warped_moving_ct,
    ddf,
    fixed_mask,
    warped_moving_mask,
    output_path,
    num_slices=12,
    show_contour=True,
    title=None,
    grid_step=8,
    view="coronal",
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fixed_pet = tensor_to_volume(fixed_pet)
    warped_moving_pet = tensor_to_volume(warped_moving_pet)
    fixed_ct = tensor_to_volume(fixed_ct)
    warped_moving_ct = tensor_to_volume(warped_moving_ct)
    ddf = tensor_to_field(ddf)
    fixed_mask = tensor_to_volume(fixed_mask)
    warped_moving_mask = tensor_to_volume(warped_moving_mask)

    slice_indices = select_slice_indices(
        fixed_mask,
        warped_moving_mask,
        num_slices=num_slices,
        view=view,
    )
    pet_vmin, pet_vmax = robust_limits(fixed_pet, warped_moving_pet)
    ct_vmin, ct_vmax = robust_limits(fixed_ct, warped_moving_ct)
    ddf_rgb_scale = get_ddf_rgb_scale(ddf)

    fig, axes = plt.subplots(
        6,
        num_slices,
        figsize=(num_slices * 0.72, 7.5),
        constrained_layout=False,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.998,
        bottom=0.005,
        top=0.90 if title else 0.94,
        wspace=0.0,
        hspace=0.015,
    )
    if title:
        fig.suptitle(title, fontsize=8, y=0.985)

    image_rows = [
        ("fixed PET", fixed_pet, "gray_r", pet_vmin, pet_vmax, fixed_mask),
        (
            "warped moving PET",
            warped_moving_pet,
            "gray_r",
            pet_vmin,
            pet_vmax,
            warped_moving_mask,
        ),
        ("fixed CT", fixed_ct, "gray", ct_vmin, ct_vmax, fixed_mask),
        (
            "warped moving CT",
            warped_moving_ct,
            "gray",
            ct_vmin,
            ct_vmax,
            warped_moving_mask,
        ),
    ]

    for row_idx, (label, volume, cmap, vmin, vmax, contour_mask) in enumerate(image_rows):
        for col_idx, slice_index in enumerate(slice_indices):
            ax = axes[row_idx, col_idx]
            ax.imshow(
                plot_slice(volume, slice_index, view=view),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            if show_contour:
                draw_mask_contours(ax, contour_mask, slice_index, view=view)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_idx == 0:
                axis_name = "z" if view == "axial" else "y"
                ax.set_title(f"{axis_name}={slice_index}", fontsize=6)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=7)

    for col_idx, slice_index in enumerate(slice_indices):
        ax = axes[4, col_idx]
        ax.imshow(plot_ddf_rgb_slice(ddf, slice_index, ddf_rgb_scale, view=view))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if col_idx == 0:
            ax.set_ylabel("DDF RGB", fontsize=7)

        ax = axes[5, col_idx]
        draw_deformation_grid(ax, ddf, slice_index, view=view, grid_step=grid_step)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if col_idx == 0:
            ax.set_ylabel("DDF grid", fontsize=7)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def run_single_case(args, weights_path):
    test_loader, test_list = build_visualization_test_loader(args)
    batch = get_batch_by_index(test_loader, args.case_index)
    case_name = os.path.basename(test_list[args.case_index])

    model_name = infer_model_name(weights_path, args.baseline_model)
    input_keys = get_registration_input_keys(args.use_ct_input)

    model = build_baseline_model(model_name, in_channels=len(input_keys))
    load_state_dict(model, weights_path, args.device)
    model.eval()
    model.to(args.device)

    model_input = make_registration_input(batch, input_keys, args.device)
    identity_grid = make_identity_grid_m11(args.spatial_size, device=args.device)

    if args.diffeomorphic:
        ddf, grid = predict_diffeomorphic_ddf_and_grid(
            model,
            model_input,
            identity_grid,
            velocity_scale=args.velocity_scale,
            int_steps=args.int_steps,
        )
    else:
        ddf, grid = predict_ddf_and_grid(model, model_input, identity_grid)

    moving_pet = batch["fdg_pt"].to(args.device)
    moving_ct = batch["fdg_ct"].to(args.device)
    moving_mask = batch["fdg_mask"].float().to(args.device)
    fixed_pet = batch["psma_pt"].to(args.device)
    fixed_ct = batch["psma_ct"].to(args.device)
    fixed_mask = batch["psma_mask"].float().to(args.device)

    warped_moving_pet = F.grid_sample(moving_pet, grid, align_corners=True)
    warped_moving_ct = F.grid_sample(moving_ct, grid, align_corners=True)
    warped_moving_mask = F.grid_sample(
        moving_mask,
        grid,
        mode="nearest",
        align_corners=True,
    )
    view_name = "axial" if args.axial else "coronal"

    output_name = (
        f"{os.path.splitext(os.path.basename(weights_path))[0]}"
        f"_case{args.case_index}_{view_name}_6x12_with_contour.png"
    )
    output_path = os.path.join(args.image_dir, output_name)
    save_figure(
        fixed_pet=fixed_pet,
        warped_moving_pet=warped_moving_pet,
        fixed_ct=fixed_ct,
        warped_moving_ct=warped_moving_ct,
        ddf=ddf,
        fixed_mask=fixed_mask,
        warped_moving_mask=warped_moving_mask,
        output_path=output_path,
        num_slices=12,
        show_contour=True,
        title=case_name,
        grid_step=args.grid_step,
        view=view_name,
    )
    print(f"Saved image: {output_path}")

    output_name_no_contour = (
        f"{os.path.splitext(os.path.basename(weights_path))[0]}"
        f"_case{args.case_index}_{view_name}_6x12_no_contour.png"
    )
    output_path_no_contour = os.path.join(args.image_dir, output_name_no_contour)
    save_figure(
        fixed_pet=fixed_pet,
        warped_moving_pet=warped_moving_pet,
        fixed_ct=fixed_ct,
        warped_moving_ct=warped_moving_ct,
        ddf=ddf,
        fixed_mask=fixed_mask,
        warped_moving_mask=warped_moving_mask,
        output_path=output_path_no_contour,
        num_slices=12,
        show_contour=False,
        title=case_name,
        grid_step=args.grid_step,
        view=view_name,
    )
    print(f"Saved image: {output_path_no_contour}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize one baseline registration result as a 6x12 PET/CT/DDF grid."
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        required=True,
        help="Path to one checkpoint, or a directory containing .pt/.pth/.ptm files.",
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="auto",
        choices=["auto", "voxelmorph", "transmorph"],
        help="Baseline architecture. Use auto when checkpoint names include the model tag.",
    )
    parser.add_argument(
        "--data_dirs",
        type=str,
        nargs="+",
        default=DEFAULT_DATA_DIRS,
        help="H5 data directories used for train/test split.",
    )
    parser.add_argument(
        "--test_counts",
        type=int,
        nargs="+",
        default=DEFAULT_TEST_COUNTS,
        help="Number of test cases to take from each data directory.",
    )
    parser.add_argument(
        "--spatial_size",
        type=int,
        nargs=3,
        default=(128, 128, 384),
        help="Input volume size used to build the identity grid.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run on, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--case_index",
        type=int,
        default=0,
        help="Index of the test-set case to visualize.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=DEFAULT_IMAGE_DIR,
        help="Directory where the PNG will be saved.",
    )
    parser.add_argument(
        "--use_ct_input",
        action="store_true",
        help="Use [fdg_pt, fdg_ct, psma_pt, psma_ct] as model input.",
    )
    parser.add_argument(
        "--diffeomorphic",
        action="store_true",
        help="Interpret model output as SVF and integrate it with scaling-and-squaring.",
    )
    parser.add_argument(
        "--velocity_scale",
        type=float,
        default=0.5,
        help="Scale applied to tanh(model_output) before SVF integration.",
    )
    parser.add_argument(
        "--int_steps",
        type=int,
        default=7,
        help="Number of scaling-and-squaring integration steps.",
    )
    parser.add_argument(
        "--grid_step",
        type=int,
        default=8,
        help="Voxel spacing between deformation-grid lines in the visualization.",
    )
    parser.add_argument(
        "--axial",
        action="store_true",
        help="Plot axial slices instead of the default coronal slices.",
    )
    return parser.parse_args()


def main(args):
    if args.velocity_scale <= 0:
        raise ValueError("--velocity_scale must be > 0.")
    if args.int_steps < 0:
        raise ValueError("--int_steps must be >= 0.")
    if args.grid_step <= 0:
        raise ValueError("--grid_step must be > 0.")

    weights_paths = list_weight_paths(args.weights_path)
    if not weights_paths:
        raise FileNotFoundError(f"No checkpoint files found in {args.weights_path}")

    run_single_case(args, weights_paths[0])


if __name__ == "__main__":
    main(parse_args())
