
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from General.segments import SEGMENT_INDEX
import colorsys
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
def build_label_colormap(max_label=128, background=(0,0,0)):
    colors = np.zeros((max_label + 1, 3), dtype=float)
    colors[0] = background
    golden_ratio = 0.61803398875
    h = 0.0
    for i in range(1, max_label + 1):
        h = (h + golden_ratio) % 1.0
        colors[i] = colorsys.hsv_to_rgb(h, 0.65, 0.95)
    return ListedColormap(colors)

MAX_LABEL = 128

# ✅ 只创建一次（全程复用）
cmap = build_label_colormap(MAX_LABEL, background=(0,0,0))

# ✅ 固定离散边界：每个整数label占一个bin
bounds = np.arange(-0.5, MAX_LABEL + 1.5, 1.0)  # -0.5, 0.5, 1.5, ... 128.5
norm = BoundaryNorm(bounds, cmap.N, clip=True)





def return_file_path(base_dir: str, patient: str, image: str) -> str:
    """base_dir/<sample_patient>/<image>"""
    patient_dir = "sample_" + str(patient)
    return os.path.join(base_dir, patient_dir, image)


def load_patient_viz_slices(
    base_dir: str,
    hp_name: str,
    patient: str,
    slice_idx: int | None = None,
    slice_axis: int = 1,
    return_full_volumes: bool = False,
    return_sitk_images: bool = False,
):
    """
    Loads a set of NIfTI(.nii.gz) files for visualization and returns the selected 2D slices.

    Assumes data layout:
        base_dir/hp_name/sample_<patient>/<filename>.nii.gz

    Volumes are converted to NumPy arrays with shape (z, y, x) (SimpleITK convention via GetArrayFromImage).

    slice_axis:
        0 -> slice along z (gives (y,x))
        1 -> slice along y (gives (z,x))   [matches your original indexing [:, slice_idx, :]]
        2 -> slice along x (gives (z,y))

    Returns:
        dict with:
            - "paths": dict of file paths
            - "slices": dict of 2D numpy arrays
            - optionally "volumes": dict of 3D numpy arrays
            - optionally "sitk": dict of SimpleITK images
            - "slice_idx": used slice index
            - "slice_axis": used slice axis
    """
    root = os.path.join(base_dir, hp_name)

    # ---- paths ----
    paths = {
        "warped_moving_pt": return_file_path(root, patient, "fdg_pt_warped.nii.gz"),
        "warped_moving_mask": return_file_path(root, patient, "fdg_mask_warped.nii.gz"),
        "fixed_pt": return_file_path(root, patient, "psma_pt.nii.gz"),
        "fixed_mask": return_file_path(root, patient, "psma_ct_mask.nii.gz"),
        "moving_mask": return_file_path(root, patient, "fdg_mask.nii.gz"),
        "warped_moving_ct": return_file_path(root, patient, "fdg_ct_warped.nii.gz"),
        "fixed_ct": return_file_path(root, patient, "psma_ct.nii.gz"),
        "moving_pt": return_file_path(root, patient, "fdg_pt.nii.gz"),
        "ddf": return_file_path(root, patient, "ddf.nii.gz"),
    }

    # sanity check (optional but helpful)
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        missing_str = "\n".join([f"- {k}: {paths[k]}" for k in missing])
        raise FileNotFoundError(f"Missing files:\n{missing_str}")

    # ---- read sitk ----
    sitk_imgs = {k: sitk.ReadImage(p) for k, p in paths.items()}

    # ---- to numpy ----
    vols = {k: sitk.GetArrayFromImage(img) for k, img in sitk_imgs.items()}

    # ddf can be (z,y,x,3) or (z,y,x,?,?) depending on your save; we just keep it as-is
    fixed_vol = vols["fixed_pt"]

    # ---- choose slice ----
    if slice_idx is None:
        # default: middle slice along slice_axis
        slice_idx = fixed_vol.shape[slice_axis] // 2

    def take_slice(arr: np.ndarray) -> np.ndarray:
        # Works for 3D volumes; for 4D (e.g. ddf), we slice the spatial axis and keep remaining dims.
        slicer = [slice(None)] * arr.ndim
        slicer[slice_axis] = slice_idx
        return arr[tuple(slicer)]

    slices = {
        "wm_slice": take_slice(vols["warped_moving_pt"]),
        "wm_mask_slice": take_slice(vols["warped_moving_mask"]),
        "fixed_slice": take_slice(vols["fixed_pt"]),
        "fixed_mask_slice": take_slice(vols["fixed_mask"]),
        "wm_ct_slice": take_slice(vols["warped_moving_ct"]),
        "fixed_ct_slice": take_slice(vols["fixed_ct"]),
        "moving_pt_slice": take_slice(vols["moving_pt"]),
        "moving_mask_slice": take_slice(vols["moving_mask"]),
        "ddf_slice": take_slice(vols["ddf"]),
    }

    out = {
        "paths": paths,
        "slices": slices,
        "slice_idx": slice_idx,
        "slice_axis": slice_axis,
    }

    if return_full_volumes:
        out["volumes"] = vols
    if return_sitk_images:
        out["sitk"] = sitk_imgs

    return out

sidx = 100
bl = '4500'
data = load_patient_viz_slices(
    base_dir= r"C:\Users\Sam\Downloads\viz",
    hp_name=fr"ctsmoothness_l{bl}_k10_mar3000_gam2.0",
    patient="0004",
    slice_idx=sidx,      # or None for middle
    slice_axis=1,      # axis=1 gives [:, slice_idx, :]
    return_full_volumes=False,
)
# data = load_patient_viz_slices(
#     base_dir=r"C:\Users\Sam\Downloads\viz",
#     hp_name=fr"baseline_l{bl}_k10",
#     patient="0004",
#     slice_idx=sidx,      # or None for middle
#     slice_axis=1,      # axis=1 gives [:, slice_idx, :]
#     return_full_volumes=False,
# )
wm_slice = data["slices"]["wm_slice"]
fixed_slice = data["slices"]["fixed_slice"]
ddf_slice = data["slices"]["ddf_slice"]
moving_pt_slice = data["slices"]["moving_pt_slice"]
moving_mask_slice = data["slices"]["moving_mask_slice"]
fixed_mask_slice = data["slices"]["fixed_mask_slice"]
wm_mask_slice = data["slices"]["wm_mask_slice"]
ddf_slice = data["slices"]["ddf_slice"]

import numpy as np
import matplotlib.pyplot as plt

def ddf_rgb_clip_rescale(ddf, clip=0.2, component_order=(0, 1, 2)):
    """
    ddf: (H,W,3)
    clip: float -> clip to [-clip, clip]
    component_order: map channels to (R,G,B). Default assumes ch0,ch1,ch2.
    """
    ddf = np.asarray(ddf, dtype=np.float32)
    if ddf.ndim != 3 or ddf.shape[-1] != 3:
        raise ValueError(f"Expected (H,W,3), got {ddf.shape}")

    ddf = ddf[..., list(component_order)]          # reorder if needed
    ddf = np.clip(ddf, -clip, clip)               # clip to [-clip, clip]
    rgb = (ddf + clip) / (2.0 * clip)             # rescale to [0,1]
    return np.clip(rgb, 0.0, 1.0)


ddf_slice = ddf_rgb_clip_rescale(ddf_slice, clip=0.1, component_order=(1, 0, 2))




# 你要存到哪里
out_dir = r"C:\Users\Sam\Downloads\export_proposed"
os.makedirs(out_dir, exist_ok=True)

# 统一旋转（你原本是k=2）
fixed_img        = np.rot90(fixed_slice, k=2)
wm_img           = np.rot90(wm_slice, k=2)
moving_pt_img    = np.rot90(moving_pt_slice, k=2)
moving_mask_img  = np.rot90(moving_mask_slice, k=2)
fixed_mask_img   = np.rot90(fixed_mask_slice, k=2)
wm_mask_img      = np.rot90(wm_mask_slice, k=2)
ddf_img          = np.rot90(ddf_slice, k=1)[:, ::-1, :]   # 你的ddf处理

# 每一张图的配置（name, image, kwargs）
items = [
    ("01_fixed_pet", fixed_img, dict(cmap="gray_r")),
    ("02_warped_moving_pet", wm_img, dict(cmap="gray_r")),
    ("03_moving_pt", moving_pt_img, dict(cmap="gray_r")),
    ("04_moving_mask", moving_mask_img, dict(cmap=cmap, norm=norm, interpolation="nearest")),
    # overlay: 固定图 + mask
    ("05_fixed_mask",  fixed_mask_img,  dict(cmap=cmap, norm=norm, interpolation="nearest")),  # ✅ 新增
    ("05_fixed_overlay", (fixed_img, fixed_mask_img), dict()),
    ("06_warped_moving_mask", wm_mask_img, dict(cmap=cmap, norm=norm, interpolation="nearest")),
    ("07_ddf_rgb", ddf_img, dict()),  # ddf_img 已经是RGB (H,W,3)
]

for name, img, kwargs in items:
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = plt.axes([0, 0, 1, 1])  # 填满整张图
    ax.set_axis_off()

    if name == "05_fixed_overlay":
        base, mask = img
        ax.imshow(base, cmap="gray_r")
        ax.imshow(mask, cmap=cmap, norm=norm, interpolation="nearest", alpha=0.6)
    else:
        ax.imshow(img, **kwargs)

    save_path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

print("Saved to:", out_dir)

# fig, axes = plt.subplots(1, 7, figsize=(12, 6))  # 1 row, 6 columns

# # Column 1: Fixed PET
# axes[0].imshow(np.rot90(fixed_slice, k=2), cmap="gray_r")
# axes[0].set_title("Fixed PET", fontsize=8)
# axes[0].axis("off")

# # Column 2: Warped Moving PET
# axes[1].imshow(np.rot90(wm_slice, k=2), cmap="gray_r")
# axes[1].set_title("Warped Moving PET", fontsize=8)
# axes[1].axis("off")

# # Column 3: Fixed CT
# axes[2].imshow(np.rot90(moving_pt_slice, k=2), cmap="gray_r")
# axes[2].set_title("moving pt", fontsize=8)
# axes[2].axis("off")

# # Column 4: Warped Moving CT
# axes[3].imshow(np.rot90(moving_mask_slice, k=2), cmap=cmap, norm=norm, interpolation="nearest")
# axes[3].set_title("Moving mask", fontsize=8)
# axes[3].axis("off")

# # Column 5: Fixed PET Mask (overlay)
# axes[4].imshow(np.rot90(fixed_slice, k=2), cmap="gray")
# axes[4].imshow(np.rot90(fixed_mask_slice, k=2), cmap=cmap, norm=norm, interpolation="nearest")
# axes[4].set_title("Fixed Mask", fontsize=8)
# axes[4].axis("off")

# # Column 6: Warped Moving CT Mask (overlay)
# axes[5].imshow(np.rot90(wm_mask_slice, k=2), cmap=cmap, norm=norm, interpolation="nearest")
# axes[5].set_title("Warped Moving Mask", fontsize=8)
# axes[5].axis("off")


# # Column 7: Warped Moving CT Mask (overlay)
# axes[6].imshow(np.rot90(ddf_slice, k=1)[:, ::-1, :])
# axes[6].set_title("ddf", fontsize=8)
# axes[6].axis("off")

# plt.tight_layout()
# plt.show()