import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm_Registration.modernbert_registration_adapter import ModernBERTSwinUNETRRegistrationModel
from llm_Registration.prompt.read_basic_prompt import read_basic_prompt

from General.segments import SEGMENT_INDEX


# =====================
# Notebook-style config
# =====================
HF_MODEL_DIR = "/data2/xiangcen/hf_models"
WEIGHTS_PATH = "/data2/xiangcen/llm_regsitration_model/toy.ptm"
OUTPUT_DIR = "llm_Registration/adapter_inference_outputs"

SPATIAL_SIZE = (64, 64, 192)
PROMPT_ORGANS = 3
CHANNEL = 0
SEED = None
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_weights(model, weights_path, device):
    if not weights_path:
        return

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f">>> Loaded weights from: {weights_path}")


def plot_depth_slices(
    volume,
    save_path,
    title,
    channel=0,
    rows=6,
    cols=6,
    cmap="magma",
    robust=True,
    rotate_ccw_90=True,
):
    """
    Plot heatmap slices from a tensor shaped (B, C, H, W, D).

    For volume_cpu with shape (H, W, D), this visualizes:
        volume_cpu[:, W_idx, :]

    If rotate_ccw_90=True, each slice is rotated 90 degrees counterclockwise.

    cmap examples:
        "magma", "inferno", "viridis", "plasma", "hot"
    """
    if volume.dim() != 5:
        raise ValueError("volume must have shape (B, C, H, W, D).")

    if channel >= volume.shape[1]:
        raise ValueError(
            f"channel={channel} is out of range for volume with {volume.shape[1]} channels."
        )

    volume_cpu = volume[0, channel].detach().float().cpu()
    # volume_cpu shape: (H, W, D)

    slice_dim_size = volume_cpu.shape[1]

    slice_indices = (
        torch.linspace(0, slice_dim_size - 1, steps=rows * cols)
        .round()
        .long()
        .tolist()
    )

    if robust:
        vmin = torch.quantile(volume_cpu.flatten(), 0.01).item()
        vmax = torch.quantile(volume_cpu.flatten(), 0.99).item()
    else:
        vmin = volume_cpu.min().item()
        vmax = volume_cpu.max().item()

    if abs(vmax - vmin) < 1e-8:
        vmin = volume_cpu.min().item()
        vmax = volume_cpu.max().item() + 1e-8

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 1.4, rows * 3.2),
        constrained_layout=True,
    )

    last_im = None

    for axis, depth_idx in zip(axes.flat, slice_indices):
        slice_img = volume_cpu[:, depth_idx, :]
        # shape before rotation: (H, D)

        if rotate_ccw_90:
            # 逆时针旋转 90°
            slice_img = torch.rot90(slice_img, k=1, dims=(0, 1))

        last_im = axis.imshow(
            slice_img,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )

        axis.set_title(f"W={depth_idx}", fontsize=8)
        axis.axis("off")

    fig.colorbar(
        last_im,
        ax=axes.ravel().tolist(),
        shrink=0.65,
        pad=0.01,
    )

    fig.suptitle(title, fontsize=10)
    fig.savefig(save_path, dpi=200)
    plt.show()
    plt.close(fig)


ensure_dir(OUTPUT_DIR)

if SEED is None:
    prompt_seed = random.SystemRandom().randint(0, 2**32 - 1)
else:
    prompt_seed = SEED

rng = random.Random(prompt_seed)
prompt, labels = read_basic_prompt(organs=PROMPT_ORGANS, rng=rng)


###
prompt_list = [
    "liver, gallbladder, stomach, pancreas, duodenum",
    "kidney_left, kidney_right, adrenal_gland_left, adrenal_gland_right, urinary_bladder",
    "lung_upper_lobe_left, lung_lower_lobe_left, lung_upper_lobe_right, lung_middle_lobe_right, lung_lower_lobe_right",
    "heart, aorta, pulmonary_vein, superior_vena_cava, inferior_vena_cava",
    "brain, skull, spinal_cord, vertebrae_C1, vertebrae_C2",
    "sacrum, vertebrae_L5, vertebrae_L4, vertebrae_L3, vertebrae_L2, vertebrae_L1",
    "rib_left_1, rib_left_2, rib_left_3, rib_right_1, rib_right_2, rib_right_3, sternum",
    "hip_left, hip_right, femur_left, femur_right, sacrum",
    "humerus_left, humerus_right, scapula_left, scapula_right, clavicula_left, clavicula_right",
    "tibia, fibula, tarsal, metatarsal, phalanges_feet",
]
prompt_list = [
    # "prostate, urinary_bladder",
    # "prostate, urinary_bladder, brain",
    "brain"
]

prompt = random.choice(prompt_list)
labels = [SEGMENT_INDEX[name.strip()] for name in prompt.split(",")]
###

print(f">>> Prompt seed: {prompt_seed}")
print(f">>> Prompt: {prompt}")
print(f">>> Labels: {labels}")

model = ModernBERTSwinUNETRRegistrationModel(
    model_dir=HF_MODEL_DIR,
    spatial_size=SPATIAL_SIZE,
    image_channels=4,
    freeze_text_encoder=True,
).to(DEVICE)

load_weights(model, WEIGHTS_PATH, DEVICE)
model.eval()

with torch.no_grad():
    adapter_outputs = model.adapter(texts=[prompt])

spatial_regularization_map = adapter_outputs["spatial_regularization_map"]
registration_guidance_feature = adapter_outputs["registration_guidance_feature"]

print(f">>> spatial_regularization_map shape: {tuple(spatial_regularization_map.shape)}")
print(f">>> registration_guidance_feature shape: {tuple(registration_guidance_feature.shape)}")

plot_depth_slices(
    spatial_regularization_map,
    Path(OUTPUT_DIR) / "spatial_regularization_map_w_slices.png",
    title=prompt,
    channel=CHANNEL,
    cmap="magma",
)
plot_depth_slices(
    registration_guidance_feature,
    Path(OUTPUT_DIR) / "registration_guidance_feature_w_slices.png",
    title=prompt,
    channel=CHANNEL,
    cmap="viridis",
)

print(f">>> Outputs saved to: {OUTPUT_DIR}")