import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm_Registration.modernbert_registration_adapter import ModernBERTSwinUNETRRegistrationModel
from llm_Registration.prompt.read_basic_prompt import read_basic_prompt


# =====================
# Notebook-style config
# =====================
HF_MODEL_DIR = "/data2/xiangcen/hf_models"
WEIGHTS_PATH = ""  # e.g. "/data1/xiangcen/models/registration_v2/model.ptm"; leave empty to skip
OUTPUT_DIR = "llm_Registration/adapter_inference_outputs"

SPATIAL_SIZE = (64, 64, 192)
PROMPT_ORGANS = 5
CHANNEL = 0
SEED = 0
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_weights(model, weights_path, device):
    if not weights_path:
        return

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f">>> Loaded weights from: {weights_path}")


def plot_depth_slices(volume, save_path, title, channel=0, rows=6, cols=6):
    """
    Plot depth slices from a tensor shaped (B, C, H, W, D).
    """
    if volume.dim() != 5:
        raise ValueError("volume must have shape (B, C, H, W, D).")
    if channel >= volume.shape[1]:
        raise ValueError(f"channel={channel} is out of range for volume with {volume.shape[1]} channels.")

    volume_cpu = volume[0, channel].detach().float().cpu()
    depth = volume_cpu.shape[-1]
    slice_indices = torch.linspace(0, depth - 1, steps=rows * cols).round().long().tolist()

    vmin = float(volume_cpu.min())
    vmax = float(volume_cpu.max())

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2), constrained_layout=True)
    for axis, depth_idx in zip(axes.flat, slice_indices):
        axis.imshow(volume_cpu[:, :, depth_idx], cmap="viridis", vmin=vmin, vmax=vmax)
        axis.set_title(f"D={depth_idx}", fontsize=8)
        axis.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.savefig(save_path, dpi=200)
    plt.show()
    plt.close(fig)


ensure_dir(OUTPUT_DIR)

rng = random.Random(SEED)
prompt, labels = read_basic_prompt(organs=PROMPT_ORGANS, rng=rng)
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

torch.save(
    {
        "prompt": prompt,
        "labels": labels,
        "spatial_regularization_map": spatial_regularization_map.cpu(),
        "registration_guidance_feature": registration_guidance_feature.cpu(),
    },
    Path(OUTPUT_DIR) / "adapter_outputs.pt",
)

plot_depth_slices(
    spatial_regularization_map,
    Path(OUTPUT_DIR) / "spatial_regularization_map_depth.png",
    title="spatial_regularization_map",
    channel=CHANNEL,
)
plot_depth_slices(
    registration_guidance_feature,
    Path(OUTPUT_DIR) / "registration_guidance_feature_depth.png",
    title="registration_guidance_feature",
    channel=CHANNEL,
)

print(f">>> Outputs saved to: {OUTPUT_DIR}")
