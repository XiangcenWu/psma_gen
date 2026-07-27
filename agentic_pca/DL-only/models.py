"""3D dual-tracer PET classification models.

All models in this module expect a floating-point tensor shaped
``(batch, 2, spatial_0, spatial_1, spatial_2)``.  The two channels are
intended to contain the preprocessed FDG and PSMA PET volumes.
"""

from __future__ import annotations

from typing import Any

import torch
from monai.networks.nets import DenseNet121
from monai.networks.nets.swin_unetr import SwinTransformer
from torch import nn


IN_CHANNELS = 2
SUPPORTED_MODELS = ("swin_tiny_3d", "densenet121_3d")


def _validate_num_classes(num_classes: int) -> int:
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise TypeError(f"num_classes must be an integer, got {type(num_classes).__name__}")
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}")
    return num_classes


def _validate_pet_input(
    x: torch.Tensor,
    *,
    minimum_spatial_size: int,
    spatial_divisor: int | None = None,
) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"model input must be a torch.Tensor, got {type(x).__name__}")
    if x.ndim != 5:
        raise ValueError(
            "expected input shape (batch, 2, spatial_0, spatial_1, spatial_2), "
            f"got {tuple(x.shape)}"
        )
    if x.shape[0] < 1:
        raise ValueError("the input batch dimension must be at least 1")
    if x.shape[1] != IN_CHANNELS:
        raise ValueError(
            f"expected exactly {IN_CHANNELS} input channels (FDG and PSMA), "
            f"got {x.shape[1]}"
        )
    if not x.is_floating_point():
        raise TypeError(f"PET input must be floating point, got dtype={x.dtype}")

    spatial_shape = tuple(int(size) for size in x.shape[2:])
    if any(size < minimum_spatial_size for size in spatial_shape):
        raise ValueError(
            f"each spatial dimension must be at least {minimum_spatial_size}, "
            f"got {spatial_shape}"
        )
    if spatial_divisor is not None and any(
        size % spatial_divisor != 0 for size in spatial_shape
    ):
        raise ValueError(
            "Swin input spatial dimensions must be divisible by the total "
            f"downsampling factor {spatial_divisor}, got {spatial_shape}"
        )


class SwinTiny3DClassifier(nn.Module):
    """Hierarchical 3D Swin encoder with a global classification head."""

    def __init__(
        self,
        *,
        num_classes: int = 4,
        use_checkpoint: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        num_classes = _validate_num_classes(num_classes)
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        embed_dim = 48
        patch_size = (4, 4, 4)
        window_size = (4, 4, 4)
        depths = (2, 2, 6, 2)
        num_heads = (3, 6, 12, 24)
        total_downsampling_factor = patch_size[0] * (2**len(depths))
        final_channels = embed_dim * (2 ** len(depths))

        self.encoder = SwinTransformer(
            in_chans=IN_CHANNELS,
            embed_dim=embed_dim,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=bool(use_checkpoint),
            spatial_dims=3,
            downsample="merging",
            use_v2=False,
        )
        self.norm = nn.LayerNorm(final_channels)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(final_channels, num_classes)

        self.model_config: dict[str, Any] = {
            "name": "swin_tiny_3d",
            "spatial_dims": 3,
            "in_channels": IN_CHANNELS,
            "num_classes": num_classes,
            "embed_dim": embed_dim,
            "patch_size": patch_size,
            "window_size": window_size,
            "depths": depths,
            "num_heads": num_heads,
            "mlp_ratio": 4.0,
            "drop_path_rate": 0.2,
            "patch_norm": True,
            "dropout": float(dropout),
            "use_checkpoint": bool(use_checkpoint),
            "total_downsampling_factor": total_downsampling_factor,
            "final_channels": final_channels,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_pet_input(
            x,
            minimum_spatial_size=self.model_config["total_downsampling_factor"],
            spatial_divisor=self.model_config["total_downsampling_factor"],
        )
        features = self.encoder(x, normalize=True)
        pooled = features[-1].mean(dim=(2, 3, 4))
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class DenseNet1213DClassifier(nn.Module):
    """MONAI 3D DenseNet-121 with validation for dual-tracer PET input."""

    def __init__(self, *, num_classes: int = 4) -> None:
        super().__init__()
        num_classes = _validate_num_classes(num_classes)
        self.network = DenseNet121(
            spatial_dims=3,
            in_channels=IN_CHANNELS,
            out_channels=num_classes,
            pretrained=False,
        )
        self.model_config: dict[str, Any] = {
            "name": "densenet121_3d",
            "spatial_dims": 3,
            "in_channels": IN_CHANNELS,
            "num_classes": num_classes,
            "pretrained": False,
            "use_checkpoint": False,
            "minimum_spatial_size": 32,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_pet_input(
            x,
            minimum_spatial_size=self.model_config["minimum_spatial_size"],
        )
        return self.network(x)


def build_model(
    name: str,
    num_classes: int = 4,
    use_checkpoint: bool = True,
) -> nn.Module:
    """Build a supported two-channel 3D PET classifier.

    ``use_checkpoint`` controls activation checkpointing in the Swin encoder.
    DenseNet-121 has no native MONAI checkpointing switch, so the argument is
    accepted for a uniform factory API but does not change that model.
    """

    if not isinstance(name, str):
        raise TypeError(f"model name must be a string, got {type(name).__name__}")
    normalized_name = name.strip().lower().replace("-", "_")
    num_classes = _validate_num_classes(num_classes)

    if normalized_name == "swin_tiny_3d":
        return SwinTiny3DClassifier(
            num_classes=num_classes,
            use_checkpoint=use_checkpoint,
        )
    if normalized_name == "densenet121_3d":
        return DenseNet1213DClassifier(num_classes=num_classes)

    supported = ", ".join(SUPPORTED_MODELS)
    raise ValueError(f"unknown model {name!r}; supported models: {supported}")


__all__ = [
    "DenseNet1213DClassifier",
    "IN_CHANNELS",
    "SUPPORTED_MODELS",
    "SwinTiny3DClassifier",
    "build_model",
]
