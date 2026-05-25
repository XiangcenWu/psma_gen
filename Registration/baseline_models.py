"""
Baseline registration models for paper comparisons.

All models in this file follow the same interface used by Registration/train.py:

    input:  Tensor with shape (B, 2, D, H, W), usually concat(moving, fixed)
    output: Tensor with shape (B, 3, D, H, W), a dense field before tanh

The training code can keep using:

    ddf = torch.tanh(model(input))
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_flow_layer(layer: nn.Conv3d) -> None:
    nn.init.normal_(layer.weight, mean=0.0, std=1e-5)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class ConvBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm: bool = False,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            )
        ]
        if norm:
            layers.append(nn.InstanceNorm3d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.extend(
            [
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            ]
        )
        if norm:
            layers.append(nn.InstanceNorm3d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm: bool = False,
    ) -> None:
        super().__init__()
        self.conv = ConvBlock3D(
            in_channels + skip_channels,
            out_channels,
            norm=norm,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode="trilinear",
            align_corners=True,
        )
        return self.conv(torch.cat([x, skip], dim=1))


class VoxelMorph3D(nn.Module):
    """
    3D VoxelMorph-style convolutional U-Net baseline.

    This is the common unsupervised registration backbone: a shared encoder
    receives the moving/fixed pair and a decoder predicts a dense displacement
    field. The final layer is initialized near zero so early deformations are
    close to identity.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 3,
        enc_channels: Tuple[int, int, int, int, int] = (32, 64, 128, 256, 256),
        dec_channels: Tuple[int, int, int, int] = (256, 128, 64, 32),
        norm: bool = False,
    ) -> None:
        super().__init__()
        c0, c1, c2, c3, c4 = enc_channels
        d3, d2, d1, d0 = dec_channels

        self.enc0 = ConvBlock3D(in_channels, c0, norm=norm)
        self.enc1 = ConvBlock3D(c0, c1, stride=2, norm=norm)
        self.enc2 = ConvBlock3D(c1, c2, stride=2, norm=norm)
        self.enc3 = ConvBlock3D(c2, c3, stride=2, norm=norm)
        self.bottleneck = ConvBlock3D(c3, c4, stride=2, norm=norm)

        self.up3 = UpBlock3D(c4, c3, d3, norm=norm)
        self.up2 = UpBlock3D(d3, c2, d2, norm=norm)
        self.up1 = UpBlock3D(d2, c1, d1, norm=norm)
        self.up0 = UpBlock3D(d1, c0, d0, norm=norm)

        self.refine = ConvBlock3D(d0, d0, norm=norm)
        self.flow = nn.Conv3d(d0, out_channels, kernel_size=3, padding=1)
        _init_flow_layer(self.flow)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.enc0(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        xb = self.bottleneck(x3)

        y = self.up3(xb, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        y = self.up0(y, x0)
        y = self.refine(y)
        return self.flow(y)


class TransformerBlock3D(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, depth, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        attn_input = self.norm1(tokens)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input)
        tokens = tokens + attn_output
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(
            batch_size,
            channels,
            depth,
            height,
            width,
        )


class TransMorph3D(nn.Module):
    """
    Compact TransMorph-style 3D registration baseline.

    The original TransMorph is a hybrid Transformer-ConvNet registration model
    that uses a hierarchical Transformer encoder and a convolutional decoder.
    This implementation keeps the same interface and spirit, while using a
    bottleneck Transformer to stay lightweight for 128 x 128 x 384 volumes.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 3,
        enc_channels: Tuple[int, int, int, int, int] = (32, 64, 128, 192, 256),
        dec_channels: Tuple[int, int, int, int] = (192, 128, 64, 32),
        transformer_depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c0, c1, c2, c3, c4 = enc_channels
        d3, d2, d1, d0 = dec_channels

        self.enc0 = ConvBlock3D(in_channels, c0, norm=True)
        self.enc1 = ConvBlock3D(c0, c1, stride=2, norm=True)
        self.enc2 = ConvBlock3D(c1, c2, stride=2, norm=True)
        self.enc3 = ConvBlock3D(c2, c3, stride=2, norm=True)
        self.patch_embed = ConvBlock3D(c3, c4, stride=2, norm=True)

        self.transformer = nn.Sequential(
            *[
                TransformerBlock3D(
                    channels=c4,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(transformer_depth)
            ]
        )

        self.up3 = UpBlock3D(c4, c3, d3, norm=True)
        self.up2 = UpBlock3D(d3, c2, d2, norm=True)
        self.up1 = UpBlock3D(d2, c1, d1, norm=True)
        self.up0 = UpBlock3D(d1, c0, d0, norm=True)

        self.refine = ConvBlock3D(d0, d0, norm=True)
        self.flow = nn.Conv3d(d0, out_channels, kernel_size=3, padding=1)
        _init_flow_layer(self.flow)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.enc0(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        xb = self.patch_embed(x3)
        xb = self.transformer(xb)

        y = self.up3(xb, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        y = self.up0(y, x0)
        y = self.refine(y)
        return self.flow(y)


BASELINE_MODEL_REGISTRY = {
    "voxelmorph": VoxelMorph3D,
    "transmorph": TransMorph3D,
}


def build_baseline_model(name: str, **kwargs) -> nn.Module:
    key = name.lower().replace("-", "_")
    if key not in BASELINE_MODEL_REGISTRY:
        available = ", ".join(sorted(BASELINE_MODEL_REGISTRY))
        raise ValueError(f"Unknown baseline model '{name}'. Available: {available}")
    return BASELINE_MODEL_REGISTRY[key](**kwargs)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


if __name__ == "__main__":
    models = {}

    try:
        from monai.networks.nets import SwinUNETR

        models["swinunetr"] = SwinUNETR(
            in_channels=2,
            out_channels=3,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            downsample="mergingv2",
            use_v2=True,
        )
    except ImportError as exc:
        print(f"Skip SwinUNETR parameter count because MONAI is unavailable: {exc}")

    for name in ("voxelmorph", "transmorph"):
        models[name] = build_baseline_model(name)

    print("Model parameter counts:")
    for name, model in models.items():
        total, trainable = count_parameters(model)
        print(
            f"{name:12s} total={total:,} ({total / 1e6:.2f}M), "
            f"trainable={trainable:,} ({trainable / 1e6:.2f}M)"
        )
