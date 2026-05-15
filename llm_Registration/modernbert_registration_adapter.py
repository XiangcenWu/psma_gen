"""Adapter from BioClinical ModernBERT text embeddings to registration volumes.

The adapter intentionally produces only two global volumes, without separate
moving/fixed branches:

1. ``spatial_regularization_map``: (B, 1, 128, 128, 384), sigmoid range.
2. ``registration_guidance_feature``: (B, C, 128, 128, 384), tanh range.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR
from transformers import AutoModel, AutoTokenizer


class ConvNormAct3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ResidualBlock3d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct3d(channels, channels),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels, affine=True),
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.block(x))


class UpBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = ConvNormAct3d(in_channels, out_channels)
        self.refine = nn.Sequential(
            ResidualBlock3d(out_channels),
            ResidualBlock3d(out_channels),
        )

    def forward(self, x: Tensor, size: tuple[int, int, int]) -> Tensor:
        x = F.interpolate(
            x,
            size=size,
            mode="trilinear",
            align_corners=False,
        )
        x = self.proj(x)
        return self.refine(x)


class VolumeDecoder3d(nn.Module):
    """Progressive 3D decoder from text embedding to registration volumes."""

    def __init__(
        self,
        text_dim: int,
        output_channels: int = 2,
        output_size: tuple[int, int, int] = (128, 128, 384),
        base_size: tuple[int, int, int] = (8, 8, 24),
        channels: tuple[int, int, int, int] = (128, 96, 64, 32),
    ) -> None:
        super().__init__()
        if len(output_size) != 3 or len(base_size) != 3:
            raise ValueError("output_size and base_size must be 3D tuples.")
        if len(channels) < 2:
            raise ValueError("channels must contain at least two stages.")

        self.output_size = tuple(output_size)
        self.base_size = tuple(base_size)
        self.channels = tuple(channels)

        base_voxels = base_size[0] * base_size[1] * base_size[2]
        self.seed = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, channels[0] * base_voxels),
            nn.GELU(),
        )
        self.stem = nn.Sequential(
            ResidualBlock3d(channels[0]),
            ResidualBlock3d(channels[0]),
        )
        self.up_blocks = nn.ModuleList(
            UpBlock3d(in_ch, out_ch)
            for in_ch, out_ch in zip(channels[:-1], channels[1:])
        )
        self.head = nn.Sequential(
            ConvNormAct3d(channels[-1], channels[-1]),
            nn.Conv3d(channels[-1], output_channels, kernel_size=1),
        )

    def _stage_sizes(self) -> list[tuple[int, int, int]]:
        num_blocks = len(self.up_blocks)
        sizes = []
        for stage_idx in range(1, num_blocks + 1):
            remaining = num_blocks - stage_idx
            size = tuple(
                min(out_dim, max(base_dim, out_dim // (2**remaining)))
                for base_dim, out_dim in zip(self.base_size, self.output_size)
            )
            sizes.append(size)
        sizes[-1] = self.output_size
        return sizes

    def forward(self, text_embedding: Tensor) -> Tensor:
        batch_size = text_embedding.shape[0]
        x = self.seed(text_embedding)
        x = x.view(batch_size, self.channels[0], *self.base_size)
        x = self.stem(x)

        for up_block, size in zip(self.up_blocks, self._stage_sizes()):
            x = up_block(x, size)

        return self.head(x)


class ModernBERTRegistrationAdapter(nn.Module):
    """Generate registration-conditioning volumes from clinical text."""

    def __init__(
        self,
        model_dir: str | Path,
        output_size: tuple[int, int, int] = (128, 128, 384),
        guidance_channels: int = 1,
        freeze_text_encoder: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            trust_remote_code=trust_remote_code,
        )
        self.text_encoder = AutoModel.from_pretrained(
            self.model_dir,
            trust_remote_code=trust_remote_code,
        )

        if freeze_text_encoder:
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()

        hidden_size = self.text_encoder.config.hidden_size
        self.decoder = VolumeDecoder3d(
            text_dim=hidden_size,
            output_channels=1 + guidance_channels,
            output_size=output_size,
        )
        self.guidance_channels = guidance_channels

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_text(
        self,
        texts: str | Iterable[str],
        max_length: int = 512,
    ) -> Tensor:
        if isinstance(texts, str):
            texts = [texts]

        tokens = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}

        with torch.set_grad_enabled(self.text_encoder.training):
            outputs = self.text_encoder(**tokens)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output

        last_hidden = outputs.last_hidden_state
        attention_mask = tokens["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * attention_mask).sum(dim=1)
        denom = attention_mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def forward(
        self,
        texts: str | Iterable[str],
        max_length: int = 512,
    ) -> dict[str, Tensor]:
        if texts is None:
            raise ValueError("texts must be provided.")

        text_embedding = self.encode_text(texts, max_length=max_length)
        volumes = self.decoder(text_embedding)
        regularization_logits = volumes[:, :1]
        guidance_logits = volumes[:, 1 : 1 + self.guidance_channels]

        return {
            "spatial_regularization_map": torch.sigmoid(regularization_logits),
            "registration_guidance_feature": torch.tanh(guidance_logits),
        }


class ModernBERTSwinUNETRRegistrationModel(nn.Module):
    """Text-guided registration model with ModernBERT adapter and SwinUNETR.

    The adapter produces two volumes with the same spatial size as moving/fixed:
    - registration_guidance_feature: concatenated with moving/fixed for SwinUNETR.
    - spatial_regularization_map: returned for use as a spatial regularization term.
    """

    def __init__(
        self,
        model_dir: str | Path,
        spatial_size: tuple[int, int, int] = (64, 64, 192),
        image_channels: int = 4,
        out_channels: int = 3,
        freeze_text_encoder: bool = True,
        trust_remote_code: bool = False,
        depths: tuple[int, int, int, int] = (2, 2, 2, 2),
        num_heads: tuple[int, int, int, int] = (3, 6, 12, 24),
    ) -> None:
        super().__init__()
        self.spatial_size = spatial_size
        self.adapter = ModernBERTRegistrationAdapter(
            model_dir=model_dir,
            output_size=spatial_size,
            guidance_channels=1,
            freeze_text_encoder=freeze_text_encoder,
            trust_remote_code=trust_remote_code,
        )
        self.registration_net = SwinUNETR(
            in_channels=image_channels + 1,  # four image channels + one guidance channel
            out_channels=out_channels,
            depths=depths,
            num_heads=num_heads,
            downsample="mergingv2",
            use_v2=True,
        )

    def forward(
        self,
        moving: Tensor,
        fixed: Tensor,
        texts: str | Iterable[str],
        max_length: int = 512,
    ) -> dict[str, Tensor]:
        text_batch = [texts] if isinstance(texts, str) else list(texts)
        if moving.shape[0] != fixed.shape[0] or len(text_batch) != moving.shape[0]:
            raise ValueError(
                "Batch size mismatch: moving, fixed, and texts must have the same batch size."
            )

        llm_outputs = self.adapter(
            texts=text_batch,
            max_length=max_length,
        )
        regularization_map = llm_outputs["spatial_regularization_map"]
        guidance_feature = llm_outputs["registration_guidance_feature"]

        registration_input = torch.cat(
            [moving, fixed, guidance_feature],
            dim=1,
        )
        ddf = self.registration_net(registration_input)

        return {
            "ddf": ddf,
            "spatial_regularization_map": regularization_map,
        }



if __name__ == "__main__":
    model = ModernBERTSwinUNETRRegistrationModel(
        model_dir="/data2/xiangcen/hf_models",
        spatial_size=(64, 64, 192),
    ).to("cuda:0")
    moving = torch.randn(1, 1, 64, 64, 192, device="cuda:0")
    fixed = torch.randn(1, 1, 64, 64, 192, device="cuda:0")
    output = model(
        moving=moving,
        fixed=fixed,
        texts="whole-body PSMA/FDG registration with liver and kidney focus",
    )
    for name, value in output.items():
        print(name, tuple(value.shape), value.dtype)
