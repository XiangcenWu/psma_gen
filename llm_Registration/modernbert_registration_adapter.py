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


class VolumeDecoder3d(nn.Module):
    """Memory-conscious decoder for fixed 128 x 128 x 384 volumes."""

    def __init__(
        self,
        text_dim: int,
        output_channels: int = 2,
        output_size: tuple[int, int, int] = (128, 128, 384),
        base_size: tuple[int, int, int] = (8, 8, 24),
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if len(output_size) != 3 or len(base_size) != 3:
            raise ValueError("output_size and base_size must be 3D tuples.")

        self.output_size = tuple(output_size)
        self.base_size = tuple(base_size)
        self.base_channels = base_channels

        base_voxels = base_size[0] * base_size[1] * base_size[2]
        self.seed = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, base_channels * base_voxels),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            ConvNormAct3d(base_channels, base_channels),
            ConvNormAct3d(base_channels, base_channels),
            nn.Conv3d(base_channels, output_channels, kernel_size=1),
        )

    def forward(self, text_embedding: Tensor) -> Tensor:
        batch_size = text_embedding.shape[0]
        x = self.seed(text_embedding)
        x = x.view(batch_size, self.base_channels, *self.base_size)
        x = self.refine(x)
        x = F.interpolate(
            x,
            size=self.output_size,
            mode="trilinear",
            align_corners=False,
        )
        return x


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
        texts: str | Iterable[str] | None = None,
        text_embedding: Tensor | None = None,
        max_length: int = 512,
    ) -> dict[str, Tensor]:
        if (texts is None) == (text_embedding is None):
            raise ValueError("Pass exactly one of texts or text_embedding.")

        if text_embedding is None:
            text_embedding = self.encode_text(texts, max_length=max_length)
        else:
            text_embedding = text_embedding.to(self.device)

        volumes = self.decoder(text_embedding)
        regularization_logits = volumes[:, :1]
        guidance_logits = volumes[:, 1 : 1 + self.guidance_channels]

        return {
            "spatial_regularization_map": torch.sigmoid(regularization_logits),
            "registration_guidance_feature": torch.tanh(guidance_logits),
        }


def build_adapter(
    model_dir: str | Path,
    output_size: tuple[int, int, int] = (128, 128, 384),
    guidance_channels: int = 1,
    freeze_text_encoder: bool = True,
) -> ModernBERTRegistrationAdapter:
    return ModernBERTRegistrationAdapter(
        model_dir=model_dir,
        output_size=output_size,
        guidance_channels=guidance_channels,
        freeze_text_encoder=freeze_text_encoder,
    )


if __name__ == "__main__":
    adapter = ModernBERTRegistrationAdapter(model_dir="xxx")
    output = adapter("whole-body PSMA/FDG registration with liver and kidney focus")
    for name, value in output.items():
        print(name, tuple(value.shape), value.dtype)
