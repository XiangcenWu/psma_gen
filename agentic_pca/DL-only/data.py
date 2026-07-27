#!/usr/bin/env python3
"""Dataset and conservative PET augmentations for the image-only baseline."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class PairedPETDataset(Dataset):
    """Load cached FDG/PSMA tensors without exposing text or PSA to the model."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        target_shape: Sequence[int],
        augment: bool = False,
        include_label: bool = True,
    ) -> None:
        self.records = list(records)
        self.target_shape = tuple(int(value) for value in target_shape)
        self.augment = augment
        self.include_label = include_label

    def __len__(self) -> int:
        return len(self.records)

    def _augment(self, image: torch.Tensor) -> torch.Tensor:
        # RAS axis 0 is left/right. The spatial operation is shared by both
        # tracers; intensity perturbations are tracer-specific.
        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(1,))

        for channel in range(image.shape[0]):
            if torch.rand(()) < 0.8:
                scale = 0.90 + 0.20 * torch.rand(())
                shift = -0.03 + 0.06 * torch.rand(())
                image[channel] = image[channel] * scale + shift
            if torch.rand(()) < 0.3:
                gamma = 0.85 + 0.30 * torch.rand(())
                image[channel] = image[channel].clamp(0.0, 1.0).pow(gamma)
            if torch.rand(()) < 0.2:
                image[channel] += torch.randn_like(image[channel]) * 0.01
        return image.clamp_(0.0, 1.0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = Path(record["cache_path"])
        array = np.load(path, allow_pickle=False)
        expected = (2, *self.target_shape)
        if array.shape != expected:
            raise ValueError(f"{path}: expected shape {expected}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{path}: cached tensor contains non-finite values")
        image = torch.from_numpy(array.astype(np.float32, copy=True))
        if self.augment:
            image = self._augment(image)
        item: dict[str, Any] = {
            "image": image,
            "case_id": str(record["case_id"]),
        }
        if self.include_label:
            item["label"] = torch.tensor(int(record["label_index"]), dtype=torch.long)
        return item


def seed_worker(worker_id: int) -> None:
    """Derive NumPy/Python worker RNGs from PyTorch's worker seed."""
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)

