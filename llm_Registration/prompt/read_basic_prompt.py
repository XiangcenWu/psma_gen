"""Randomly sample a simple organ prompt from General.segments."""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from General.segments import SEGMENT_INDEX


def read_basic_prompt(
    organs: int = 5,
    rng: random.Random | None = None,
) -> tuple[str, list[int]]:
    """Return a random organ prompt string and corresponding integer labels."""
    if organs < 1:
        raise ValueError("organs must be >= 1.")

    segment_items = list(SEGMENT_INDEX.items())
    if organs > len(segment_items):
        raise ValueError(f"organs must be <= {len(segment_items)}.")

    rng = rng or random
    sampled_organs = rng.sample(segment_items, organs)

    prompt = ", ".join(name for name, _ in sampled_organs)
    labels = [label for _, label in sampled_organs]
    return prompt, labels


if __name__ == "__main__":
    max_prompt_organs = 5
    
    prompt_pairs = [
            read_basic_prompt(organs=max_prompt_organs)
            for _ in range(3)
        ]
    print(prompt_pairs)
