#!/usr/bin/env python3
"""Query experiment IDs from the versioned ablation registry."""

from __future__ import annotations

import argparse
from pathlib import Path

from .ablation_specs import DEFAULT_REGISTRY, load_experiment_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-file", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--group")
    parser.add_argument("--index", type=int)
    parser.add_argument("--count", action="store_true")
    args = parser.parse_args()
    registry = load_experiment_registry(args.experiments_file)
    ids = [
        spec.id
        for spec in registry.experiments.values()
        if args.group is None or spec.group == args.group
    ]
    if not ids:
        raise ValueError(f"No experiments found for group {args.group!r}")
    if args.count:
        print(len(ids))
        return 0
    if args.index is not None:
        if not 0 <= args.index < len(ids):
            raise IndexError(
                f"Index {args.index} is outside [0, {len(ids)})"
            )
        print(ids[args.index])
        return 0
    print("\n".join(ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
