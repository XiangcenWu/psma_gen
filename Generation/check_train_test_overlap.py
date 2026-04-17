import os
import sys
from collections import Counter
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from General.dataset_sample import split_multiple_train_test


DEFAULT_DATA_DIRS = [
    "/data2/xiangcen/data/pet_gen/processed/batch1_h5_v2",
    "/data2/xiangcen/data/pet_gen/processed/batch2_h5_v2",
]
DEFAULT_VAL_COUNTS = [40, 40]
DEFAULT_SEED = 325
MAX_PRINT = 20


def find_duplicates(items):
    counter = Counter(items)
    return sorted(item for item, count in counter.items() if count > 1)


def print_section(title, items, max_print):
    print(f"\n[{title}] count = {len(items)}")
    if not items:
        print("None")
        return

    for item in items[:max_print]:
        print(item)

    remaining = len(items) - max_print
    if remaining > 0:
        print(f"... and {remaining} more")


def main():
    data_dirs = DEFAULT_DATA_DIRS
    val_counts = DEFAULT_VAL_COUNTS
    seed = DEFAULT_SEED
    max_print = MAX_PRINT

    if len(data_dirs) != len(val_counts):
        raise ValueError("DEFAULT_DATA_DIRS and DEFAULT_VAL_COUNTS must have the same length")

    missing_dirs = [data_dir for data_dir in data_dirs if not os.path.isdir(data_dir)]
    if missing_dirs:
        raise FileNotFoundError(
            "The following data directories do not exist:\n"
            + "\n".join(missing_dirs)
        )

    train_list, test_list = split_multiple_train_test(
        data_dirs,
        val_counts,
        seed,
    )

    train_paths = sorted(set(train_list))
    test_paths = sorted(set(test_list))

    train_names = [Path(path).name for path in train_list]
    test_names = [Path(path).name for path in test_list]

    train_stems = [Path(path).stem for path in train_list]
    test_stems = [Path(path).stem for path in test_list]

    overlap_paths = sorted(set(train_paths) & set(test_paths))
    overlap_names = sorted(set(train_names) & set(test_names))
    overlap_stems = sorted(set(train_stems) & set(test_stems))

    duplicate_train_names = find_duplicates(train_names)
    duplicate_test_names = find_duplicates(test_names)

    print("=== DDPM Train/Test Overlap Check ===")
    print(f"Seed: {seed}")
    print(f"Train samples: {len(train_list)}")
    print(f"Test samples: {len(test_list)}")
    print(f"Train unique paths: {len(train_paths)}")
    print(f"Test unique paths: {len(test_paths)}")

    print_section("Exact path overlap between train and test", overlap_paths, max_print)
    print_section(
        "Filename overlap between train and test (.h5 name)",
        overlap_names,
        max_print,
    )
    print_section(
        "Case-id overlap between train and test (filename stem)",
        overlap_stems,
        max_print,
    )
    print_section(
        "Repeated filenames inside train split",
        duplicate_train_names,
        max_print,
    )
    print_section(
        "Repeated filenames inside test split",
        duplicate_test_names,
        max_print,
    )

    if overlap_paths or overlap_names or overlap_stems:
        print("\nResult: potential overlap detected. Please review the sections above.")
    else:
        print("\nResult: no overlap detected between train and test under the current split.")


if __name__ == "__main__":
    main()
