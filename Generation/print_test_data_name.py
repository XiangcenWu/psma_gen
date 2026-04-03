import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from General.dataset_sample import split_multiple_train_test


DEFAULT_DATA_DIRS = [
    "/data2/xiangcen/data/pet_gen/processed/batch1_h5_v2",
    "/data2/xiangcen/data/pet_gen/processed/batch2_h5_v2",
]
DEFAULT_VAL_COUNTS = [40, 40]
DEFAULT_SEED = 325


def main():
    data_dirs = DEFAULT_DATA_DIRS
    val_counts = DEFAULT_VAL_COUNTS
    seed = DEFAULT_SEED

    if len(data_dirs) != len(val_counts):
        raise ValueError("DEFAULT_DATA_DIRS and DEFAULT_VAL_COUNTS must have the same length")

    _, test_list = split_multiple_train_test(
        data_dirs,
        val_counts,
        seed,
    )

    print(f"Total test samples: {len(test_list)}")
    for sample_path in test_list:
        print(Path(sample_path).name)


if __name__ == "__main__":
    main()
