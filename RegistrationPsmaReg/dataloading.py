import random
from collections import defaultdict
from pathlib import Path
import h5py
import torch


DATA_KEYS = ("ct", "pet", "ct_label", "pet_label", "body_label")
LABEL_KEYS = ("ct_label", "pet_label", "body_label")


def get_train_test_h5_lists(root_dir="PSMAReg_h5", test_ratio=0.2, seed=20240524):
    """
    Return stratified train/test H5 file lists.

    Stratification is based on the parent folder name, e.g.:
        PSMAReg_h5/2timepoints/PSMARegPSMA_0049.h5
        PSMAReg_h5/3timepoints/PSMARegPSMA_0006.h5

    Returns:
        train_list, test_list
    """
    root_dir = Path(root_dir)
    h5_files = sorted(root_dir.rglob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found under {root_dir}")

    files_by_group = defaultdict(list)
    for file_path in h5_files:
        files_by_group[file_path.parent.name].append(str(file_path))

    rng = random.Random(seed)
    train_list = []
    test_list = []

    for group in sorted(files_by_group):
        group_files = files_by_group[group]
        rng.shuffle(group_files)

        if len(group_files) <= 1:
            num_test = 0
        else:
            num_test = max(1, round(len(group_files) * test_ratio))

        test_list.extend(group_files[:num_test])
        train_list.extend(group_files[num_test:])

    return sorted(train_list), sorted(test_list)


def _timepoint_from_key(key):
    prefix = key.split("_", 1)[0]
    if prefix.isdigit():
        return int(prefix)
    return None


def _get_timepoints(h5_file):
    timepoints = set()
    for key in h5_file.keys():
        timepoint = _timepoint_from_key(key)
        if timepoint is not None:
            timepoints.add(timepoint)
    return sorted(timepoints)


def _sample_adjacent_timepoints(timepoints, rng):
    if len(timepoints) < 2:
        raise ValueError("At least two timepoints are required.")

    adjacent_pairs = list(zip(timepoints[:-1], timepoints[1:]))
    return rng.choice(adjacent_pairs)


def _get_adjacent_timepoint_pairs(timepoints):
    if len(timepoints) < 2:
        raise ValueError("At least two timepoints are required.")
    return list(zip(timepoints[:-1], timepoints[1:]))


def _get_all_timepoint_pairs(timepoints):
    if len(timepoints) < 2:
        raise ValueError("At least two timepoints are required.")

    pairs = []
    for moving_index, moving_timepoint in enumerate(timepoints[:-1]):
        for fixed_timepoint in timepoints[moving_index + 1:]:
            pairs.append((moving_timepoint, fixed_timepoint))
    return pairs


def _read_tensor(h5_file, dataset_name):

    tensor = torch.from_numpy(h5_file[dataset_name][:])
    if dataset_name.endswith(LABEL_KEYS):
        return tensor.to(torch.int16)
    return tensor.float()


def _read_spacing(h5_file, timepoint):
    attr_name = f"{timepoint}_spacing"
    if attr_name in h5_file.attrs:
        return tuple(h5_file.attrs[attr_name])
    return tuple(h5_file[f"{timepoint}_ct"].attrs["spacing"])


def _load_timepoint_pair_h5(h5_file, file_name, moving_timepoint, fixed_timepoint):
    data = {
        "file_name": str(file_name),
        "pair_name": f"{Path(file_name).stem}_t{moving_timepoint}_to_t{fixed_timepoint}",
        "moving_timepoint": moving_timepoint,
        "fixed_timepoint": fixed_timepoint,
        "moving_spacing": _read_spacing(h5_file, moving_timepoint),
        "fixed_spacing": _read_spacing(h5_file, fixed_timepoint),
    }

    for role, timepoint in (
        ("moving", moving_timepoint),
        ("fixed", fixed_timepoint),
    ):
        for key in DATA_KEYS:
            data[f"{role}_{key}"] = _read_tensor(h5_file, f"{timepoint}_{key}")

    return data


def load_random_timepoint_pair_h5(file_name, seed=None):
    """
    Load one random adjacent timepoint pair from a patient H5 file.

    The earlier timepoint is returned as moving.
    The later timepoint is returned as fixed.

    Returns:
        dict with moving/fixed ct, pet, ct_label, pet_label, body_label,
        spacing, timepoint indices, and file_name.
    """

    rng = random.Random(seed) if seed is not None else random

    with h5py.File(file_name, "r") as h5_file:
        timepoints = _get_timepoints(h5_file)
        moving_timepoint, fixed_timepoint = _sample_adjacent_timepoints(timepoints, rng)

        data = _load_timepoint_pair_h5(
            h5_file,
            file_name,
            moving_timepoint,
            fixed_timepoint,
        )


    return data


def load_all_adjacent_timepoint_pairs_h5(file_name):
    """
    Load all adjacent timepoint pairs from a patient H5 file.

    For a case with timepoints [1, 2, 3, 4], this returns:
        1->2, 2->3, 3->4

    Returns:
        list[dict], where every dict has the same keys as
        load_random_timepoint_pair_h5.
    """
    with h5py.File(file_name, "r") as h5_file:
        timepoints = _get_timepoints(h5_file)
        adjacent_pairs = _get_adjacent_timepoint_pairs(timepoints)
        return [
            _load_timepoint_pair_h5(
                h5_file,
                file_name,
                moving_timepoint,
                fixed_timepoint,
            )
            for moving_timepoint, fixed_timepoint in adjacent_pairs
        ]


def load_all_timepoint_pairs_h5(file_name):
    """
    Load all earlier-to-later timepoint pairs from a patient H5 file.

    For a case with timepoints [1, 2, 3, 4], this returns:
        1->2, 1->3, 1->4, 2->3, 2->4, 3->4

    A case with 7 timepoints returns 21 pairs. The moving timepoint is always
    earlier than the fixed timepoint.

    Returns:
        list[dict], where every dict has the same keys as
        load_random_timepoint_pair_h5.
    """
    with h5py.File(file_name, "r") as h5_file:
        timepoints = _get_timepoints(h5_file)
        timepoint_pairs = _get_all_timepoint_pairs(timepoints)
        return [
            _load_timepoint_pair_h5(
                h5_file,
                file_name,
                moving_timepoint,
                fixed_timepoint,
            )
            for moving_timepoint, fixed_timepoint in timepoint_pairs
        ]


class ReadH5PsmaRegd():
    def __init__(self, pair_mode="random_adjacent", seed=None):
        self.pair_mode = pair_mode
        self.seed = seed

    def __call__(self, file_name):
        if self.pair_mode == "random_adjacent":
            return load_random_timepoint_pair_h5(file_name, seed=self.seed)
        if self.pair_mode == "all_pairs":
            return load_all_timepoint_pairs_h5(file_name)
        if self.pair_mode == "all_adjacent":
            return load_all_adjacent_timepoint_pairs_h5(file_name)

        raise ValueError(
            "Unsupported pair_mode. Use 'random_adjacent', 'all_adjacent', "
            "or 'all_pairs'."
        )


if __name__ == "__main__":
    train_files, test_files = get_train_test_h5_lists()
    print(f"Train files ({len(train_files)}): {train_files}")
    print(f"Test files ({len(test_files)}): {test_files}")
    
    
    data = load_random_timepoint_pair_h5(train_files[0])
    print(data.keys())
    pairs = load_all_adjacent_timepoint_pairs_h5(train_files[0])
    print(f"All adjacent pairs: {len(pairs)}")
    pairs = load_all_timepoint_pairs_h5(train_files[0])
    print(f"All earlier-to-later pairs: {len(pairs)}")


