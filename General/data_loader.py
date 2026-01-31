import torch
import h5py

from monai.data import (
    Dataset,
    DataLoader,
)



def load_h5(file_name: str):
    """
    Returns:
        fdg_ct, fdg_pt, fdg_mask,
        psma_ct, psma_pt, psma_mask
    """

    with h5py.File(file_name, 'r') as h5_file:
        fdg_ct = torch.from_numpy(h5_file['fdg_ct'][:])
        fdg_pt = torch.from_numpy(h5_file['fdg_pt'][:])
        fdg_mask = torch.from_numpy(h5_file['fdg_mask'][:])

        psma_ct = torch.from_numpy(h5_file['psma_ct'][:])
        psma_pt = torch.from_numpy(h5_file['psma_pt'][:])
        psma_mask = torch.from_numpy(h5_file['psma_mask'][:])

        fdg_spacing = tuple(h5_file.attrs['fdg_spacing'])
        psma_spacing = tuple(h5_file.attrs['psma_spacing'])

    return (
        fdg_ct,
        fdg_pt,
        fdg_mask,
        psma_ct,
        psma_pt,
        psma_mask,
        fdg_spacing,
        psma_spacing
    )


def read_h5_to_dict(file_name: str):
    h5_file = load_h5(file_name)
    return {'fdg_ct': h5_file[0], 'fdg_pt': h5_file[1], \
            'fdg_mask': h5_file[2], \
            'psma_ct': h5_file[3], 'psma_pt': h5_file[4], \
            'psma_mask': h5_file[5], \
            'fdg_spacing': h5_file[6], 'psma_spacing': h5_file[7]}



class ReadH5d():
    def __call__(self, file_name):
        return read_h5_to_dict(file_name)



def create_data_loader(data_list, transform, batch_size=1, drop_last=False, shuffle=True):
    _dataset = Dataset(data_list, transform)
    return DataLoader(_dataset, num_workers=8, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
