import torch
import h5py

from monai.data import (
    Dataset,
    DataLoader,
)


def read_h5_to_dict(file_name: str):

    with h5py.File(file_name, 'r') as h5_file:

        fdg_ct = torch.from_numpy(h5_file['fdg_ct'][:])

        fdg_pt = torch.from_numpy(h5_file['fdg_pt'][:])

        psma_ct = torch.from_numpy(h5_file['psma_ct'][:])

        psma_pt = torch.from_numpy(h5_file['psma_pt'][:])

    

    return {'fdg_ct': fdg_ct, 'fdg_pt': fdg_pt, 'psma_ct': psma_ct, 'psma_pt': psma_pt}



class ReadH5d():
    def __call__(self, file_name):
        return read_h5_to_dict(file_name)



def create_data_loader(data_list, transform, batch_size, drop_last=False, shuffle=True):
    set = Dataset(data_list, transform)
    return DataLoader(set, num_workers=8, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
