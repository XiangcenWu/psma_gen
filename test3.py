
import sys
import os
import argparse
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monai.networks.nets import SwinUNETR
from General.data_loader import create_data_loader, ReadH5d
import torch.nn as nn

from General.dataset_sample import split_multiple_train_test
from Registration.training import make_identity_grid_m11

from inferencing import inference_batch


train_list, test_list = split_multiple_train_test(
    ['/data1/xiangcen/data/pet_gen/processed/batch1_h5','/data1/xiangcen/data/pet_gen/processed/batch2_h5'],
    [40, 40]
)



train_list, test_list = split_multiple_train_test(
    ['/data2/xiangcen/data/pet_gen/processed/batch1_h5','/data2/xiangcen/data/pet_gen/processed/batch2_h5'],
    [40, 40]
)