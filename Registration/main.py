from monai.networks.nets import SwinUNETR
from General.data_loader import create_data_loader, ReadH5d

from monai.losses import DiceLoss
from General.dataset_sample import split_train_test

from Registration.training import train_batch, make_identity_grid_m11
import torch

device='cuda:0'

loss_function = DiceLoss(
    to_onehot_y=False,   # Automatically converts class indices to one-hot
    softmax=False,       # Applies Softmax to the network output internally
    include_background=False
)

model = SwinUNETR(
    img_size = (128, 128, 384),
    in_channels = 2,
    out_channels = 3,
    depths = (2, 2, 2, 2),
    num_heads = (3, 6, 12, 24),
    downsample="mergingv2",
    use_v2=True,
)

train_transform = ReadH5d()

train_list, test_list = split_train_test('/data1/xiangcen/pet_gen/processed/batch1_h5')

train_loader = create_data_loader(train_list, train_transform)
test_loader = create_data_loader(test_list, train_transform)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

identity_grid = make_identity_grid_m11((128, 128, 384), device=device)

for b in range(500):

    loss_batch = train_batch(model, train_loader, optimizer, loss_function, identity_grid)
    print(loss_batch)

    torch.save(model.state_dict(), '/data1/xiangcen/models/registration/baseline.ptm')
