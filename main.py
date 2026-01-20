import torch
from monai.networks.nets import DiffusionModelUNet




from Training.data_loader import create_data_loader





device = 'cuda:0'


model = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
).to(device)




x = torch.rand(1, 1, 128, 128, 384).to(device)



o = model(x, 100)


print(o.shape)



