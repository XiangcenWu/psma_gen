import torch
from monai.networks.nets import DiffusionModelUNet
import os
from Training.data_loader import create_data_loader, ReadH5d


from Training.DDPM_Baseline import *



device='cpu'

diffusion = CTtoPETDiffusion(device=device)


base_dir = '/data/xiangcen/pet_gen/processed/batch1'

patients = [os.path.join(base_dir, patient_name, 'data_h5.h5') for patient_name in os.listdir(base_dir)]
train_loader = create_data_loader(patients, ReadH5d(), batch_size=2)

for batch in train_loader:
    psma_ct_tensor = batch['psma_ct'].to(device)
    psma_pt_tensor = batch['psma_pt'].to(device)


    optimizer = optim.AdamW(diffusion.model.parameters(), lr=learning_rate)
    diffusion.train_step(psma_ct_tensor, psma_pt_tensor, optimizer)

    break

