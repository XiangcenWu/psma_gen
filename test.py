from Registration.inferencing import compute_tre_single, tre_surface_from_masks
import SimpleITK as sitk
import torch
from data_process.cropAintensiry import read_nii, read_nii_tensor, get_nii_spacing 
from General.segments import SEGMENT_INDEX
from Registration.inferencing import get_binary_mask_with_label

warped_fdg_mask = read_nii_tensor(read_nii(r"C:\Projects\differentHPs\baseline_l5000_k10\sample_0034\fdg_mask_warped.nii.gz"))
psma_mask = read_nii_tensor(read_nii(r"C:\Projects\differentHPs\baseline_l5000_k10\sample_0034\psma_ct_mask.nii.gz"))


fdg_spacing = get_nii_spacing(read_nii(r"C:\Projects\differentHPs\baseline_l5000_k10\sample_0034\fdg_mask_warped.nii.gz"))
psma_spacing = get_nii_spacing(read_nii(r"C:\Projects\differentHPs\baseline_l5000_k10\sample_0034\psma_ct_mask.nii.gz"))
average_spacing = (torch.tensor(fdg_spacing) + torch.tensor(psma_spacing)) / 2
print(fdg_spacing, psma_spacing)
print(average_spacing)

masks_names = list(SEGMENT_INDEX.keys())

mask_list = []
for name in masks_names:
    if name in SEGMENT_INDEX:
        mask_list.append(SEGMENT_INDEX[name])
    else:
        raise ValueError(f"Unknown segment names: {name}")
for idx, names in enumerate(masks_names):


    mask_idx = mask_list[idx]
    binary_mask_fdg = get_binary_mask_with_label(warped_fdg_mask.unsqueeze(0).unsqueeze(0), mask_idx)
    binary_mask_psma = get_binary_mask_with_label(psma_mask.unsqueeze(0).unsqueeze(0), mask_idx)
    print(f'{names}: {compute_tre_single(binary_mask_fdg, binary_mask_psma, average_spacing)}')
    print(f'{names}: {tre_surface_from_masks(binary_mask_fdg, binary_mask_psma, average_spacing)}')


