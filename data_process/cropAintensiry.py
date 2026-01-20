import SimpleITK as sitk
import torch
from monai.transforms import ScaleIntensityRangePercentiles, Resize
import os
import h5py
from tqdm import tqdm

def bbox_range(arr, seg, ):
    assert arr.shape == seg.shape, "the shapes of img and seg are not equal"
    assert (seg == 0).all() == False, "the values in the seg are all zeros"
    len_x, len_y, len_z = seg.shape
    # rx, ry, rz = radius[0], radius[1], radius[2]
    for x_a in range(len_x):
        if (seg[x_a, :, :] == 0).all() != True:
            break
    for x_b in range(len_x - 1, -1, -1):
        if (seg[x_b, :, :] == 0).all() != True:
            break
    for y_a in range(len_y):
        if (seg[:, y_a, :] == 0).all() != True:
            break
    for y_b in range(len_y - 1, -1, -1):
        if (seg[:, y_b, :] == 0).all() != True:
            break
    for z_a in range(len_z):
        if (seg[:, :, z_a] == 0).all() != True:
            break
    for z_b in range(len_z - 1, -1, -1):
        if (seg[:, :, z_b] == 0).all() != True:
            break
    
    return x_a, x_b, y_a, y_b, z_a, z_b

def cropAbyB(crop_tensor: torch.tensor, crop_reference_tensor: torch.tensor) -> torch.tensor:
    x_a, x_b, y_a, y_b, z_a, z_b = bbox_range(crop_tensor, crop_reference_tensor)
    return crop_tensor[x_a:x_b, y_a:y_b, z_a:z_b]


def read_nii(file_dir):
    itk_image = sitk.ReadImage(file_dir) 
    return itk_image

def read_nii_tensor(itk_image: sitk.Image):
    img_array = sitk.GetArrayFromImage(itk_image)
    return torch.from_numpy(img_array).permute(1, 2, 0).float()


def save_h5(file_name: str,
            fdg_ct_tensor: torch.tensor,
            fdg_pt_tensor: torch.tensor,
            psma_ct_tensor: torch.tensor,
            psma_pt_tensor: torch.tensor):

    with h5py.File(file_name, 'w') as h5_file:
        h5_file.create_dataset('fdg_ct', data=fdg_ct_tensor)
        h5_file.create_dataset('fdg_pt', data=fdg_pt_tensor)
        
        h5_file.create_dataset('psma_ct', data=psma_ct_tensor)
        h5_file.create_dataset('psma_pt', data=psma_pt_tensor)
        
        
        
patient_dir = '/data/xiangcen/pet_gen/processed/batch1'
patient_dir = [os.path.join(patient_dir, patient_name) for patient_name in os.listdir(patient_dir)]


for dir in tqdm(patient_dir, desc="Processing patients", unit="patient"):
    fdg_ct = read_nii(os.path.join(dir, 'fdgCT_series2.nii.gz'))
    psma_ct = read_nii(os.path.join(dir, 'psmaCT_series2.nii.gz'))  

    fdg_pt = read_nii(os.path.join(dir, 'fdgPT_series1.nii.gz'))   
    psma_pt = read_nii(os.path.join(dir, 'psmaPT_series1.nii.gz'))

    fdg_mask = read_nii(os.path.join(dir, 'fdgCT_body_mask.nii.gz'))
    psma_mask = read_nii(os.path.join(dir, 'psmaCT_body_mask.nii.gz'))
    
    # resample PTs to CTs
    fdg_pt = sitk.Resample(fdg_pt, fdg_ct, sitk.Transform(), sitk.sitkLinear)
    psma_pt = sitk.Resample(psma_pt, psma_ct, sitk.Transform(), sitk.sitkLinear)
    
    
    fdg_ct = read_nii_tensor(fdg_ct)
    psma_ct = read_nii_tensor(psma_ct)  

    fdg_pt = read_nii_tensor(fdg_pt)   
    psma_pt = read_nii_tensor(psma_pt)

    fdg_mask = read_nii_tensor(fdg_mask)
    psma_mask = read_nii_tensor(psma_mask)


    fdg_ct = cropAbyB(fdg_ct, fdg_mask)
    fdg_pt = cropAbyB(fdg_pt, fdg_mask)
    fdg_mask = cropAbyB(fdg_mask, fdg_mask)

    psma_ct = cropAbyB(psma_ct, psma_mask)
    psma_pt = cropAbyB(psma_pt, psma_mask)
    psma_mask = cropAbyB(psma_mask, psma_mask)


    # print(fdg_ct.shape, psma_ct.shape)
    # print(fdg_ct.unique(), psma_ct.unique())
    # print(fdg_mask.unique(), psma_mask.unique())


    scaler = ScaleIntensityRangePercentiles(5, 95, 0, 1, clip=True)
    resizer = Resize((128, 128, 384), mode="trilinear")
    resizer_mask = Resize((128, 128, 384), mode="nearest-exact")



    fdg_ct = resizer(scaler(fdg_ct.unsqueeze(0)))
    fdg_pt = resizer(scaler(fdg_pt.unsqueeze(0)))


    psma_ct = resizer(scaler(psma_ct.unsqueeze(0)))
    psma_pt = resizer(scaler(psma_pt.unsqueeze(0)))
    
    
    save_h5(
        os.path.join(dir, 'data_h5.h5'),
        fdg_ct,
        fdg_pt,
        psma_ct,
        psma_pt
    )
    # print(f'{os.path.join(dir, 'data_h5.h5')} is saved')