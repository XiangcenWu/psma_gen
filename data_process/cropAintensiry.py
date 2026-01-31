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
    """
    Returns tensor in (x, y, z) order to match SimpleITK spacing
    """
    img_array = sitk.GetArrayFromImage(itk_image)
    return torch.from_numpy(img_array).permute(2, 1, 0).float()

def get_nii_spacing(itk_image: sitk.Image):
    return tuple(itk_image.GetSpacing())


def save_h5(file_name: str,
            fdg_ct_tensor: torch.tensor,
            fdg_pt_tensor: torch.tensor,
            fdg_mask_tensor: torch.tensor,
            psma_ct_tensor: torch.tensor,
            psma_pt_tensor: torch.tensor,
            psma_mask_tensor: torch.tensor,
            ):

    with h5py.File(file_name, 'w') as h5_file:
        h5_file.create_dataset('fdg_ct', data=fdg_ct_tensor)
        h5_file.create_dataset('fdg_pt', data=fdg_pt_tensor)
        h5_file.create_dataset('fdg_mask', data=fdg_mask_tensor)


        h5_file.create_dataset('psma_ct', data=psma_ct_tensor)
        h5_file.create_dataset('psma_pt', data=psma_pt_tensor)
        h5_file.create_dataset('psma_mask', data=psma_mask_tensor)
        

        
def fusion_ts_mask(total_mask_tensor, appendicular_bones_mask_tensor):
    assert total_mask_tensor.shape == appendicular_bones_mask_tensor.shape

    invalid = (appendicular_bones_mask_tensor > 11)
    if invalid.any():
        raise ValueError("appendicular_bones_mask_tensor contains labels > 11")
    # ---- 1. dtype 检查 & 统一 ----
    # if total_mask_tensor.dtype != torch.int16:
    total_mask_tensor = total_mask_tensor.to(torch.int16)

# if appendicular_bones_mask_tensor.dtype != torch.int16:
    appendicular_bones_mask_tensor = appendicular_bones_mask_tensor.to(torch.int16)

    # ---- 2. 复制 total 作为输出 ----
    fused = total_mask_tensor.clone()

    # ---- 3. remap appendicular bones: 1–11 -> 118–128 ----
    app = appendicular_bones_mask_tensor

    mask = (app >= 1) & (app <= 11)
    remapped_app = torch.zeros_like(app, dtype=torch.int16)
    remapped_app[mask] = app[mask] + 117

    # ---- 4. 融合（appendicular 覆盖 total）----
    fused[mask] = remapped_app[mask]

    return fused




def compute_new_voxel_dimension(
        old_spacing,   # (sx, sy, sz)
        old_size,      # (nx, ny, nz)
        new_size       # (nx', ny', nz')
    ):
    """
    Calculate new voxel dimensions after resizing,
    assuming physical size (FOV) is preserved.

    Returns:
        new_spacing: (sx', sy', sz')
    """
    new_spacing = tuple(
        old_spacing[i] * old_size[i] / new_size[i]
        for i in range(3)
    )
    return new_spacing



def crop_and_intensity(patient_dir,
                    save_dir, img_size=(128, 128, 384)):


    patient_dir = [os.path.join(patient_dir, patient_name) for patient_name in os.listdir(patient_dir)]

    
    for idx, dir in enumerate(tqdm(patient_dir, desc="cropping and intensity range", unit="case")):

        patient_name = f"patient_{idx:04d}.h5"
        
        # load all ct and pt images
        fdg_ct = read_nii(os.path.join(dir, 'fdgCT_series2.nii.gz'))
        psma_ct = read_nii(os.path.join(dir, 'psmaCT_series2.nii.gz'))  

        fdg_pt = read_nii(os.path.join(dir, 'fdgPT_series1.nii.gz'))   
        psma_pt = read_nii(os.path.join(dir, 'psmaPT_series1.nii.gz'))



        # get the voxel dimension of psma_ct and fdg_ct
        psma_ct_spacing = get_nii_spacing(psma_ct)
        fdg_ct_spacing = get_nii_spacing(fdg_ct)


        # get the maks for both fdg and psma
        fdg_total_mask = read_nii(os.path.join(dir, 'fdgCT_total_mask.nii.gz'))
        psma_total_mask = read_nii(os.path.join(dir, 'psmaCT_total_mask.nii.gz'))
        fdg_app_mask = read_nii(os.path.join(dir, 'fdgCT_appendicular_bones_mask.nii.gz'))
        psma_app_mask = read_nii(os.path.join(dir, 'psmaCT_appendicular_bones_mask.nii.gz'))
        fdg_mask = fusion_ts_mask(read_nii_tensor(fdg_total_mask), read_nii_tensor(fdg_app_mask))
        psma_mask = fusion_ts_mask(read_nii_tensor(psma_total_mask), read_nii_tensor(psma_app_mask))
        
        # resample PTs to CTs
        fdg_pt = sitk.Resample(fdg_pt, fdg_ct, sitk.Transform(), sitk.sitkLinear)
        psma_pt = sitk.Resample(psma_pt, psma_ct, sitk.Transform(), sitk.sitkLinear)
        # get the voxel dimension of fdg and psma 
        # fdg_ct, fdt_pt / psma_ct, psma_pt
        # spacing should be the same as it has been resampled
        psma_pt_spacing = get_nii_spacing(psma_pt)
        fdg_pt_spacing = get_nii_spacing(fdg_pt)
        # print to check
        print(psma_pt_spacing, psma_ct_spacing, fdg_pt_spacing, fdg_ct_spacing)

       

        # load as pytorch tensor
        fdg_ct = read_nii_tensor(fdg_ct)
        psma_ct = read_nii_tensor(psma_ct)  
        fdg_pt = read_nii_tensor(fdg_pt)
        psma_pt = read_nii_tensor(psma_pt)

        # get the size of the original tensor
        fdg_ct_size = tuple(fdg_ct.shape)
        psma_ct_size = tuple(psma_ct.shape)
        fdg_pt_size = tuple(fdg_pt.shape)
        psma_pt_size= tuple(psma_pt.shape)



        # cropping by mask
        fdg_ct = cropAbyB(fdg_ct, fdg_mask)
        fdg_pt = cropAbyB(fdg_pt, fdg_mask)
        fdg_mask = cropAbyB(fdg_mask, fdg_mask)

        psma_ct = cropAbyB(psma_ct, psma_mask)
        psma_pt = cropAbyB(psma_pt, psma_mask)
        psma_mask = cropAbyB(psma_mask, psma_mask)

        cropped_psma_size = tuple(fdg_ct.shape)
        cropped_fdg_size = tuple(fdg_ct.shape)

        psma_spacing = compute_new_voxel_dimension(
            psma_ct_spacing,
            psma_ct_size,
            cropped_psma_size
        )

        fdg_spacing = compute_new_voxel_dimension(
            fdg_ct_spacing,
            fdg_ct_size,
            cropped_fdg_size
        )

        print(psma_spacing, fdg_spacing)



        # scaler = ScaleIntensityRangePercentiles(5, 95, 0, 1, clip=True)
        # resizer = Resize(img_size, mode="trilinear")
        # resizer_mask = Resize(img_size, mode="nearest-exact")



        # fdg_ct = resizer(scaler(fdg_ct.unsqueeze(0)))
        # fdg_pt = resizer(scaler(fdg_pt.unsqueeze(0)))
        # fdg_mask = resizer_mask(fdg_mask.unsqueeze(0))


        # psma_ct = resizer(scaler(psma_ct.unsqueeze(0)))
        # psma_pt = resizer(scaler(psma_pt.unsqueeze(0)))
        # psma_mask = resizer_mask(psma_mask.unsqueeze(0))



        
        
        # save_h5(
        #     os.path.join(save_dir, patient_name),
        #     fdg_ct,
        #     fdg_pt,
        #     fdg_mask,
        #     psma_ct,
        #     psma_pt,
        #     psma_mask
        # )



if __name__ == "__main__":

    crop_and_intensity(patient_dir='/data/xiangcen/pet_gen/processed/batch1', 
    save_dir='/data/xiangcen/pet_gen/processed/batch1_h5',
    img_size=(128, 128, 384))