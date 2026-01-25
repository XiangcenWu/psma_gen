import os
from totalsegmentator.python_api import totalsegmentator
from tqdm import tqdm



def inference_ts(ct_dir, output_dir, task='body'):
    totalsegmentator(ct_dir, output_dir, ml=True, task=task)


def get_ct_bodymask(patients_dir):
    
    fdgCT_name = 'fdgCT_series2.nii.gz'
    psmaCT_name = 'psmaCT_series2.nii.gz'
    
    fdgCT_mask_name = 'fdgCT_body_mask.nii.gz'
    psmaCT_mask_name = 'psmaCT_body_mask.nii.gz'
    
    
    for patient_dir in tqdm(os.listdir(patients_dir), desc='Get Ct Mask', unit='case'):
        print(patient_dir)

        fdgCT_dir = os.path.join(patients_dir, patient_dir, fdgCT_name)
        psmaCT_dir = os.path.join(patients_dir, patient_dir, psmaCT_name)
    
        fdgCT_mask_dir = os.path.join(patients_dir, patient_dir, fdgCT_mask_name)
        psmaCT_mask_dir = os.path.join(patients_dir, patient_dir, psmaCT_mask_name)


        inference_ts(fdgCT_dir, fdgCT_mask_dir)
        inference_ts(psmaCT_dir, psmaCT_mask_dir)



def get_ct_totalmask(patients_dir):
    
    fdgCT_name = 'fdgCT_series2.nii.gz'
    psmaCT_name = 'psmaCT_series2.nii.gz'
    
    fdgCT_mask_name = 'fdgCT_total_mask.nii.gz'
    psmaCT_mask_name = 'psmaCT_total_mask.nii.gz'
    
    
    for patient_dir in tqdm(os.listdir(patients_dir), desc='Get Ct full Mask', unit='case'):
        print(patient_dir)

        fdgCT_dir = os.path.join(patients_dir, patient_dir, fdgCT_name)
        psmaCT_dir = os.path.join(patients_dir, patient_dir, psmaCT_name)
    
        fdgCT_mask_dir = os.path.join(patients_dir, patient_dir, fdgCT_mask_name)
        psmaCT_mask_dir = os.path.join(patients_dir, patient_dir, psmaCT_mask_name)

        if os.path.exists(fdgCT_mask_dir) and os.path.exists(psmaCT_mask_dir):
            continue
            

        inference_ts(fdgCT_dir, fdgCT_mask_dir, task='total')
        inference_ts(psmaCT_dir, psmaCT_mask_dir, task='total')