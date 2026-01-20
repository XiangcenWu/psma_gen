import os
from totalsegmentator.python_api import totalsegmentator


os.environ["TOTALSEG_WEIGHTS_PATH"] = '/data/xiangcen/TotalSegmentator'


def inference_ts(ct_dir, output_dir, task='body'):
    totalsegmentator(ct_dir, output_dir, ml=True, task=task)


if __name__ == '__main__':
    
    fdgCT_name = 'fdgCT_series2.nii.gz'
    psmaCT_name = 'psmaCT_series2.nii.gz'
    
    fdgCT_mask_name = 'fdgCT_body_mask.nii.gz'
    psmaCT_mask_name = 'psmaCT_body_mask.nii.gz'
    
    
    patients_dir = '/data/xiangcen/pet_gen/processed/batch1'
    for patient_dir in os.listdir(patients_dir):
        print(patient_dir)

        fdgCT_dir = os.path.join(patients_dir, patient_dir, fdgCT_name)
        psmaCT_dir = os.path.join(patients_dir, patient_dir, psmaCT_name)
    
        fdgCT_mask_dir = os.path.join(patients_dir, patient_dir, fdgCT_mask_name)
        psmaCT_mask_dir = os.path.join(patients_dir, patient_dir, psmaCT_mask_name)


        inference_ts(fdgCT_dir, fdgCT_mask_dir)
        inference_ts(psmaCT_dir, psmaCT_mask_dir)

