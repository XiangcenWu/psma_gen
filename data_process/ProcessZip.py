import os
import shutil
import zipfile
import rarfile
from pathlib import Path
from tqdm import tqdm

from Dicom2Nii import separate_dicom_modalities




def process_zip_files(pressed_dir, output_dir):


    zip_files_list = os.listdir(pressed_dir)


    for zip_file in tqdm(zip_files_list, desc="Processing zip files", unit="case"):
        zip_file = os.path.join(pressed_dir, zip_file)
        print(f'processing {zip_file}')
        # make tem dir
        tmp_dir = os.path.join(pressed_dir, 'tmp_dir')
        if os.path.isdir(tmp_dir):
            print(f"There are previous generations, delete dir {tmp_dir} \n new {tmp_dir} is generated.")
            # delete it if it exist
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
        
        
        if Path(zip_file).suffix.lower() == '.zip':
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(tmp_dir)
        elif  Path(zip_file).suffix.lower() == '.rar':
            with rarfile.RarFile(zip_file, 'r') as rf:
                rf.extractall(tmp_dir)

        
        # get the current zip files' patients
        patients = os.listdir(tmp_dir)
        print(patients)
        

        # somethimes zipped files are one floder with oirginal name
        intermediate_dir = None
        if len(patients) == 1:
            intermediate_dir = patients[0]
            patients = os.listdir(os.path.join(tmp_dir, intermediate_dir))
        
        
        for patient in patients:
            if intermediate_dir:
                patient_dir = os.path.join(tmp_dir, intermediate_dir, patient)
            else:
                patient_dir = os.path.join(tmp_dir, patient)

                
                
            psma_dir, fdg_dir = seperate_psma_dir(patient_dir)
            psma_dir, fdg_dir = os.path.join(patient_dir, psma_dir, 'DICOM'), \
                os.path.join(patient_dir, fdg_dir, 'DICOM')
            print(psma_dir, fdg_dir)
            
            # set the patient's output dir
            patient_output_dir = os.path.join(output_dir, patient)
            if not os.path.isdir(patient_output_dir):
                os.mkdir(patient_output_dir)

            # generate nii files
            separate_dicom_modalities(psma_dir, patient_output_dir, 'psma')
            separate_dicom_modalities(fdg_dir, patient_output_dir, 'fdg')



        shutil.rmtree(tmp_dir)

def seperate_psma_dir(patient_dir):
    # List all immediate subdirectories

    subdirs = [d for d in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, d))]
    # Separate them

    psma_dir = [d for d in subdirs if "PSMA" in d][0]

    fdg_dir = [d for d in subdirs if "PSMA" not in d][0]
    
    return psma_dir, fdg_dir