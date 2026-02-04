import argparse
# personal imports
from ProcessZip import process_zip_files
from Inference_TS import get_ct_masks
from Dicom2Nii import separate_dicom_modalities
from cropAintensiry import crop_and_intensity
import os




if __name__ == '__main__':
    parser = \
        argparse.ArgumentParser(description="pressed_dir is the dir with all pressed files for example 9.8.zip  \n" + \
            "patient_dir are all patiens names with all niifty files in it" + \
                "h5_dir is final processed h5 file dir")
    parser.add_argument(
        "--pressed_dir",
        type=str,
        required=True,
        help="Path to the directory containing ZIP/RAR archive files."
    )
    parser.add_argument(
        "--patient_dir",
        type=str,
        required=True,
        help="Path to the output directory where processed patient data will be saved."
    )
    parser.add_argument(
        "--ts_dir",
        type=str,
        required=True,
        help="Path to the ts model dir."
    )
    
    parser.add_argument(
        "--h5_dir",
        type=str,
        required=True,
        help="Path to save h5 cropped files."
    )



    args = parser.parse_args()



    os.environ["TOTALSEG_WEIGHTS_PATH"] = args.ts_dir
    
    
    process_zip_files(args.pressed_dir, args.patient_dir)


    # get total mask
    get_ct_masks(args.patient_dir, task='total')
    # get body mask
    get_ct_masks(args.patient_dir, task='body')
    # get appendicular_bones mask
    get_ct_masks(args.patient_dir, task='appendicular_bones')
    
    
    crop_and_intensity(patient_dir=args.patient_dir, 
    save_dir=args.h5_dir,
    img_size=(128, 128, 384))