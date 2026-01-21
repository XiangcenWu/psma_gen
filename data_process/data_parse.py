import argparse
# personal imports
from ProcessZip import process_zip_files
from Inference_TS import get_ct_bodymask
from Dicom2Nii import separate_dicom_modalities
from cropAintensiry import crop_and_intensity
import os

parser = \
    argparse.ArgumentParser(description="Process medical archives, run inference, and convert to H5.")
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



args = parser.parse_args()



os.environ["TOTALSEG_WEIGHTS_PATH"] = args.ts_dir

# process all zip/rar files and saved to one place
process_zip_files(args.pressed_dir, args.patient_dir)
# inference using totalsegmentator
get_ct_bodymask(args.patient_dir)
# crop and intensity range then convert to h5
crop_and_intensity(args.patient_dir)