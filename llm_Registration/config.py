import torch

from General.segments import SEGMENT_INDEX


LLM_MODEL_PATH = "llm_models/Qwen3.5-9B"
REGISTRATION_WEIGHTS_PATH = "/data1/xiangcen/models/registration_v2/ctsmoothness_l4500_k10_mar3000_gam2.0.ptm"
SINGLE_CASE_OUTPUT_DIR = "/share/home/xcwu/pet_reg_results_llm/single_case_json"
DEFAULT_PATIENT_PATH = "/data2/xiangcen/data/pet_gen/processed/batch3_h5_v2/patient_0066.h5"

SPATIAL_SIZE = (128, 128, 384)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LLM_MAX_NEW_TOKENS = 10000
DEFAULT_MAX_ORGANS_FOR_LLM = 20

EXCLUDED_MASK_NAMES = {"kidney_cyst_left", "kidney_cyst_right"}
MASK_NAMES = [
    name for name in SEGMENT_INDEX.keys()
    if name not in EXCLUDED_MASK_NAMES
]
