import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm_Registration.inference_single_case import inference_single_case_json


MODEL_PATH = "llm_models/Qwen3.5-9B"


REGION_PRESETS = {
    "pelvis": {
        "description": "Pelvic region: bladder, prostate, sacrum, L5/S1, hips, femurs, iliopsoas",
        "keywords": [
            "urinary_bladder",
            "prostate",
            "sacrum",
            "vertebrae_S1",
            "vertebrae_L5",
            "hip_left",
            "hip_right",
            "femur_left",
            "femur_right",
            "iliopsoas_left",
            "iliopsoas_right",
        ],
    },
    "abdomen": {
        "description": "Abdominal region: liver, spleen, kidneys, stomach, pancreas, bowel",
        "keywords": [
            "spleen",
            "kidney_right",
            "kidney_left",
            "liver",
            "stomach",
            "pancreas",
            "small_bowel",
            "duodenum",
            "colon",
        ],
    },
    "thorax": {
        "description": "Thoracic region: lungs, heart, aorta, vessels, ribs",
        "keywords": [
            "lung_upper_lobe_left",
            "lung_lower_lobe_left",
            "lung_upper_lobe_right",
            "lung_middle_lobe_right",
            "lung_lower_lobe_right",
            "heart",
            "aorta",
            "sternum",
            "rib_left",
            "rib_right",
        ],
    },
    "spine": {
        "description": "Spinal region: vertebrae and spinal cord",
        "keywords": ["vertebrae", "sacrum", "spinal_cord"],
    },
    "bone": {
        "description": "Whole-body skeletal structures",
        "keywords": [
            "skull",
            "vertebrae",
            "rib",
            "sternum",
            "humerus",
            "scapula",
            "clavicula",
            "femur",
            "hip",
            "tibia",
            "fibula",
            "ulna",
            "radius",
        ],
    },
}


AVAILABLE_TOOLS = {
    "finetune_registration_roi": {
        "description": "Fine-tune the registration model on a selected anatomical ROI.",
    },
    "run_traditional_local_registration": {
        "description": "Run a traditional local registration method on selected organs or a selected region.",
    },
    "run_pair_specific_finetune": {
        "description": "Perform pair-specific test-time adaptation for this one case.",
    },
    "generate_qc_report": {
        "description": "Generate a QC report without modifying the registration result.",
    },
    "accept_current_result": {
        "description": "Accept the current warped result without further refinement.",
    },
    "no_action": {
        "description": "No safe or useful action is available.",
    },
}


AGENT_SYSTEM_PROMPT = """
You are a medical image registration agent.

You will receive:
1. A JSON summary of one registration result.
2. An optional user request.

Your job:
1. Summarize whether the registration improved.
2. Identify which organs or regions still need improvement.
3. Decide which tool should be called next.
4. Output exactly one valid JSON object.

You cannot directly process medical images.
You can only read the provided JSON.
You must not output Markdown.
You must not output Python code.
You must not output explanations outside JSON.

Available tools:
- finetune_registration_roi
- run_traditional_local_registration
- run_pair_specific_finetune
- generate_qc_report
- accept_current_result
- no_action

Available anatomical regions:
- pelvis
- abdomen
- thorax
- spine
- bone

Decision rules:
1. If the user explicitly requests a region, prioritize that region.
2. If no user region is provided, prioritize organs with low dice_after, high tre_after, or small improvement.
3. If mean_dice_after is good and most organs improved, choose accept_current_result.
4. If only a few organs remain poor, choose finetune_registration_roi.
5. If the global result is poor or many organs remain poor, choose run_pair_specific_finetune.
6. If the target is a small local anatomical area, choose run_traditional_local_registration or finetune_registration_roi.
7. If the information is insufficient, choose generate_qc_report.
8. Do not invent organ names that are not present in the input JSON.

Output JSON format:
{
  "case_id": "patient_0064",
  "overall_summary": "The registration improved globally, but several abdominal organs still need improvement.",
  "needs_update": true,
  "target_region": "abdomen",
  "priority_organs": [
    {
      "name": "kidney_right",
      "label": 2,
      "dice_after": 0.83,
      "tre_after": 3.65,
      "reason": "High priority because the user requested the abdominal region."
    }
  ],
  "recommended_tool": "finetune_registration_roi",
  "tool_args": {
    "region": "abdomen",
    "organ_names": ["spleen", "kidney_right", "kidney_left", "liver"],
    "labels": [1, 2, 3, 5],
    "max_steps": 100,
    "lr": 0.00001
  },
  "safety_checks": [
    "Accept only if ROI Dice improves.",
    "Reject if global Dice decreases by more than 0.01.",
    "Reject if deformation folding increases beyond threshold."
  ],
  "reason": "The requested region is abdomen and the tool can refine organ-wise alignment."
}
Important constraints:
1. priority_organs must contain at most 3 organs.
2. Each organ reason must be shorter than 15 words.
3. overall_summary must be one sentence.
4. reason must be one sentence.
5. Do not include analysis. Output only the final JSON object.
"""


def load_qwen(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def extract_json(text: str) -> dict:
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        raise ValueError(f"No JSON object found in model output:\n{text}")

    return json.loads(match.group(0))


def safe_json_dumps(obj: Any, indent: int = 2) -> str:
    return json.dumps(obj, indent=indent, ensure_ascii=False)


def infer_region_from_user_request(user_request: Optional[str]) -> Optional[str]:
    if not user_request:
        return None

    text = user_request.lower()
    if any(k in text for k in ["pelvis", "pelvic", "bladder", "prostate", "hip"]):
        return "pelvis"
    if any(k in text for k in ["abdomen", "abdominal", "liver", "spleen", "kidney", "bowel", "pancreas"]):
        return "abdomen"
    if any(k in text for k in ["thorax", "thoracic", "chest", "lung", "heart"]):
        return "thorax"
    if any(k in text for k in ["spine", "vertebra", "vertebrae", "spinal"]):
        return "spine"
    if any(k in text for k in ["bone", "skeleton", "skeletal", "rib", "femur", "skull"]):
        return "bone"

    return None


def organ_matches_region(organ_name: str, region: str) -> bool:
    organ_name_lower = organ_name.lower()
    return any(
        keyword.lower() in organ_name_lower
        for keyword in REGION_PRESETS[region]["keywords"]
    )


def summarize_organs_for_llm(
    registration_json: Dict[str, Any],
    user_request: Optional[str] = None,
    max_organs: int = 20,
) -> Dict[str, Any]:
    organs = registration_json.get("organs", {})
    target_region_hint = infer_region_from_user_request(user_request)
    organ_rows = []

    for name, item in organs.items():
        dice_after = float(item.get("dice_after") or 0.0)
        tre_after = float(item.get("tre_after") or 999.0)
        dice_delta = float(item.get("dice_delta") or 0.0)
        tre_delta = float(item.get("tre_delta") or 0.0)

        matched_target_region = (
            target_region_hint is not None
            and organ_matches_region(name, target_region_hint)
        )

        priority_score = 0.0
        priority_score += (1.0 - dice_after) * 2.0
        priority_score += min(tre_after / 20.0, 1.0)
        if dice_delta < 0.05:
            priority_score += 0.5
        if tre_delta > -1.0:
            priority_score += 0.3
        if matched_target_region:
            priority_score += 2.0

        organ_rows.append(
            {
                "name": name,
                "label": item.get("label"),
                "dice_before": item.get("dice_before"),
                "dice_after": item.get("dice_after"),
                "dice_delta": item.get("dice_delta"),
                "tre_before": item.get("tre_before"),
                "tre_after": item.get("tre_after"),
                "tre_delta": item.get("tre_delta"),
                "matched_target_region": matched_target_region,
                "priority_score": priority_score,
            }
        )

    organ_rows = sorted(
        organ_rows,
        key=lambda x: x["priority_score"],
        reverse=True,
    )

    return {
        "target_region_hint": target_region_hint,
        "selected_organs_for_llm": organ_rows[:max_organs],
    }


def build_llm_input(
    registration_json: Dict[str, Any],
    user_request: Optional[str] = None,
    max_organs: int = 20,
) -> Dict[str, Any]:
    organ_summary = summarize_organs_for_llm(
        registration_json=registration_json,
        user_request=user_request,
        max_organs=max_organs,
    )

    return {
        "case_id": registration_json.get("case_id"),
        "data_path": registration_json.get("data_path"),
        "user_request": user_request,
        "target_region_hint_from_user": organ_summary["target_region_hint"],
        "metrics": registration_json.get("metrics", {}),
        "available_regions": REGION_PRESETS,
        "available_tools": AVAILABLE_TOOLS,
        "selected_organs_for_llm": organ_summary["selected_organs_for_llm"],
    }


def validate_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(decision, dict):
        raise ValueError("LLM decision must be a dictionary.")

    required_fields = [
        "case_id",
        "overall_summary",
        "needs_update",
        "target_region",
        "priority_organs",
        "recommended_tool",
        "tool_args",
        "reason",
    ]
    for field in required_fields:
        if field not in decision:
            raise ValueError(f"Missing required field in LLM decision: {field}")

    if decision["recommended_tool"] not in AVAILABLE_TOOLS:
        raise ValueError(f"Unsupported tool: {decision['recommended_tool']}")

    if decision["target_region"] is not None and decision["target_region"] not in REGION_PRESETS:
        raise ValueError(f"Unsupported target_region: {decision['target_region']}")

    if not isinstance(decision["priority_organs"], list):
        raise ValueError("priority_organs must be a list.")

    if not isinstance(decision["tool_args"], dict):
        raise ValueError("tool_args must be a dictionary.")

    tool_args = decision["tool_args"]
    if decision["recommended_tool"] in [
        "finetune_registration_roi",
        "run_pair_specific_finetune",
        "run_traditional_local_registration",
    ]:
        if "region" not in tool_args:
            tool_args["region"] = decision["target_region"]
        if "organ_names" not in tool_args:
            tool_args["organ_names"] = [
                x.get("name") for x in decision["priority_organs"] if "name" in x
            ]
        if "labels" not in tool_args:
            tool_args["labels"] = [
                x.get("label") for x in decision["priority_organs"] if "label" in x
            ]

        tool_args["max_steps"] = int(tool_args.get("max_steps", 100))
        tool_args["lr"] = float(tool_args.get("lr", 1e-5))
        tool_args["max_steps"] = min(max(tool_args["max_steps"], 10), 300)
        tool_args["lr"] = min(max(tool_args["lr"], 1e-6), 1e-4)

    return decision


def ask_qwen_for_registration_decision(
    tokenizer,
    model,
    registration_json: Dict[str, Any],
    user_request: Optional[str] = None,
    max_organs: int = 20,
) -> Dict[str, Any]:
    llm_input = build_llm_input(
        registration_json=registration_json,
        user_request=user_request,
        max_organs=max_organs,
    )

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": safe_json_dumps(llm_input, indent=2)},
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer([prompt], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=10000,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()



    print("\n========== LLM Raw Output ==========")
    print(answer)
    print("====================================\n")

    return validate_decision(extract_json(answer))


def summarize_single_case_registration(
    patient_path: str,
    user_request: Optional[str] = None,
    max_organs: int = 20,
    debug: bool = True,
) -> Dict[str, Any]:
    registration_json = inference_single_case_json(patient_path)
    tokenizer, model = load_qwen(MODEL_PATH)
    decision = ask_qwen_for_registration_decision(
        tokenizer=tokenizer,
        model=model,
        registration_json=registration_json,
        user_request=user_request,
        max_organs=max_organs,
    )

    return {
        "case_id": registration_json.get("case_id"),
        "user_request": user_request,
        "registration_json": registration_json,
        "llm_summary": decision,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run single-case registration inference and summarize it with an LLM."
    )
    parser.add_argument(
        "--user_request",
        type=str,
        default=None,
        help="Optional user request, e.g. improve pelvic region.",
    )
    parser.add_argument(
        "--max_organs",
        type=int,
        default=20,
        help="Maximum number of priority organs sent to the LLM.",
    )

    args = parser.parse_args()
    result = summarize_single_case_registration(
        patient_path="/data2/xiangcen/data/pet_gen/processed/batch3_h5_v2/patient_0066.h5",
        user_request=args.user_request,
        max_organs=args.max_organs,
    )
    print("\n========== Final Registration Summary ==========")
    print(safe_json_dumps(result, indent=2))
    print("================================================\n")
