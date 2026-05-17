import json
import re
import copy
import time
import torch
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# 1. Model path
# ============================================================

MODEL_PATH = "./Qwen3.5-9B"


# ============================================================
# 2. Region-to-label presets
# ============================================================

REGION_PRESETS = {
    "pelvis": {
        "description": "Pelvic region, including bladder, prostate, sacrum, L5/S1, hips, femurs, and iliopsoas",
        "labels": [21, 22, 25, 26, 27, 75, 76, 77, 78, 88, 89],
    },
    "abdomen": {
        "description": "Abdominal region, including liver, spleen, kidneys, stomach, pancreas, and bowel",
        "labels": [1, 2, 3, 5, 6, 7, 18, 19, 20],
    },
    "thorax": {
        "description": "Thoracic region, including lungs, heart, and aorta",
        "labels": [10, 11, 12, 13, 14, 51, 52],
    },
    "spine": {
        "description": "Spinal region, including sacrum, cervical/thoracic/lumbar/sacral vertebrae, and spinal cord",
        "labels": list(range(25, 51)) + [79],
    },
    "bone": {
        "description": "Skeletal structures, including vertebrae, ribs, skull, femurs, hips, and extremity bones",
        "labels": (
            list(range(25, 51))
            + list(range(69, 79))
            + list(range(90, 117))
            + list(range(118, 129))
        ),
    },
}


# ============================================================
# 3. Agent system prompt
# ============================================================

AGENT_SYSTEM_PROMPT = """
You are a medical image registration agent.

Your job is to choose the next tool call based on:
1. the user's registration goal,
2. the current registration metrics,
3. the latest tool observation,
4. the full action history,
5. safety constraints.

You cannot directly process medical images.
You must only read structured JSON summaries.
You must output exactly one valid JSON object.
Do not output Markdown.
Do not output Python code.
Do not output explanations outside JSON.

Available actions:
- evaluate_registration
- finetune_registration_roi
- accept_result
- reject_result
- stop_with_failure

Available regions:
- pelvis
- abdomen
- thorax
- spine
- bone

Available tool meaning:
1. evaluate_registration:
   Recompute or summarize current registration quality.
2. finetune_registration_roi:
   Run ROI-specific pair-wise registration refinement.
3. accept_result:
   Accept the current best registration result.
4. reject_result:
   Reject the latest result and keep the previous best result.
5. stop_with_failure:
   Stop because no safe improvement is available.

Safety rules:
1. Prefer improving the user-requested ROI Dice.
2. Do not accept a result if global_mean_dice decreases by more than 0.01.
3. Do not accept a result if negative_jacobian_ratio is greater than 0.03.
4. If a refinement does not improve roi_mean_dice, reject it.
5. If the latest refinement passed the hard safety check and improved the requested ROI, usually choose accept_result.
6. Stop after repeated failed refinements.
7. Do not keep calling the same refinement tool if the latest safe result is already good enough.

For finetune_registration_roi, output:
{
  "action": "finetune_registration_roi",
  "region": "pelvis",
  "max_steps": 100,
  "lr": 0.00001,
  "reason": "The pelvic ROI Dice is low and should be refined.",
  "expected_improvement": "Improve bladder, prostate, sacrum, and hip alignment."
}

For evaluate_registration, output:
{
  "action": "evaluate_registration",
  "reason": "Need to inspect the current registration quality before deciding refinement."
}

For accept_result, output:
{
  "action": "accept_result",
  "reason": "The ROI Dice improved and global alignment was preserved.",
  "improvement_summary": "The refinement improved the requested region without violating safety constraints."
}

For reject_result, output:
{
  "action": "reject_result",
  "reason": "The refinement worsened global Dice or deformation plausibility.",
  "suggested_fix": "Rollback to the previous best result."
}

For stop_with_failure, output:
{
  "action": "stop_with_failure",
  "reason": "No safe improvement was achieved after multiple attempts.",
  "suggested_fix": "Keep the base registration result."
}
"""


FINAL_SUMMARY_PROMPT = """
You are a medical image registration assistant.

You will receive a final JSON state from a registration agent.
Summarize the result in clear English.

Rules:
1. Do not make any diagnosis.
2. Only discuss registration quality, Dice changes, safety checks, accepted/rejected status, and suggested next steps.
3. Mention which region was refined.
4. Mention whether the final result was accepted or rejected.
5. Keep the response concise.
"""


# ============================================================
# 4. Load Qwen
# ============================================================

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


# ============================================================
# 5. JSON extraction
# ============================================================

def extract_json(text: str) -> dict:
    """
    Extract a JSON object from Qwen output.
    This handles raw JSON, Markdown code fences, and extra text.
    """
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


# ============================================================
# 6. Qwen generation functions
# ============================================================

def generate_qwen_text(
    tokenizer,
    model,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 512,
    debug_title: Optional[str] = None,
) -> str:
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
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if debug_title is not None:
        print(f"\n========== {debug_title} ==========")
        print(answer)
        print("=" * (22 + len(debug_title)) + "\n")

    return answer


def ask_qwen_next_action(tokenizer, model, state: dict, debug: bool = True) -> dict:
    """
    Send the current agent state JSON to Qwen.
    Qwen returns the next tool call as JSON.
    """
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": safe_json_dumps(state, indent=2),
        },
    ]

    answer = generate_qwen_text(
        tokenizer=tokenizer,
        model=model,
        messages=messages,
        max_new_tokens=512,
        debug_title="Qwen Next Action" if debug else None,
    )

    return extract_json(answer)


def ask_qwen_final_summary(tokenizer, model, final_state: dict, debug: bool = True) -> str:
    messages = [
        {"role": "system", "content": FINAL_SUMMARY_PROMPT},
        {
            "role": "user",
            "content": safe_json_dumps(final_state, indent=2),
        },
    ]

    return generate_qwen_text(
        tokenizer=tokenizer,
        model=model,
        messages=messages,
        max_new_tokens=512,
        debug_title="Qwen Final Summary" if debug else None,
    )


# ============================================================
# 7. Action validation
# ============================================================

def validate_agent_action(action: dict) -> dict:
    allowed_actions = {
        "evaluate_registration",
        "finetune_registration_roi",
        "accept_result",
        "reject_result",
        "stop_with_failure",
    }

    if not isinstance(action, dict):
        raise ValueError("The action must be a dictionary.")

    if "action" not in action:
        raise ValueError("Missing required field: action")

    if action["action"] not in allowed_actions:
        raise ValueError(f"Unsupported action: {action['action']}")

    if action["action"] == "finetune_registration_roi":
        if "region" not in action:
            raise ValueError("Missing required field for finetune_registration_roi: region")

        if action["region"] not in REGION_PRESETS:
            raise ValueError(f"Unsupported region: {action['region']}")

        action["max_steps"] = int(action.get("max_steps", 100))
        action["lr"] = float(action.get("lr", 1e-5))

        # Hard parameter limits.
        action["max_steps"] = min(max(action["max_steps"], 10), 300)
        action["lr"] = min(max(action["lr"], 1e-6), 1e-4)

        if "expected_improvement" not in action:
            action["expected_improvement"] = ""

    if "reason" not in action:
        action["reason"] = ""

    return action


# ============================================================
# 8. Simple user-goal region hint
# ============================================================

def infer_region_hint_from_user_goal(user_goal: str) -> Optional[str]:
    """
    This is only a weak hint for the state.
    The LLM still decides the final region.
    """
    text = user_goal.lower()

    if any(k in text for k in ["pelvis", "pelvic", "bladder", "prostate", "hip"]):
        return "pelvis"
    if any(k in text for k in ["abdomen", "abdominal", "liver", "spleen", "kidney", "bowel", "pancreas"]):
        return "abdomen"
    if any(k in text for k in ["thorax", "thoracic", "chest", "lung", "heart"]):
        return "thorax"
    if any(k in text for k in ["spine", "vertebra", "vertebrae", "spinal"]):
        return "spine"
    if any(k in text for k in ["bone", "skeletal", "skeleton", "rib", "femur", "skull"]):
        return "bone"

    return None


# ============================================================
# 9. Dummy result helpers
# ============================================================

def get_region_metric(metrics: dict, region: str) -> float:
    roi_mean_dice = metrics.get("roi_mean_dice", {})
    if isinstance(roi_mean_dice, dict):
        return float(roi_mean_dice.get(region, 0.0))
    return float(roi_mean_dice)


def count_previous_refinements(state: dict, region: Optional[str] = None) -> int:
    count = 0
    for item in state.get("history", []):
        action = item.get("action", {})
        if action.get("action") == "finetune_registration_roi":
            if region is None or action.get("region") == region:
                count += 1
    return count


# ============================================================
# 10. Tool 1: Base registration inference
# ============================================================

def run_base_registration(case_data: dict, user_goal: str) -> dict:
    """
    Run the trained base registration model once.

    TODO_REAL_IMPLEMENTATION:
    Replace this dummy function with your real base registration inference.

    Real implementation should do:
    1. Load fixed PSMA PET.
    2. Load moving FDG PET.
    3. Load fixed TotalSegmentator mask.
    4. Load moving TotalSegmentator mask.
    5. Run trained registration model:
       flow = registration_model(moving_pet, fixed_pet)
    6. Warp moving FDG:
       warped_moving_pet = warp_linear(moving_pet, flow)
    7. Warp moving mask:
       warped_moving_mask = warp_nearest(moving_mask, flow)
    8. Compute metrics:
       - global_mean_dice
       - roi_mean_dice for pelvis / abdomen / thorax / spine / bone
       - per_label_dice
       - negative_jacobian_ratio
       - smoothness
    9. Save result files:
       - flow
       - warped PET
       - warped mask
       - QC figures if available
    10. Return a JSON-serializable result.
    """
    print(">>> Running tool: run_base_registration")
    print(">>> NOTE: This is a DUMMY base registration result.")

    target_region = infer_region_hint_from_user_goal(user_goal)

    # Dummy metrics. You can adjust these for testing different behaviors.
    roi_mean_dice = {
        "pelvis": 0.61,
        "abdomen": 0.72,
        "thorax": 0.76,
        "spine": 0.81,
        "bone": 0.79,
    }

    bad_regions = []
    for region, dice in roi_mean_dice.items():
        if dice < 0.70:
            bad_regions.append(region)

    if target_region is not None and target_region not in bad_regions:
        # Make the user-requested region slightly suspicious in dummy mode.
        bad_regions.append(target_region)

    result = {
        "result_id": "base_result",
        "tool": "run_base_registration",
        "status": "success",
        "is_dummy": True,
        "metrics": {
            "global_mean_dice": 0.74,
            "roi_mean_dice": roi_mean_dice,
            "negative_jacobian_ratio": 0.004,
            "smoothness_loss": 0.031,
            "bad_regions": bad_regions,
            "bad_labels": {
                "urinary_bladder": 0.55,
                "prostate": 0.48,
                "hip_left": 0.62,
                "hip_right": 0.63,
                "liver": 0.71,
                "spleen": 0.69,
            },
        },
        "artifacts": {
            "flow_path": "DUMMY_PATH/base_flow.nii.gz",
            "warped_pet_path": "DUMMY_PATH/base_warped_fdg_pet.nii.gz",
            "warped_mask_path": "DUMMY_PATH/base_warped_mask.nii.gz",
            "qc_report_path": "DUMMY_PATH/base_qc.json",
        },
    }

    return result


# ============================================================
# 11. Tool 2: Evaluate registration
# ============================================================

def evaluate_registration(case_data: dict, state: dict) -> dict:
    """
    Evaluate current registration result.

    TODO_REAL_IMPLEMENTATION:
    Replace this dummy function with your real evaluation code.

    Real implementation should:
    1. Read the current best warped mask.
    2. Compare it with fixed mask.
    3. Compute global Dice.
    4. Compute ROI Dice.
    5. Compute per-label Dice.
    6. Compute Jacobian determinant statistics.
    7. Return JSON metrics.
    """
    print(">>> Running tool: evaluate_registration")
    print(">>> NOTE: This is a DUMMY evaluation result.")

    best_metrics = copy.deepcopy(state.get("best_metrics", {}))

    return {
        "status": "success",
        "is_dummy": True,
        "current_result_id": state.get("best_result_id"),
        "metrics": best_metrics,
        "comment": "Current best registration metrics were summarized.",
    }


# ============================================================
# 12. Tool 3: ROI-specific fine-tuning
# ============================================================

def finetune_registration_roi(
    case_data: dict,
    state: dict,
    roi_labels: List[int],
    region: str,
    max_steps: int = 100,
    lr: float = 1e-5,
) -> dict:
    """
    Run ROI-specific pair-wise registration refinement.

    TODO_REAL_IMPLEMENTATION:
    Replace this dummy function with your real pair-specific fine-tuning code.

    Real implementation should do:
    1. Load current best registration model or deformation field.
    2. Select ROI labels from roi_labels.
    3. Build ROI binary masks from fixed_mask and moving_mask.
    4. Run pair-specific fine-tuning:
       - optionally freeze encoder
       - train decoder / flow head / deformation field
       - optimize ROI Dice + PET/CT similarity + smoothness + Jacobian penalty
    5. Warp moving FDG PET again.
    6. Warp moving mask again.
    7. Compute before metrics.
    8. Compute after metrics.
    9. Save refined flow / warped PET / warped mask / QC result.
    10. Return JSON observation.
    """
    print(">>> Running tool: finetune_registration_roi")
    print(">>> NOTE: This is a DUMMY ROI fine-tuning result.")
    print(f">>> Region: {region}")
    print(f">>> ROI labels: {roi_labels}")
    print(f">>> max_steps: {max_steps}")
    print(f">>> lr: {lr}")

    best_metrics = copy.deepcopy(state.get("best_metrics", {}))
    before_global = float(best_metrics.get("global_mean_dice", 0.0))
    before_roi = get_region_metric(best_metrics, region)
    before_jac = float(best_metrics.get("negative_jacobian_ratio", 0.0))
    before_smoothness = float(best_metrics.get("smoothness_loss", 0.0))

    previous_refinements = count_previous_refinements(state, region=region)

    # Dummy behavior:
    # First refinement improves a lot.
    # Later repeated refinements improve less.
    if previous_refinements == 0:
        roi_delta = 0.07
        global_delta = -0.002
        jac_delta = 0.002
    elif previous_refinements == 1:
        roi_delta = 0.015
        global_delta = -0.004
        jac_delta = 0.004
    else:
        roi_delta = -0.005
        global_delta = -0.012
        jac_delta = 0.010

    after_roi = max(0.0, min(1.0, before_roi + roi_delta))
    after_global = max(0.0, min(1.0, before_global + global_delta))
    after_jac = max(0.0, before_jac + jac_delta)
    after_smoothness = max(0.0, before_smoothness + 0.002)

    result_id = f"refined_{region}_step_{state.get('current_step', 0)}"

    result = {
        "result_id": result_id,
        "tool": "finetune_registration_roi",
        "status": "success",
        "is_dummy": True,
        "region": region,
        "labels": roi_labels,
        "parameters": {
            "max_steps": max_steps,
            "lr": lr,
        },
        "before": {
            "global_mean_dice": before_global,
            "roi_mean_dice": before_roi,
            "negative_jacobian_ratio": before_jac,
            "smoothness_loss": before_smoothness,
        },
        "after": {
            "global_mean_dice": after_global,
            "roi_mean_dice": after_roi,
            "negative_jacobian_ratio": after_jac,
            "smoothness_loss": after_smoothness,
        },
        "improvement": {
            "roi_mean_dice_delta": after_roi - before_roi,
            "global_mean_dice_delta": after_global - before_global,
            "negative_jacobian_ratio_delta": after_jac - before_jac,
            "smoothness_loss_delta": after_smoothness - before_smoothness,
        },
        "artifacts": {
            "flow_path": f"DUMMY_PATH/{result_id}_flow.nii.gz",
            "warped_pet_path": f"DUMMY_PATH/{result_id}_warped_fdg_pet.nii.gz",
            "warped_mask_path": f"DUMMY_PATH/{result_id}_warped_mask.nii.gz",
            "qc_report_path": f"DUMMY_PATH/{result_id}_qc.json",
        },
    }

    return result


# ============================================================
# 13. Hard safety check
# ============================================================

def safety_check_refinement(observation: dict) -> dict:
    """
    Hard safety rules.
    This should override the LLM when needed.
    """
    if observation.get("called_tool") != "finetune_registration_roi":
        return {
            "safe": True,
            "accepted_by_rule": None,
            "reason": "No refinement result to check.",
        }

    result = observation["result"]

    before = result["before"]
    after = result["after"]

    roi_delta = float(after["roi_mean_dice"]) - float(before["roi_mean_dice"])
    global_delta = float(after["global_mean_dice"]) - float(before["global_mean_dice"])
    jac = float(after["negative_jacobian_ratio"])

    if jac > 0.03:
        return {
            "safe": False,
            "accepted_by_rule": False,
            "reason": "Rejected because negative_jacobian_ratio is greater than 0.03.",
        }

    if global_delta < -0.01:
        return {
            "safe": False,
            "accepted_by_rule": False,
            "reason": "Rejected because global_mean_dice dropped by more than 0.01.",
        }

    if roi_delta <= 0:
        return {
            "safe": False,
            "accepted_by_rule": False,
            "reason": "Rejected because ROI Dice did not improve.",
        }

    return {
        "safe": True,
        "accepted_by_rule": True,
        "reason": "Accepted by hard rule because ROI Dice improved and safety constraints were satisfied.",
    }


# ============================================================
# 14. Rule-based fallback action
# ============================================================

def rule_based_fallback_action(state: dict) -> dict:
    """
    Fallback when Qwen output is invalid.
    This keeps the pipeline running safely.
    """
    print(">>> Using rule-based fallback action.")

    target_region = state.get("target_region_hint")
    latest_observation = state.get("latest_observation", {})

    if latest_observation.get("called_tool") == "run_base_registration":
        if target_region is not None:
            return {
                "action": "finetune_registration_roi",
                "region": target_region,
                "max_steps": 100,
                "lr": 1e-5,
                "reason": "Fallback: refine the user-requested region after base registration.",
                "expected_improvement": "Improve the requested ROI alignment.",
            }

    if latest_observation.get("called_tool") == "finetune_registration_roi":
        safety = latest_observation.get("safety_check", {})
        if safety.get("accepted_by_rule") is True:
            return {
                "action": "accept_result",
                "reason": "Fallback: latest refinement passed the safety check.",
                "improvement_summary": "The latest refinement improved ROI Dice and satisfied safety rules.",
            }
        return {
            "action": "reject_result",
            "reason": "Fallback: latest refinement failed the safety check.",
            "suggested_fix": "Rollback to the previous best result.",
        }

    return {
        "action": "evaluate_registration",
        "reason": "Fallback: evaluate current registration quality.",
    }


# ============================================================
# 15. Dispatcher
# ============================================================

def dispatch_agent_action(action: dict, case_data: dict, state: dict) -> dict:
    action = validate_agent_action(action)

    if action["action"] == "evaluate_registration":
        result = evaluate_registration(case_data=case_data, state=state)
        return {
            "called_tool": "evaluate_registration",
            "result": result,
        }

    if action["action"] == "finetune_registration_roi":
        region = action["region"]
        labels = REGION_PRESETS[region]["labels"]

        result = finetune_registration_roi(
            case_data=case_data,
            state=state,
            roi_labels=labels,
            region=region,
            max_steps=action["max_steps"],
            lr=action["lr"],
        )

        return {
            "called_tool": "finetune_registration_roi",
            "region": region,
            "region_description": REGION_PRESETS[region]["description"],
            "labels": labels,
            "result": result,
        }

    if action["action"] == "accept_result":
        return {
            "called_tool": "accept_result",
            "result": {
                "status": "accepted",
                "best_result_id": state.get("best_result_id"),
                "best_metrics": state.get("best_metrics"),
                "reason": action.get("reason", ""),
                "improvement_summary": action.get("improvement_summary", ""),
            },
        }

    if action["action"] == "reject_result":
        return {
            "called_tool": "reject_result",
            "result": {
                "status": "rejected",
                "rollback_to": state.get("best_result_id"),
                "reason": action.get("reason", ""),
                "suggested_fix": action.get("suggested_fix", ""),
            },
        }

    if action["action"] == "stop_with_failure":
        return {
            "called_tool": "stop_with_failure",
            "result": {
                "status": "stopped",
                "best_result_id": state.get("best_result_id"),
                "best_metrics": state.get("best_metrics"),
                "reason": action.get("reason", ""),
                "suggested_fix": action.get("suggested_fix", ""),
            },
        }

    raise ValueError(f"Unknown action: {action}")


# ============================================================
# 16. State update
# ============================================================

def update_best_result_if_safe(state: dict, observation: dict) -> dict:
    """
    Update best result only when the refinement passes hard safety rules.
    """
    if observation.get("called_tool") != "finetune_registration_roi":
        return state

    safety = observation.get("safety_check", {})
    if safety.get("accepted_by_rule") is not True:
        return state

    result = observation["result"]
    region = result["region"]

    state["best_result_id"] = result["result_id"]

    if "best_metrics" not in state:
        state["best_metrics"] = {}

    if "roi_mean_dice" not in state["best_metrics"]:
        state["best_metrics"]["roi_mean_dice"] = {}

    state["best_metrics"]["global_mean_dice"] = result["after"]["global_mean_dice"]
    state["best_metrics"]["roi_mean_dice"][region] = result["after"]["roi_mean_dice"]
    state["best_metrics"]["negative_jacobian_ratio"] = result["after"]["negative_jacobian_ratio"]
    state["best_metrics"]["smoothness_loss"] = result["after"]["smoothness_loss"]

    state["accepted_refinements"].append(
        {
            "result_id": result["result_id"],
            "region": region,
            "improvement": result["improvement"],
            "safety_check": safety,
        }
    )

    return state


def should_stop_after_action(action: dict, observation: dict, step: int, max_agent_steps: int) -> Optional[str]:
    """
    Return final status string if the loop should stop.
    Otherwise return None.
    """
    if action["action"] == "accept_result":
        return "accepted_by_llm"

    if action["action"] == "reject_result":
        return "rejected_by_llm"

    if action["action"] == "stop_with_failure":
        return "stopped_with_failure"

    if step >= max_agent_steps:
        return "max_steps_reached"

    return None


# ============================================================
# 17. Main agent loop
# ============================================================

def run_registration_agent_loop(
    tokenizer,
    model,
    user_goal: str,
    case_data: dict,
    max_agent_steps: int = 4,
    debug: bool = True,
) -> dict:
    print(f"\nUser goal: {user_goal}")

    target_region_hint = infer_region_hint_from_user_goal(user_goal)

    # ------------------------------------------------------------
    # Step 0: Always run base registration first.
    # ------------------------------------------------------------
    base_result = run_base_registration(
        case_data=case_data,
        user_goal=user_goal,
    )

    state = {
        "case_id": case_data.get("case_id", "DUMMY_CASE_001"),
        "user_goal": user_goal,
        "target_region_hint": target_region_hint,
        "current_step": 0,
        "max_agent_steps": max_agent_steps,
        "best_result_id": base_result["result_id"],
        "best_metrics": copy.deepcopy(base_result["metrics"]),
        "latest_observation": {
            "called_tool": "run_base_registration",
            "result": base_result,
        },
        "accepted_refinements": [],
        "history": [
            {
                "step": 0,
                "action": {
                    "action": "run_base_registration",
                    "reason": "Initial base registration inference.",
                },
                "observation": {
                    "called_tool": "run_base_registration",
                    "result": base_result,
                },
            }
        ],
        "status": "running",
    }

    print("\n========== Initial Base Result ==========")
    print(safe_json_dumps(base_result, indent=2))
    print("=========================================\n")

    # ------------------------------------------------------------
    # Agent loop
    # ------------------------------------------------------------
    for step in range(1, max_agent_steps + 1):
        state["current_step"] = step

        print(f"\n========== Agent Step {step} ==========")

        # 1. Send current JSON state to Qwen.
        try:
            action = ask_qwen_next_action(
                tokenizer=tokenizer,
                model=model,
                state=state,
                debug=debug,
            )
            action = validate_agent_action(action)
        except Exception as e:
            print("!!! Qwen action generation or validation failed.")
            print(f"!!! Error: {e}")
            action = rule_based_fallback_action(state)
            action = validate_agent_action(action)

        print("Validated agent action:")
        print(safe_json_dumps(action, indent=2))

        # 2. Execute selected tool.
        observation = dispatch_agent_action(
            action=action,
            case_data=case_data,
            state=state,
        )

        # 3. Apply hard safety check.
        safety = safety_check_refinement(observation)
        observation["safety_check"] = safety

        print("Tool observation:")
        print(safe_json_dumps(observation, indent=2))

        # 4. Update latest observation and history.
        state["latest_observation"] = observation
        state["history"].append(
            {
                "step": step,
                "action": action,
                "observation": observation,
            }
        )

        # 5. Update best result only if safe.
        state = update_best_result_if_safe(state, observation)

        # 6. Stop if action is terminal or max steps reached.
        stop_status = should_stop_after_action(
            action=action,
            observation=observation,
            step=step,
            max_agent_steps=max_agent_steps,
        )

        if stop_status is not None:
            state["status"] = stop_status
            break

    if state["status"] == "running":
        state["status"] = "finished_without_terminal_action"

    return state


# ============================================================
# 18. Save final state
# ============================================================

def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ============================================================
# 19. Main test
# ============================================================

if __name__ == "__main__":
    tokenizer, model = load_qwen(MODEL_PATH)

    # TODO_REAL_IMPLEMENTATION:
    # In your real project, put real tensors, paths, and models here.
    #
    # Example:
    # case_data = {
    #     "case_id": "CASE_001",
    #     "fixed_pet_path": "/path/to/psma_pet.nii.gz",
    #     "moving_pet_path": "/path/to/fdg_pet.nii.gz",
    #     "fixed_mask_path": "/path/to/psma_totalseg_mask.nii.gz",
    #     "moving_mask_path": "/path/to/fdg_totalseg_mask.nii.gz",
    #     "registration_model": registration_model,
    #     "device": "cuda",
    #     "output_dir": "/path/to/output",
    # }
    #
    # For now, this is dummy data.
    case_data = {
        "case_id": "DUMMY_CASE_001",
    }

    user_goal = "The bladder and prostate are not well aligned. Please improve the pelvic registration."

    final_state = run_registration_agent_loop(
        tokenizer=tokenizer,
        model=model,
        user_goal=user_goal,
        case_data=case_data,
        max_agent_steps=4,
        debug=True,
    )

    print("\n========== Final Agent State ==========")
    print(safe_json_dumps(final_state, indent=2))
    print("=======================================\n")

    save_json(final_state, "agent_final_state.json")
    print("Final agent state saved to: agent_final_state.json")

    final_summary = ask_qwen_final_summary(
        tokenizer=tokenizer,
        model=model,
        final_state=final_state,
        debug=True,
    )

    print("\n========== Final Human-Readable Summary ==========")
    print(final_summary)
    print("==================================================\n")