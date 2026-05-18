import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm_Registration.agent_tools import finetune_registration_model_on_roi
from llm_Registration.config import (
    DEFAULT_MAX_ORGANS_FOR_LLM,
    DEVICE,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL_PATH,
    REGISTRATION_WEIGHTS_PATH,
)
from llm_Registration.inference_single_case import (
    build_registration_model,
    load_model_weights,
    load_single_case_batch,
    make_case_json_from_h5,
)


MAX_AGENT_STEPS = 3
TOOL_NAME = "finetune_registration_model_on_roi"
RESULT_OUTPUT_DIR = os.path.dirname(__file__)


AGENT_SYSTEM_PROMPT = """
You are a medical image registration agent.

You will receive a JSON state for one patient. The state contains:
1. The latest registration metrics.
2. The highest-priority organs selected from that result.
3. The action history.

Your job is to decide whether to accept the current registration or call the only available tool.

Available actions:
- finetune_registration_model_on_roi
- accept_current_model

The only tool is finetune_registration_model_on_roi.
It fine-tunes the current registration model for this one patient.
It requires mask_labels, a list of integer labels from 1 to 128.

Decision rules:
1. If the latest result is already acceptable, choose accept_current_model.
2. If important organs still have low dice_after, high tre_after, or poor improvement, choose finetune_registration_model_on_roi.
3. Choose at most 3 labels.
4. Use only organ names and labels present in selected_organs_for_llm.
5. Do not invent organ names or labels.
6. Output exactly one valid JSON object.
7. Do not output Markdown, Python code, or extra explanation.

Output format for fine-tuning:
{
  "action": "finetune_registration_model_on_roi",
  "overall_summary": "The registration improved globally, but bladder alignment remains poor.",
  "needs_update": true,
  "target_organs": [
    {
      "name": "urinary_bladder",
      "label": 21,
      "reason": "Low Dice after registration."
    }
  ],
  "tool_args": {
    "mask_labels": [21]
  },
  "reason": "The selected labels remain poorly aligned and should be refined."
}

Output format for accepting:
{
  "action": "accept_current_model",
  "overall_summary": "The current registration is acceptable.",
  "needs_update": false,
  "target_organs": [],
  "tool_args": {
    "mask_labels": []
  },
  "reason": "The current metrics are sufficient and no further refinement is needed."
}
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


def extract_json(text: str) -> Dict[str, Any]:
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


def save_agent_result(result: Dict[str, Any]) -> str:
    patient_name = os.path.splitext(os.path.basename(result["patient_path"]))[0]
    output_path = os.path.join(
        RESULT_OUTPUT_DIR,
        f"{patient_name}_agent_result.json",
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return output_path


def summarize_organs_for_llm(
    registration_json: Dict[str, Any],
    max_organs: int = DEFAULT_MAX_ORGANS_FOR_LLM,
) -> List[Dict[str, Any]]:
    organ_rows = []

    for name, item in registration_json.get("organs", {}).items():
        dice_after = float(item.get("dice_after") or 0.0)
        tre_after = float(item.get("tre_after") or 999.0)
        dice_delta = float(item.get("dice_delta") or 0.0)
        tre_delta = float(item.get("tre_delta") or 0.0)

        priority_score = 0.0
        priority_score += (1.0 - dice_after) * 2.0
        priority_score += min(tre_after / 20.0, 1.0)
        if dice_delta < 0.05:
            priority_score += 0.5
        if tre_delta > -1.0:
            priority_score += 0.3

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
                "priority_score": priority_score,
            }
        )

    organ_rows = sorted(
        organ_rows,
        key=lambda x: x["priority_score"],
        reverse=True,
    )
    return organ_rows[:max_organs]


def build_llm_state_view(state: Dict[str, Any]) -> Dict[str, Any]:
    registration_json = state["registration_json"]
    return {
        "patient_path": state["patient_path"],
        "current_step": state["current_step"],
        "max_agent_steps": state["max_agent_steps"],
        "metrics": registration_json.get("metrics", {}),
        "selected_organs_for_llm": summarize_organs_for_llm(
            registration_json,
            max_organs=state["max_organs_for_llm"],
        ),
        "history": state["history"],
    }


def validate_agent_decision(
    decision: Dict[str, Any],
    llm_state_view: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(decision, dict):
        raise ValueError("LLM decision must be a dictionary.")

    if "action" not in decision:
        raise ValueError("LLM decision missing required field: action")

    if decision["action"] not in {TOOL_NAME, "accept_current_model"}:
        raise ValueError(f"Unsupported action: {decision['action']}")

    decision.setdefault("overall_summary", "")
    decision.setdefault("needs_update", decision["action"] == TOOL_NAME)
    decision.setdefault("target_organs", [])
    decision.setdefault("tool_args", {})
    decision.setdefault("reason", "")

    available_labels = {
        int(item["label"])
        for item in llm_state_view.get("selected_organs_for_llm", [])
        if item.get("label") is not None
    }

    tool_args = decision["tool_args"]
    mask_labels = [int(label) for label in tool_args.get("mask_labels", [])]

    if decision["action"] == TOOL_NAME:
        mask_labels = [
            label for label in mask_labels
            if label in available_labels and 1 <= label <= 128
        ]
        if not mask_labels:
            raise ValueError("Tool action requires non-empty valid mask_labels.")
        tool_args["mask_labels"] = mask_labels[:3]
    else:
        tool_args["mask_labels"] = []

    return decision


def ask_llm_for_next_action(
    tokenizer,
    llm_model,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    llm_state_view = build_llm_state_view(state)
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": safe_json_dumps(llm_state_view, indent=2)},
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
    device = next(llm_model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = llm_model.generate(
            **inputs,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print("\n========== LLM Raw Output ==========")
    print(answer)
    print("====================================\n")

    return validate_agent_decision(
        extract_json(answer),
        llm_state_view=llm_state_view,
    )


def make_registration_json_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return make_case_json_from_h5(
        model=state["registration_model"],
        batch=state["patient_batch"],
        patient_path=state["patient_path"],
        device=DEVICE,
    )


def run_registration_agent(
    patient_path: str,
    max_agent_steps: int = MAX_AGENT_STEPS,
    max_organs_for_llm: int = DEFAULT_MAX_ORGANS_FOR_LLM,
) -> Dict[str, Any]:
    patient_batch = load_single_case_batch(patient_path)

    registration_model = build_registration_model().to(DEVICE)
    load_model_weights(registration_model, REGISTRATION_WEIGHTS_PATH, DEVICE)

    tokenizer, llm_model = load_qwen(LLM_MODEL_PATH)

    state = {
        "patient_path": patient_path,
        "patient_batch": patient_batch,
        "registration_model": registration_model,
        "registration_json": None,
        "current_step": 0,
        "max_agent_steps": max_agent_steps,
        "max_organs_for_llm": max_organs_for_llm,
        "history": [],
        "status": "running",
    }

    for step in range(max_agent_steps + 1):
        state["current_step"] = step
        state["registration_json"] = make_registration_json_from_state(state)

        decision = ask_llm_for_next_action(
            tokenizer=tokenizer,
            llm_model=llm_model,
            state=state,
        )

        history_item = {
            "step": step,
            "metrics_before_action": state["registration_json"].get("metrics", {}),
            "decision": decision,
        }

        if decision["action"] == "accept_current_model":
            state["history"].append(history_item)
            state["status"] = "accepted_by_llm"
            break

        if step >= max_agent_steps:
            state["history"].append(history_item)
            state["status"] = "max_steps_reached"
            break

        state["registration_model"] = finetune_registration_model_on_roi(
            patient_path=patient_path,
            mask_labels=decision["tool_args"]["mask_labels"],
            model=state["registration_model"],
            weights_path=REGISTRATION_WEIGHTS_PATH,
            device=DEVICE,
        )
        state["registration_json"] = make_registration_json_from_state(state)
        history_item["metrics_after_action"] = state["registration_json"].get("metrics", {})
        state["history"].append(history_item)

    return {
        "patient_path": state["patient_path"],
        "status": state["status"],
        "final_registration_json": state["registration_json"],
        "history": state["history"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the single-case registration agent loop."
    )
    parser.add_argument(
        "patient_path",
        type=str,
        help="Path to one patient .h5 file.",
    )

    args = parser.parse_args()
    result = run_registration_agent(args.patient_path)
    output_path = save_agent_result(result)
    print("\n========== Final Agent State ==========")
    print(safe_json_dumps(result, indent=2))
    print(f"Saved final agent state to: {output_path}")
    print("=======================================\n")
