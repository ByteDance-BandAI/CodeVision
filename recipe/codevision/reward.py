# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import string
from typing import Union, List
from collections import Counter
import math_verify
from fractions import Fraction

from verl.utils.reward_score.llmjudge import LLMJudgeClient

LLM_JUDGE_CLIENT = LLMJudgeClient(
    base_url=os.getenv("LLM_JUDGE_BASE_URL", ""),
    model_name=os.getenv("LLM_JUDGE_MODEL_NAME", None),
    temperature=0.0,
)


def extract_answer(solution_str: str) -> str | None:
    if not solution_str:
        return None

    cleaned = re.sub(r"<tool_response>.*?</tool_response>", "", solution_str, flags=re.DOTALL)
    matches = re.findall(r"<answer>(.*?)</answer>", cleaned, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()

    parts = cleaned.strip().split("<answer>")
    if len(parts) > 1:
        return parts[-1].strip()

    lines = cleaned.strip().splitlines()
    return lines[-1].strip() if lines and len(lines) < 500 else None


def verify_answer_rule(prediction_text: str, ground_truth: str) -> str | None:
    if not prediction_text:
        return None
    text = prediction_text.upper()
    is_choice_question = str(ground_truth).upper() in ["A", "B", "C", "D"]
    if is_choice_question:
        possible_options = re.findall(r"\b([A-D])\b", text)
        if possible_options:
            return possible_options[-1]
    else:
        try:
            ans = float(prediction_text)
            return str(ans)
        except:
            possible_numbers = re.findall(r"[-+]?\d*\.?\d+", text)
            if possible_numbers:
                return possible_numbers[-1]
    return None


def are_answers_equivalent(
    prediction: str, ground_truth: str, tolerance: float = 1e-4
) -> bool:
    if not isinstance(prediction, str) or not isinstance(ground_truth, str):
        return False

    if prediction is None or ground_truth is None:
        return False
    try:
        if float(prediction) == float(ground_truth):
            return True
    except:
        if prediction == ground_truth:
            return True
    extracted_pred = verify_answer_rule(str(prediction), str(ground_truth))
    if extracted_pred is None:
        return False
    gt_str = str(ground_truth).strip()
    if extracted_pred.upper() == gt_str.upper():
        return True
    try:
        if abs(float(Fraction(extracted_pred)) - float(Fraction(gt_str))) <= tolerance:
            return True
    except:
        pass
    return False


def compute_score(
    solution_str, ground_truth, data_source, extra_info
) -> tuple[float, str]:
    answer = extract_answer(solution_str)

    if answer is None:
        return 0.0, ""

    if are_answers_equivalent(answer, ground_truth):
        return 1.0, ""
    
    else:
        # llm-as-judge verify
        judge_client = LLM_JUDGE_CLIENT
        acc_score, reason = judge_client.verify(
            answer=answer,
            ground_truth=ground_truth,
            question=extra_info["question"],
            task=data_source,
        )
        return acc_score, reason