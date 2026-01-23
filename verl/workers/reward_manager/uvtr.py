# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict
from typing import Any

import torch
import re

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("uvtr")
class UVTRRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, config, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", format_reward_weight: float = 0.0) -> None:
        """
        Initialize the UVTRRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
            format_reward_weight: Weight for the format reward. Defaults to 0.0 (disabled).
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def _compute_format_reward(self, text: str) -> float:
        """Return 1.0 if the output format is valid, else 0.0.

        Rules:
        - Starts with <think>
        - Matching numbers of <think> and </think>, at least one pair
        - Exactly one <answer>...</answer>
        - The sole <answer>...</answer> is at the end (allowing trailing whitespace)
        - No overlapping of <answer> with any <think>...</think> block; and no <think>.. <answer> ..</think>
        """
        if text is None:
            return 0.0

        stripped_text = re.sub(r"<tool_response>.*?</tool_response>", "", text, flags=re.DOTALL)
        stripped_text = stripped_text.strip()
        if not stripped_text.startswith("<think>"):
            return 0.0
        open_think_tags = re.findall(r"<think>", stripped_text)
        close_think_tags = re.findall(r"</think>", stripped_text)
        if len(open_think_tags) != len(close_think_tags) or len(open_think_tags) == 0:
            return 0.0
        answer_matches = list(re.finditer(r"<answer>.*?</answer>", stripped_text, re.DOTALL))
        if len(answer_matches) != 1:
            return 0.0
        the_answer_match = answer_matches[0]
        if not re.search(r"<answer>.*?</answer>\s*$", stripped_text, re.DOTALL):
            return 0.0
        answer_start = the_answer_match.start()
        answer_end = the_answer_match.end()
        for think_match in re.finditer(r"<think>.*?</think>", stripped_text, re.DOTALL):
            think_start = think_match.start()
            think_end = think_match.end()
            if (think_start < answer_start < think_end) or (think_start < answer_end < think_end):
                if re.search(r"<think>.*?<answer>", stripped_text[think_start:think_end], re.DOTALL):
                    return 0.0
        return 1.0

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            elif isinstance(score, tuple):
                reward, reason = score
            else:
                reward = score
            
            reward_extra_info["accuracy"].append(reward)

            # Compute format reward and attach to extra info for later aggregation
            fmt_reward = self._compute_format_reward(response_str)
            reward_extra_info["format_reward"].append(fmt_reward)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
