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
import asyncio
import copy
import json
import logging
import os
from io import BytesIO
import math
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.utils.reward_score.tool_use_judge import ToolUseJudgeClient
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _to_jsonable(obj):
    try:
        import numpy as _np  # local import to avoid hard dependency at module import time
    except Exception:
        _np = None

    if _np is not None:
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, _np.generic):
            return obj.item()

    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Any,
        orgin_image_data: Any,
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.orgin_image_data = orgin_image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []
        # Track tools used across the entire loop
        self.tools_used: list[str] = []
        # Track crop boxes used by tool code, as [x1,y1,x2,y2]
        self.used_crop_boxes: list[list[float]] = []
        # Track tool execution status across the entire loop
        self.tool_exec_error_count: int = 0
        self.tool_exec_success_count: int = 0


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        cls.max_image_pixels_for_model = config.data.get("max_image_resolution", 4096 * 28 * 28)
        cls.min_image_pixels_for_model = config.data.get("min_image_resolution", 1 * 28 * 28)
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        print(f"Initialized tools: {cls.tools}")
        # Initialize tool-use judge client
        cls.tool_use_judge = ToolUseJudgeClient(
            base_url=os.getenv("TOOL_JUDGE_BASE_URL", os.getenv("LLM_JUDGE_BASE_URL", "")),
            model_name=os.getenv("TOOL_JUDGE_MODEL_NAME", os.getenv("LLM_JUDGE_MODEL_NAME", None)),
            temperature=0.0,
        )

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )
        print(f"[INFO]: system_prompt: {cls.system_prompt}")
        # Initialize interactions from config file
        cls.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if cls.interaction_config_file:
            cls.interaction_map: dict[str, BaseInteraction] = cls._initialize_interactions(cls.interaction_config_file)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        orgin_image_data = copy.deepcopy(kwargs.get("origin_multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        # Deep-copy to avoid cross-sample mutation of tool inputs
        tools_kwargs = copy.deepcopy(kwargs.get("tools_kwargs", {}))

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)

        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            orgin_image_data=orgin_image_data,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={},
        )
        # Attach per-sample constraints if present in dataset extra_info
        required_transforms = None
        gt_bbox = None
        try:
            # extra_info is expected to be available from caller kwargs when constructing raw prompt
            # In case it's not present, we keep None values
            ei = kwargs.get("extra_info", {})  # type: ignore[name-defined]
            if isinstance(ei, dict):
                required_transforms = ei.get("transform", None)
                gt_bbox = ei.get("bbox", ei.get("bbox_2d", None))
        except Exception:
            pass

        output.extra_fields.update({
            "turn_scores": json.dumps(_to_jsonable(agent_data.turn_scores), ensure_ascii=False),
            "tools_used": json.dumps(_to_jsonable(agent_data.tools_used), ensure_ascii=False),
            "tool_count": len(agent_data.tools_used),
            # Treat as used if either transforms were parsed or any code executions occurred
            "used_any_tool": (len(agent_data.tools_used) > 0) or ((agent_data.tool_exec_error_count + agent_data.tool_exec_success_count) > 0),
            # Serialize complex structures to JSON strings to keep 1D object arrays across workers
            "tool_crop_boxes": json.dumps(agent_data.used_crop_boxes, ensure_ascii=False),
            "required_transforms": json.dumps(_to_jsonable(required_transforms), ensure_ascii=False) if required_transforms is not None else None,
            "gt_bbox": json.dumps(_to_jsonable(gt_bbox), ensure_ascii=False) if gt_bbox is not None else None,
            # Tool execution success: 1 if any tool used and no errors; else 0
            "tool_exec_success": agent_data.tool_exec_success_count / (agent_data.tool_exec_error_count + agent_data.tool_exec_success_count) if (len(agent_data.tools_used) > 0 and agent_data.tool_exec_error_count + agent_data.tool_exec_success_count > 0) else 0,
            # Expose raw execution counters for code-based usage penalty
            "tool_exec_success_count": agent_data.tool_exec_success_count,
            "tool_exec_error_count": agent_data.tool_exec_error_count,
            # Total code executions (success + error)
            "code_count": agent_data.tool_exec_error_count + agent_data.tool_exec_success_count,
        })
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data, return_tensors="pt")
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        return AgentState.GENERATING

    async def _handle_generating_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        # Check termination conditions
        if len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        # Track originals for tools and resized copies for model
        new_original_images_this_turn: list[Any] = []
        new_resized_images_this_turn: list[Any] = []

        tasks = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            # Parse code_image_tool usage into semantic transforms using LLM+heuristic
            if tool_call and getattr(tool_call, "name", None):
                # Try to parse code/description from JSON args
                try:
                    args = json.loads(tool_call.arguments)
                    code = args.get("code", "")
                    description = args.get("description", "")
                except Exception:
                    code, description = "", ""

                analysis = self.tool_use_judge.analyze(code=code, description=description)
                transforms = analysis.get("transforms", []) or []
                crop_boxes = analysis.get("crop_boxes", []) or []
                # Append normalized transforms to tools_used
                for t in transforms:
                    agent_data.tools_used.append(t)
                # Append any found crop boxes
                for box in crop_boxes:
                    try:
                        if len(box) == 4:
                            agent_data.used_crop_boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
                    except Exception:
                        pass

            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs))

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        for tool_response in responses:
            # Track execution success/failure from tool response
            text_payload = getattr(tool_response, "text", None) or ""
            if isinstance(text_payload, str) and "Error" in text_payload:
                agent_data.tool_exec_error_count += 1
            else:
                agent_data.tool_exec_success_count += 1
            # Prepare valid images first so placeholders match valid image count
            valid_resized_imgs_this_tool: list[Any] = []
            valid_original_imgs_this_tool: list[Any] = []

            if tool_response.image:
                # Ensure container types
                if agent_data.orgin_image_data is None:
                    agent_data.orgin_image_data = []
                elif not isinstance(agent_data.orgin_image_data, list):
                    agent_data.orgin_image_data = [agent_data.orgin_image_data]
                if agent_data.image_data is None:
                    agent_data.image_data = []
                elif not isinstance(agent_data.image_data, list):
                    agent_data.image_data = [agent_data.image_data]

                imgs = tool_response.image if isinstance(tool_response.image, list) else [tool_response.image]
                # Sanitize originals (drop invalid), then resize
                sanitized = self._sanitize_images_for_processor(imgs)
                for img in sanitized:
                    if img is None:
                        continue
                    valid_original_imgs_this_tool.append(img)
                if valid_original_imgs_this_tool:
                    resized = self._resize_images_for_model(
                        valid_original_imgs_this_tool, self.max_image_pixels_for_model, self.min_image_pixels_for_model
                    )
                    for rimg in resized:
                        if rimg is None:
                            continue
                        valid_resized_imgs_this_tool.append(rimg)

                # Append to per-turn buffers; commit to agent_data.* only after length check passes
                for img in valid_original_imgs_this_tool:
                    new_original_images_this_turn.append(img)
                for rimg in valid_resized_imgs_this_tool:
                    new_resized_images_this_turn.append(rimg)

            # Create message from tool response with placeholders equal to valid images
            if tool_response.image or tool_response.video:
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if valid_resized_imgs_this_tool:
                    for _ in valid_resized_imgs_this_tool:
                        content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

        # Add all messages at once to avoid duplication
        agent_data.messages.extend(add_messages)

        # Update tools_kwargs with new ORIGINAL images for future tool calls
        if new_original_images_this_turn and agent_data.tools_kwargs:
            # Update tools_kwargs to include new images for tools that expect them
            for tool_name, tool_kwargs in agent_data.tools_kwargs.items():
                if "create_kwargs" in tool_kwargs and "image" in tool_kwargs["create_kwargs"]:
                    # Update the image list with new images
                    current_images = tool_kwargs["create_kwargs"]["image"]
                    if isinstance(current_images, list):
                        # Add new images to the existing list
                        tool_kwargs["create_kwargs"]["image"] = current_images + new_original_images_this_turn
                    else:
                        # Convert single image to list and add new images
                        tool_kwargs["create_kwargs"]["image"] = [current_images] + new_original_images_this_turn

        # Update prompt with tool responses.
        if self.processor is not None:
            raw_tool_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            current_images = new_resized_images_this_turn if new_resized_images_this_turn else None
            model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    add_messages, add_generation_prompt=True, tokenize=True
                ),
            )
        response_ids = response_ids[len(self.system_prompt) :]

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Commit images only if we are not terminating this turn
        if new_original_images_this_turn:
            if agent_data.orgin_image_data is None:
                agent_data.orgin_image_data = []
            for img in new_original_images_this_turn:
                agent_data.orgin_image_data.append(img)
        if new_resized_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            for rimg in new_resized_images_this_turn:
                agent_data.image_data.append(rimg)
        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )

        agent_data.user_turns += 1
        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Append the user interaction message to the conversation
        agent_data.messages.extend(add_messages)

        # Update prompt with user responses.
        if self.processor is not None:
            raw_user_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_user_response], images=None, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    add_messages, add_generation_prompt=True, tokenize=True
                ),
            )

        response_ids = response_ids[len(self.system_prompt) :]

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(self, tool_call: FunctionCall, tools_kwargs: dict[str, Any]) -> ToolResponse:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, _, _ = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return ToolResponse(
                text=f"Error when executing tool: {e}",
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs)

    def _resize_images_for_model(
        self,
        images: list[Any],
        max_pixels: int = 4096 * 28 * 28,
        min_pixels: int = 28 * 28,
    ) -> list[Any]:
        """
        Resize images to satisfy both a minimum total pixel count and a maximum pixel limit.
        
        Args:
            images: List of PIL Images or image objects
            max_pixels: Maximum number of pixels allowed (default: 4096 * 28 * 28)
            min_pixels: Minimum total pixels (default: 28 * 28)
            
        Returns:
            List of resized images
        """
        if not images:
            return images
            
        resized_images = []
        for img in images:
            if hasattr(img, 'size'):  # PIL Image
                width, height = img.size
                # Skip invalid zero-sized images
                if width < 1 or height < 1:
                    logger.warning("Skipping invalid image with non-positive dimensions: %s", img)
                    continue
                # Step 1: ensure minimum total pixels (scale up while keeping aspect ratio)
                current_pixels = width * height
                new_width, new_height = width, height
                if current_pixels < min_pixels and current_pixels > 0:
                    scale_up = (min_pixels / current_pixels) ** 0.5
                    new_width = max(1, int(math.ceil(width * scale_up)))
                    new_height = max(1, int(math.ceil(height * scale_up)))

                # Step 2: ensure max pixel limit (scale down if needed)
                new_pixels = new_width * new_height
                if new_pixels > max_pixels and new_pixels > 0:
                    scale_down = (max_pixels / new_pixels) ** 0.5
                    new_width = max(1, int(new_width * scale_down))
                    new_height = max(1, int(new_height * scale_down))

                if new_width != width or new_height != height:
                    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    resized_images.append(resized_img)
                else:
                    resized_images.append(img)
            else:
                # If it's not a PIL Image, pass through unchanged
                resized_images.append(img)
                
        return resized_images

    def _sanitize_images_for_processor(self, images: list[Any]) -> list[Any]:
        """Remove or fix images with invalid shapes to avoid downstream processor errors.

        - Drops images with width or height <= 0
        - Attempts to decode bytes into PIL images
        - Leaves non-PIL/non-array inputs unchanged if dimensions cannot be checked
        """
        if not images:
            return []

        sanitized: list[Any] = []
        for img in images:
            try:
                # Decode bytes into PIL if needed
                if isinstance(img, (bytes, bytearray)):
                    try:
                        pil_img = Image.open(BytesIO(img))
                        pil_img.load()
                        img = pil_img
                    except Exception as e:
                        logger.warning("Dropping invalid image bytes: %s", e)
                        continue

                if hasattr(img, 'size'):
                    width, height = img.size
                    if width > 0 and height > 0:
                        # Drop extreme aspect ratios (Qwen2VL requires abs aspect ratio < 200)
                        aspect = max(width / float(height), height / float(width))
                        if aspect >= 200.0:
                            logger.warning("Dropping image due to extreme aspect ratio: %sx%s (%.2f)", width, height, aspect)
                            continue
                        sanitized.append(img)
                    else:
                        logger.warning("Dropping image with non-positive size: %sx%s", width, height)
                    continue

                # Handle array/tensor-like inputs with a shape
                shape = getattr(img, 'shape', None)
                if shape is not None and len(shape) >= 2:
                    height = shape[-2]
                    width = shape[-1]
                    if int(height) > 0 and int(width) > 0:
                        # Drop extreme aspect ratios for array-like inputs
                        try:
                            aspect = max(float(width) / float(height), float(height) / float(width))
                            if aspect >= 200.0:
                                logger.warning("Dropping array image due to extreme aspect ratio: %s", shape)
                                continue
                        except Exception:
                            pass
                        sanitized.append(img)
                    else:
                        logger.warning("Dropping array image with non-positive shape: %s", shape)
                    continue

                # Unknown type; keep as-is
                sanitized.append(img)
            except Exception as e:
                logger.warning("Dropping image due to sanitize error: %s", e)
                continue

        return sanitized

    @classmethod
    def _initialize_interactions(cls, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        logger.info(f"Initialize interactions from configuration: interaction_map: {list(interaction_map.keys())}")
        return interaction_map
