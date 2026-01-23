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

import ast
import logging
import os
import sys
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray
import ray.actor
from qwen_vl_utils import fetch_image

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class CodeExecutionWorker:
    """Worker for executing code-based image processing operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="code-rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    logger.warning(f"Error when executing code-based image processing: {e}")
        else:
            return fn(*fn_args, **fn_kwargs)


def init_code_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize code execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(CodeExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class CodeImageTool(BaseTool):
    """A unified tool for image processing using executable Python code.

    This tool allows MLLM to write Python code to perform various image operations
    including zoom, flip, rotate, contrast, brightness adjustments, and more.
    The tool provides a safe execution environment with predefined image processing
    libraries and utilities.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the image processing code
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 20)
        self.rate_limit = config.get("rate_limit", 50)
        self.timeout = config.get("timeout", 30)
        self.max_code_length = config.get("max_code_length", 2000)

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_code_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        logger.info(f"Initialized CodeImageTool with config: {config}")

    def _validate_image_size(self, image: Any) -> tuple[bool, Optional[int], Optional[int], Optional[str]]:
        """Validate output image dimensions and aspect ratio, consistent with _sanitize_images_for_processor.

        Rules:
        - width > 0 and height > 0
        - extreme aspect ratio is invalid: max(w/h, h/w) >= 200.0

        Returns: (is_valid, width, height, error_message)
        """
        try:
            width_val: Optional[int] = None
            height_val: Optional[int] = None
            # PIL Image-like
            if hasattr(image, "size"):
                w, h = image.size
                width_val = int(w)
                height_val = int(h)
            else:
                # Array/tensor-like
                shape = getattr(image, "shape", None)
                if shape is not None and len(shape) >= 2:
                    height_val = int(shape[-2])
                    width_val = int(shape[-1])

            if width_val is None or height_val is None:
                return True, width_val, height_val, None  # Unknown type; treat as valid to avoid false negatives
            if width_val <= 0 or height_val <= 0:
                return False, width_val, height_val, f"The result has an invalid image size ({width_val}x{height_val}). Width and height must be positive. Please check your code."
            # Check extreme aspect ratio (align with _sanitize_images_for_processor threshold)
            try:
                aspect = max(float(width_val) / float(height_val), float(height_val) / float(width_val))
                if aspect >= 200.0:
                    return False, width_val, height_val, f"The result has an invalid image aspect ratio ({width_val}x{height_val}, aspect={aspect:.2f}) >= 200.0. Please check your code."
            except Exception:
                pass
            return True, width_val, height_val, None
        except Exception as e:
            return False, None, None, f"Failed to validate image dimensions: {str(e)}. Please check your code."

    def _validate_code(self, code: str) -> tuple[bool, str]:
        """Validate the Python code for safety and syntax."""
        try:
            # Check code length
            if len(code) > self.max_code_length:
                return False, f"Code too long. Maximum allowed length: {self.max_code_length}"

            # Parse the code to check syntax
            tree = ast.parse(code)
            
            # Check for dangerous operations
            dangerous_imports = [
                'os', 'sys', 'subprocess', 'eval', 'exec', 'compile',
                'open', 'file', 'input', 'raw_input', '__import__'
            ]
            
            dangerous_functions = [
                'exit', 'quit', 'help', 'dir', 'vars', 'globals', 'locals'
            ]
            
            for node in ast.walk(tree):
                # Check for dangerous imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in dangerous_imports:
                            return False, f"Dangerous import detected: {alias.name}"
                
                if isinstance(node, ast.ImportFrom):
                    if node.module in dangerous_imports:
                        return False, f"Dangerous import detected: {node.module}"
                
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in dangerous_functions:
                        return False, f"Dangerous function call detected: {node.func.id}"
                
                # Check for attribute access to dangerous modules
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name) and node.value.id in dangerous_imports:
                        return False, f"Dangerous attribute access detected: {node.value.id}.{node.attr}"
            
            return True, "Code validation passed"
            
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Code validation error: {e}"

    def _create_safe_globals(self, image) -> dict:
        """Create a safe global namespace for code execution."""
        import PIL
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
        import numpy as np
        import math
        
        # Try to import OpenCV, but don't fail if it's not available
        try:
            import cv2
            cv2_available = True
        except ImportError:
            cv2 = None
            cv2_available = False
        
        allowed_imports = {"PIL", "PIL.Image", "PIL.ImageEnhance", "PIL.ImageFilter", "PIL.ImageOps",
                       "PIL.ImageDraw", "PIL.ImageFont", "numpy", "math", "cv2"}

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if any(name == mod or name.startswith(mod + ".") for mod in allowed_imports):
                return __import__(name, globals, locals, fromlist, level)
            raise ImportError(f"Import of '{name}' is not allowed in safe mode")
        
        # Create a copy of the image to avoid modifying the original
        image_copy = image.copy()
        
        safe_globals = {
            '__builtins__': {
                'len': len, 'range': range, 'enumerate': enumerate, 'zip': zip,
                'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
                'int': int, 'float': float, 'str': str, 'bool': bool,
                'list': list, 'tuple': tuple, 'dict': dict, 'set': set,
                'print': print, 'isinstance': isinstance, 'type': type,
                'hasattr': hasattr, 'getattr': getattr, 'setattr': setattr,
                'math': math, 'np': np,
                '__import__': safe_import,
            },
            # Image processing libraries
            'PIL': PIL,
            'Image': Image,
            'ImageEnhance': ImageEnhance,
            'ImageFilter': ImageFilter,
            'ImageOps': ImageOps,
            'ImageDraw': ImageDraw,
            'ImageFont': ImageFont,
            'numpy': np,
            'math': math,
            # The input image
            'image': image_copy,
            'img': image_copy,  # Alias for convenience
            # Common constants
            'PI': math.pi,
            'E': math.e,
            # Drawing utilities
            'draw': ImageDraw.Draw(image_copy),
        }
        
        # Add OpenCV if available
        if cv2_available:
            safe_globals['cv2'] = cv2
            safe_globals['cv'] = cv2  # Alias for convenience
        
        return safe_globals

    def _execute_code(self, code: str, image) -> tuple[Any, str]:
        """Execute the provided code safely and return the result."""
        try:
            # Validate code first
            is_valid, validation_msg = self._validate_code(code)
            if not is_valid:
                return None, f"Code validation failed: {validation_msg}"
            
            # Create safe execution environment
            safe_globals = self._create_safe_globals(image)
            safe_locals = {}
            
            # Execute the code
            exec(code, safe_globals, safe_locals)
            
            # Try to get the result from common variable names
            result = None
            for var_name in ['result', 'output', 'processed_image', 'img', 'image']:
                if var_name in safe_locals:
                    result = safe_locals[var_name]
                    break
                elif var_name in safe_globals and safe_globals[var_name] is not image:
                    result = safe_globals[var_name]
                    break
            
            # If no result found, return the modified image
            if result is None:
                result = safe_globals.get('image', image)
            
            return result, "Code executed successfully"
            
        except Exception as e:
            return None, f"Code execution error: {str(e)}"

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """
        Creates a new instance for code-based image processing tool.

        Args:
            instance_id: An optional unique identifier for the instance.
            **kwargs: Should contain 'image' key with image data (single image or list of images).

        Returns:
            Tuple of (instance_id, ToolResponse)
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Handle create_kwargs parameter if passed
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        # Get image from kwargs
        image = kwargs.get("image")
        if image is None:
            raise ValueError("Missing required 'image' parameter in kwargs")

        # Handle both single image and list of images
        if isinstance(image, list):
            # Process list of images
            images = []
            for img_data in image:
                img = fetch_image({"image": img_data})
                images.append(img)
        else:
            # Single image
            img = fetch_image({"image": image})
            images = [img]

        self._instance_dict[instance_id] = {
            "images": images,
            "response": "",
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """
        Execute the image processing code.

        Args:
            instance_id: The instance id of the tool.
            parameters: Dictionary containing 'code', 'description', and 'image_index' parameters.

        Returns:
            Tuple of (tool_response, tool_reward_score, tool_metrics)
        """
        code = parameters.get("code", "")
        image_index = parameters.get("image_index", 0)
        
        if not code or not isinstance(code, str):
            return (
                ToolResponse(text="Error: 'code' parameter is missing or not a string. Please keep thinking step-by-step inside the <think></think> tags to determine the next action. If additional tools are required, call them inside the <tool_calls></tool_calls> tags. Otherwise, provide your final answer within the <answer></answer> tags."),
                -0.05,
                {"success": False, "error": "missing_code"},
            )

        if not isinstance(image_index, int) or image_index < 0:
            return (
                ToolResponse(text="Error: 'image_index' must be a non-negative integer. Please keep thinking step-by-step inside the <think></think> tags to determine the next action. If additional tools are required, call them inside the <tool_calls></tool_calls> tags. Otherwise, provide your final answer within the <answer></answer> tags."),
                -0.05,
                {"success": False, "error": "invalid_image_index"},
            )

        if instance_id not in self._instance_dict:
            return (
                ToolResponse(text="Error: Instance not found. Please create an instance first."),
                -0.05,
                {"success": False, "error": "instance_not_found"},
            )

        instance_data = self._instance_dict[instance_id]
        images = instance_data["images"]
        
        # Validate image_index
        if image_index >= len(images):
            return (
                ToolResponse(text=f"Error: image_index {image_index} is out of range. Available images: 0 to {len(images)-1}. Please keep thinking step-by-step inside the <think></think> tags to determine the next action. If additional tools are required, call them inside the <tool_calls></tool_calls> tags. Otherwise, provide your final answer within the <answer></answer> tags."),
                -0.05,
                {"success": False, "error": "image_index_out_of_range"},
            )
        
        # Select the image to process
        image = images[image_index]

        try:
            # Execute the code using the worker pool
            result, message = ray.get(
                self.execution_pool.execute.remote(self._execute_code, code, image)
            )
            
            if result is None:
                return (
                    ToolResponse(text=f"Error: {message}. Please keep thinking step-by-step inside the <think></think> tags to determine the next action. If additional tools are required, call them inside the <tool_calls></tool_calls> tags. Otherwise, provide your final answer within the <answer></answer> tags."),
                    -0.05,
                    {"success": False, "error": "execution_failed", "message": message},
                )
            
            # Ensure result is a PIL Image
            if hasattr(result, 'save'):  # It's already a PIL Image
                processed_image = result
            else:
                return (
                    ToolResponse(text="Error: Code must return a PIL Image object."),
                    -0.05,
                    {"success": False, "error": "invalid_return_type"},
                )

            # Validate processed image size (output image)
            is_valid, w, h, err = self._validate_image_size(processed_image)
            if not is_valid:
                return (
                    ToolResponse(text=f"Error: {err}" if err else "Error: Invalid output image size."),
                    -0.05,
                    {"success": False, "error": "invalid_output_image_size", "width": w, "height": h, "processed_image_index": image_index},
                )
            
            response_text = f"Here is the processed image. Now, analyze the returned results. Please keep thinking step-by-step inside the <think></think> tags to determine the next action. If additional tools are required, call them inside the <tool_calls></tool_calls> tags. Otherwise, provide your final answer within the <answer></answer> tags."
            
            return (
                ToolResponse(
                    image=[processed_image],
                    text=response_text,
                ),
                0.0,
                {"success": True, "message": message, "processed_image_index": image_index, "total_images": len(images)},
            )
            
        except Exception as e:
            logger.error(f"Error in code-based image processing on image {image_index}: {e}")
            return (
                ToolResponse(text=f"Error processing image {image_index}: {str(e)}. Please keep thinking step-by-step inside the <think></think> tags to determine the next action. If additional tools are required, call them inside the <tool_calls></tool_calls> tags. Otherwise, provide your final answer within the <answer></answer> tags."),
                -0.05,
                {"success": False, "error": "unexpected_error", "message": str(e), "processed_image_index": image_index},
            )

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
