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
import json
import random
import requests
import logging
from pathlib import Path
from textwrap import dedent
from typing import Tuple, Optional, Dict, Any, List, Union
from openai import OpenAI, DefaultHttpxClient
import time
import re

# Configure logging
logger = logging.getLogger(__name__)


class LLMJudgeClient:
    """
    A unified LLM-based judge client for evaluating model outputs.
    
    Features:
    - Consistent 0/1 scoring across all task types
    - Unified parsing and evaluation logic
    - Support for multiple task types (OCR, Math, Code, General)
    - Robust error handling and retry logic
    - Clean, maintainable code structure
    """
    
    # Default prompts for different task types
    DEFAULT_PROMPTS = {
        'ocr': dedent("""
            You are a strict text content auditor. Your task is to determine if the predicted text accurately reproduces the ground truth text.
            
            Rules:
            1. Ignore case differences (e.g., "Apple" = "apple")
            2. Ignore spacing around punctuation (e.g., "hello , world" = "hello, world")
            3. Focus on content accuracy
            
            Ground Truth: {ground_truth}
            Predicted: {answer}
            
            Response with ONLY "1" if correct, "0" if incorrect:
        """).strip(),
        
        'math': dedent("""
            You are a math expert evaluator. Determine if the model's answer is mathematically correct.
            
            Rules:
            1. Focus on mathematical correctness
            2. Accept equivalent forms (e.g., "2.0" = "2", "1/2" = "0.5")
            3. Ignore minor formatting differences
            
            Question: {question}
            Expected Answer: {ground_truth}
            Model Answer: {answer}
            
            Response with ONLY "1" if correct, "0" if incorrect:
        """).strip(),
        
        'code': dedent("""
            You are a code evaluator. Determine if the model's code solution is correct.
            
            Rules:
            1. Focus on functional correctness
            2. Accept different but equivalent implementations
            3. Consider edge cases handling
            
            Question: {question}
            Expected Solution: {ground_truth}
            Model Solution: {answer}
            
            Response with ONLY "1" if correct, "0" if incorrect:
        """).strip(),
        
        'general': dedent("""
            You are an expert evaluator. Determine if the model's answer is correct based on the standard answer.
            
            Rules:
            1. Focus on factual accuracy and semantic equivalence
            2. Accept reasonable paraphrasing
            3. Ensure the answer fully addresses the question
            
            Question: {question}
            Standard Answer: {ground_truth}
            Model Answer: {answer}
            
            Response with ONLY "1" if correct, "0" if incorrect:
        """).strip()
    }
    
    # Task type mappings
    TASK_MAPPINGS = {
        'ocr': ['ocr', 'optical', 'text_recognition', 'IAM-line', 'CASIA', 'Hiertext'],
        'math': ['math', 'mathematics', 'arithmetic', 'algebra', 'geometry', 'calculus'],
        'code': ['code', 'programming', 'coding', 'algorithm', 'implementation'],
        'general': ['general', 'default', 'standard', 'qa', 'question_answering']
    }
    
    def __init__(self, base_url: str, model_name: Optional[str] = None, 
                 temperature: float = 0.0, api_key: str = "EMPTY", 
                 timeout: int = 100, max_retries: int = 3,
                 config_dir: Optional[str] = None,
                 use_external_prompts: bool = True):
        """
        Initialize the LLM Judge Client.
        
        Args:
            base_url: The base URL for the LLM API
            model_name: Optional model name to use
            temperature: Temperature for LLM generation (default 0.0 for deterministic)
            api_key: API key (default "EMPTY" for local models)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            config_dir: Directory containing prompt configurations
            use_external_prompts: Whether to load prompts from external files
        """
        self.base_url = base_url
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = None
        self.prompts = self.DEFAULT_PROMPTS.copy()
        
        if not base_url:
            logger.warning("No base_url provided. LLMJudgeClient will not be functional.")
            return
            
        self._initialize_client(api_key, model_name, timeout)
        
        if use_external_prompts and config_dir:
            self._load_external_prompts(config_dir)

    def _initialize_client(self, api_key: str, model_name: Optional[str], timeout: int):
        """Initialize the OpenAI client."""
        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url=self.base_url,
                http_client=DefaultHttpxClient(trust_env=False, timeout=timeout),
            )
            
            self.model_name = model_name or self._get_model_name()
            logger.info(f"LLMJudgeClient initialized: {self.base_url}, model={self.model_name}")
                       
        except Exception as e:
            logger.error(f"Failed to initialize LLMJudgeClient: {e}")
            raise
    
    def _get_model_name(self) -> str:
        """Get the model name from the API."""
        try:
            sess = requests.Session()
            sess.trust_env = False
            resp = sess.get(f"{self.base_url}/models", timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            if not data.get("data"):
                raise ValueError("No models found in API response")
                
            if len(data["data"]) == 1:
                return data["data"][0]["id"]
            else:
                available_models = [m["id"] for m in data["data"]]
                raise ValueError(f"Multiple models found: {available_models}. Please specify model_name.")
                
        except Exception as e:
            logger.error(f"Failed to get model name: {e}")
            raise ValueError(f"Could not determine model name: {e}")
    
    def _load_external_prompts(self, config_dir: Union[str, Path]):
        """Load external prompt templates if available."""
        config_path = Path(config_dir)
        
        # Mapping of task types to config files
        config_files = {
            'ocr': 'ocr_prompt.txt',
            'math': 'math_prompt.txt',
            'code': 'code_prompt.txt',
            'general': 'general_prompt.txt'
        }
        
        for task_type, filename in config_files.items():
            file_path = config_path / filename
            if file_path.exists():
                try:
                    prompt = file_path.read_text(encoding='utf-8').strip()
                    # Convert old format to new format if needed
                    prompt = self._convert_prompt_format(prompt)
                    self.prompts[task_type] = prompt
                    logger.info(f"Loaded external prompt for {task_type} from {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to load prompt from {file_path}: {e}")

    def _convert_prompt_format(self, prompt: str) -> str:
        """Convert old prompt formats to unified format."""
        # Ensure prompt asks for 0/1 response
        if "yes/no" in prompt.lower() or "verdict" in prompt.lower():
            prompt = re.sub(r'(?:yes/no|verdict.*?):', 
                           'Response with ONLY "1" if correct, "0" if incorrect:', 
                           prompt, flags=re.IGNORECASE)
        
        # Standardize variable names
        prompt = prompt.replace("{predicted_text}", "{answer}")
        prompt = prompt.replace("{Standard Answer}", "{ground_truth}")
        prompt = prompt.replace("{Model Answer}", "{answer}")
        
        return prompt

    def _make_request_with_retry(self, messages: List[Dict], max_tokens: int = 10) -> str:
        """Make a request with retry logic and exponential backoff."""
        if not self.client:
            raise RuntimeError("Client not initialized")
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=self.temperature
                )
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.warning(f"Request attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                # Exponential backoff with jitter
                time.sleep((2 ** attempt) + random.uniform(0, 1))
    
    def _parse_score(self, response: str) -> Tuple[float, str]:
        """
        Parse LLM response to extract score.
        Handles various response formats and returns (score, error_message).
        """
        response = response.strip().lower()
        
        # Remove common prefixes
        prefixes_to_remove = ['judgment:', 'judgement:', 'verdict:', 'answer:', 'response:']
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Direct score extraction
        if response == '1' or response == '1.0':
            return 1.0, ""
        elif response == '0' or response == '0.0':
            return 0.0, ""
        
        # Handle yes/no responses (for backward compatibility)
        if response in ['yes', 'correct', 'true', 'right']:
            return 1.0, ""
        elif response in ['no', 'incorrect', 'false', 'wrong']:
            return 0.0, ""
        
        # Try to find score in response
        score_patterns = [
            r'\b1\b(?![\d.])',  # Match standalone 1
            r'\b0\b(?![\d.])',  # Match standalone 0
            r'score[:\s]*([01])',  # Match "score: 1" or "score: 0"
            r'([01])[\s]*(?:out of|/)\s*1',  # Match "1 out of 1" or "0/1"
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, response)
            if match:
                score = match.group(1) if match.groups() else match.group(0)
                return float(score), ""
        
        # If we can't parse, default to 0 with error message
        logger.warning(f"Could not parse score from response: {response}")
        return 0.0, f"Failed to parse score from: {response}"
    
    def _get_task_type(self, task: str) -> str:
        """Determine the task type from the task name."""
        task_lower = task.lower()
        
        for task_type, keywords in self.TASK_MAPPINGS.items():
            if any(keyword in task_lower for keyword in keywords):
                return task_type
        
        return 'general'  # Default fallback
    
    def verify(self, answer: str, ground_truth: str, question: str = "", 
               task: str = 'general') -> Tuple[float, str]:
        """
        Main verification method with unified scoring.
        
        Args:
            answer: The model's answer to evaluate
            ground_truth: The correct/expected answer
            question: The original question (optional for some tasks)
            task: The task type or name
            
        Returns:
            Tuple of (score, error_message) where score is 0.0 or 1.0
        """
        if not self.client:
            logger.error("Client not initialized")
            return 0.0, "Client not initialized"
        
        try:
            # Determine task type
            task_type = self._get_task_type(task)
            
            # Get appropriate prompt template
            prompt_template = self.prompts.get(task_type, self.prompts['general'])
            
            # Build prompt with available variables
            prompt_vars = {
                'answer': answer,
                'ground_truth': ground_truth,
                'question': question
            }
            
            # Only include variables that exist in the template
            prompt = prompt_template
            for var, value in prompt_vars.items():
                if f'{{{var}}}' in prompt:
                    prompt = prompt.replace(f'{{{var}}}', str(value))
            
            # Make request
            response = self._make_request_with_retry(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10
            )
            
            # Parse response
            return self._parse_score(response)
            
        except Exception as e:
            logger.error(f"Verification failed for task {task}: {e}")
            return 0.0, str(e)
    
    def get_prompt(self, task: str = 'general') -> str:
        """Get the prompt template for a given task type."""
        task_type = self._get_task_type(task)
        return self.prompts.get(task_type, self.prompts['general'])
    
    def set_prompt(self, task: str, prompt: str):
        """Set a custom prompt for a task type."""
        task_type = self._get_task_type(task)
        self.prompts[task_type] = self._convert_prompt_format(prompt)
    
    # Simplified legacy support methods
    def verify_ocr(self, answer: str, ground_truth: str) -> Tuple[float, str]:
        """Legacy OCR verification method."""
        return self.verify(answer, ground_truth, "", "ocr")
    
    def verify_general(self, answer: str, ground_truth: str, question: str) -> Tuple[float, str]:
        """Legacy general verification method."""
        return self.verify(answer, ground_truth, question, "general")
    
    def verify_math(self, answer: str, ground_truth: str, question: str) -> Tuple[float, str]:
        """Legacy math verification method."""
        return self.verify(answer, ground_truth, question, "math")
    
    def verify_code(self, answer: str, ground_truth: str, question: str) -> Tuple[float, str]:
        """Legacy code verification method."""
        return self.verify(answer, ground_truth, question, "code")