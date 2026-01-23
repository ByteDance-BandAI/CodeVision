import logging
import copy
import os
import re
from collections import defaultdict
from typing import Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset.rl_dataset import RLHFDataset


class CustomRLHFDataset(RLHFDataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.auto_generate_code_image_tool_kwargs = config.get("auto_generate_code_image_tool_kwargs", True)
        self.code_image_tool_name = config.get("code_image_tool_name", "code_image_tool")
        self.enable_mm_resize = config.get("enable_mm_resize", False)
        self.max_image_resolution = config.get("max_image_resolution", 4096 * 28 * 28)
        self.min_image_resolution = config.get("min_image_resolution", 1 * 28 * 28)
        self.allow_heterogeneous_schemas = config.get("allow_heterogeneous_schemas", False)

        self.replace_system_prompt = config.get("replace_system_prompt", False)
        if self.replace_system_prompt:
            # Load new system prompt from file
            self.new_sp_path = config.get("new_sp_path", "")
            if not os.path.exists(self.new_sp_path):
                raise ValueError(f"new_sp_path {self.new_sp_path} does not exist")
            print(f"Warning: replace system prompt is enabled, will replace the system prompt with {self.new_sp_path}")
            with open(self.new_sp_path, 'r') as f:
                self.new_sp = f.read()
            print(f"new_sp: {self.new_sp}")

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        if self.allow_heterogeneous_schemas:
            import pandas as pd

            print(
                f"`allow_heterogeneous_schemas` is enabled, it supports input files with different column names. "
                f"For any column that is missing in a given row, the value will be filled with `None`."
            )

            dataframes = []
            for parquet_file in self.data_files:
                dataframe = pd.read_parquet(parquet_file, use_threads=True)
                dataframes.append(dataframe)
            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df.replace({np.nan: None}, inplace=True)  # replace NaN with None

            self.dataframe: list[dict] = combined_df.to_dict(orient="records")
            print(f"dataset columns: {combined_df.columns.tolist()}")
        else:
            dataframes = []
            for parquet_file in self.data_files:
                # read parquet files and cache
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
                dataframes.append(dataframe)
            self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        self.dataframe = self.maybe_resize_mm_data(self.dataframe)
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                    )
                    images = (
                        [process_image(image) for image in doc[image_key]]
                        if image_key in doc and doc[image_key]
                        else None
                    )
                    videos = (
                        [process_video(video) for video in doc[video_key]]
                        if video_key in doc and doc[video_key]
                        else None
                    )

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    return len(
                        tokenizer.apply_chat_template(
                            doc[prompt_key], add_generation_prompt=True, **self.apply_chat_template_kwargs
                        )
                    )

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = list(example.get(self.prompt_key))

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def _replace_system_prompt(self, messages: list[dict]) -> None:
        """
        Replace system prompt in messages with the new system prompt.
        
        Args:
            messages: List of message dictionaries to modify in-place
        """
        # Count system messages
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        num_system_messages = len(system_messages)
        
        if num_system_messages == 0:
            # No system message found, insert at the beginning
            messages.insert(0, {"role": "system", "content": self.new_sp})
        elif num_system_messages == 1:
            # One system message found, replace it
            if messages[0]["role"] != "system":
                raise ValueError(
                    "The first message must be system when replace_system_prompt is True"
                )
            messages[0]["content"] = self.new_sp
        else:
            # Multiple system messages found, which is not allowed
            raise ValueError(f"Found {num_system_messages} system messages, expected 0 or 1")
    
    def maybe_resize_mm_data(self, dataframe: Union[datasets.Dataset, list[dict]] = None):

        def _add_pixel_example(example):
            imgs = example.get('images', None)
            if imgs:
                new_imgs = []
                for img_dict in imgs:
                    img_dict['max_pixels'] = self.max_image_resolution
                    img_dict['min_pixels'] = self.min_image_resolution
                    new_imgs.append(img_dict)
                example['images'] = new_imgs
            return example

        if self.enable_mm_resize:
            print(f"Enabling multimodal data resize, the max resolution for the image is {self.max_image_resolution=}.")
            if isinstance(dataframe, datasets.Dataset):
                dataframe = dataframe.map(_add_pixel_example, num_proc=self.num_workers, desc="Adding pixel info")
            else:
                dataframe = [_add_pixel_example(row) for row in dataframe]

        return dataframe

    def _generate_code_image_tool_kwargs(self, images):
        """
        Generate tools_kwargs for CodeImageTool based on available images.
        
        Args:
            images: List of processed images or single image
            
        Returns:
            dict: tools_kwargs structure for CodeImageTool
        """
        # CodeImageTool expects image to be a list
        if isinstance(images, list) and len(images) > 0:
            # Already a list, use as is
            image_data = images
        elif images is not None:
            # Single image, wrap in list
            image_data = [images]
        else:
            # No images available
            return {}
        
        # Generate tools_kwargs for CodeImageTool
        tools_kwargs = {
            self.code_image_tool_name: {
                "create_kwargs": {
                    "image": image_data
                }
            }
        }
        
        return tools_kwargs

    def _process_image_original(self, image: dict) -> Image.Image:
        """
        Process image without resizing constraints for cropping purposes.
        
        Args:
            image: Image dictionary containing image data
            
        Returns:
            PIL Image without resizing constraints
        """
        from verl.utils.dataset.vision_utils import process_image
        
        # Create a copy of the image dict to avoid modifying the original
        image_copy = image.copy()
        
        # Remove any resizing constraints that might have been added
        image_copy.pop('max_pixels', None)
        image_copy.pop('min_pixels', None)
        
        # Process the image without resizing constraints
        return process_image(image_copy)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.replace_system_prompt:
            self._replace_system_prompt(messages)

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            multi_modal_data = {}

            images = None
            origin_images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                origin_images = [self._process_image_original(image) for image in row_dict_images]
                images = [process_image(image) for image in row_dict_images]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            row_dict_videos = row_dict.pop(self.video_key, None)
            if row_dict_videos:
                videos = [process_video(video) for video in row_dict_videos]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]
            
            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data
            
            # Store original unresized images for cropping purposes
            if origin_images is not None:
                origin_multi_modal_data = {"image": origin_images}
                row_dict["origin_multi_modal_data"] = origin_multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            if self.apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                    "models like GLM can copy chat_template.jinja from instruct models"
                )
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        elif (
            self.processor is not None
            and hasattr(self.processor, "image_processor")
            and "Qwen3VL" in self.processor.image_processor.__class__.__name__
        ):
            from verl.models.transformers.qwen3_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)

            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        
        # Auto-generate tools_kwargs for CodeImageTool if images are present and tools_kwargs is empty
        if (self.auto_generate_code_image_tool_kwargs and 
            not tools_kwargs and 
            multi_modal_data and 
            "image" in multi_modal_data):
            images_for_tool = multi_modal_data["image"]
            if "origin_multi_modal_data" in row_dict and "image" in row_dict["origin_multi_modal_data"]:
                images_for_tool = row_dict["origin_multi_modal_data"]["image"]
            tools_kwargs = self._generate_code_image_tool_kwargs(images_for_tool)
        
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()