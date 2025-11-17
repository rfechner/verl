# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
import logging
import os
import re
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)
datasets.disable_caching()

def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
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
        log_prob_from_chat=False,
        log_prob_from_rollouts_dir=False,
        log_prob_from_chat_file=False
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
        self.max_response_length = config.get("max_response_length", 3072) # only used in case log_prob_from_chat=True
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
        
        # flags for special data loading
        self.log_prob_from_chat = log_prob_from_chat
        self.log_prob_from_rollouts_dir = log_prob_from_rollouts_dir

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            if self.log_prob_from_chat or \
                self.log_prob_from_rollouts_dir:
                dataframe = datasets.load_dataset("json", data_files=parquet_file)["train"] # logprob are stored as jsonl for fast iteration.
            else:
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        if self.log_prob_from_rollouts_dir:
            return # don't have to filter out long prompts.
        
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
        messages: list = example.pop(self.prompt_key)

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

    def get_item_from_rollouts_dir_source(self, row_dict):
        """
            Easier, because we've already applied the tokenizers etc onto the data.
            Just have to re-tokenize and calculate the masks.
        """
        question, response = row_dict['input'], row_dict['output']

        prompt_tokenized = self.tokenizer(question, return_tensors='pt', add_special_tokens=False)
        response_tokenized = self.tokenizer(response, return_tensors='pt', add_special_tokens=False)

        prompt_input_ids = prompt_tokenized.pop("input_ids")
        prompt_attention_mask = prompt_tokenized.pop("attention_mask")
        response_input_ids = response_tokenized.pop('input_ids')
        response_attention_mask = response_tokenized.pop('attention_mask')

        prompt_input_ids, prompt_attention_mask = verl_F.postprocess_data(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        response_input_ids, response_attention_mask = verl_F.postprocess_data(
            input_ids=response_input_ids,
            attention_mask=response_attention_mask,
            max_length=self.max_response_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=False,
            truncation=self.truncation,
        )

        full_input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=-1)
        full_attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)
        full_position_ids = compute_position_id_with_mask(full_attention_mask)

        tensors = {
            'input_ids' : full_input_ids.squeeze(),
            'attention_mask' : full_attention_mask.squeeze(),
            'position_ids' : full_position_ids.squeeze(),
            'prompts' : prompt_input_ids.squeeze(),
            'responses' : response_input_ids.squeeze(),
            'response_mask' : response_attention_mask.squeeze()
        }
        non_tensors = {
            'data_global_step' :  row_dict['step'],
            'input' : question,
            'output' : response,
            'score' :  row_dict['score'],
            'reward' : row_dict['reward']
        }

        return {**tensors, **non_tensors}
    
    def get_item_from_chat_with_suffix_source(self, row_dict):
        """
            chat with suffix file schema:

            prompt "Whats 2+2?"
            reponse: "The answer is of course"
            suffix: "4"
            other: ...

            we want full prompt to be:
                "\n\nsystem...Whats 2+2? Lets think ...\nassistant:\nThe answer is of course 4"
            
            and a response mask:
                padding - prompt - instruction following - reponse - suffix - padding
                0           0           0                   0           1       0
        """
        messages = self._build_messages(row_dict)        
        suffix = row_dict.pop('suffix')
        instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

        # build in-distribution chat
        prompt, _ = messages
        prompt['content'] += " " + instruction_following

        raw_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=False)
        raw_prompt = raw_prompt[:raw_prompt.rindex(self.tokenizer.eos_token)] # remove everything after (including) eos token            
        prompt_tokenized = self.tokenizer(raw_prompt, return_tensors='pt', add_special_tokens=False)
        suffix_tokens = self.tokenizer(suffix, return_tensors='pt', add_special_tokens=False)

        prompt_input_ids = prompt_tokenized.pop("input_ids")
        prompt_attention_mask = prompt_tokenized.pop("attention_mask")
        suffix_input_ids = suffix_tokens.pop('input_ids')
        suffix_attention_mask = suffix_tokens.pop('attention_mask')

        prompt_input_ids, prompt_attention_mask = verl_F.postprocess_data(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        suffix_input_ids, suffix_attention_mask = verl_F.postprocess_data(
            input_ids=suffix_input_ids,
            attention_mask=suffix_attention_mask,
            max_length=self.max_response_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=False,
            truncation=self.truncation,
        )

        full_input_ids = torch.cat([prompt_input_ids, suffix_input_ids], dim=-1)
        full_attention_mask = torch.cat([prompt_attention_mask, suffix_attention_mask], dim=-1)
        full_position_ids = compute_position_id_with_mask(full_attention_mask)

        tensors = {
            'input_ids' : full_input_ids.squeeze(),
            'attention_mask' : full_attention_mask.squeeze(),
            'position_ids' : full_position_ids.squeeze(),
            'prompts' : prompt_input_ids.squeeze(),
            'responses' : suffix_input_ids.squeeze(),
            'response_mask' : suffix_attention_mask.squeeze()
        }
        non_tensors = {
            **row_dict
        }
        return {**tensors, **non_tensors}
    
    def get_item_from_chat_source(self, row_dict):
        """
            chat file schema:

            prompt: [{"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."}]
            other: ...

            We want full prompt to be the full chat up to and including the assistant response:
                "\n\nsystem...user prompt...\nassistant:\nresponse..."

            Response mask:
                padding - prompt - response - padding
                0           0         1          0
        """
        messages = self._build_messages(row_dict)
        prompt, response = messages  # prompt is user message, response is assistant message

        # Tokenize separately for clean masks
        prompt_only = self.tokenizer.apply_chat_template([prompt], tokenize=False, add_special_tokens=False, add_generation_prompt=True)
        prompt_only_tokenized = self.tokenizer(prompt_only, return_tensors='pt', add_special_tokens=False)
        response_tokenized = self.tokenizer(response['content'], return_tensors='pt', add_special_tokens=False)

        prompt_input_ids = prompt_only_tokenized.pop("input_ids")
        prompt_attention_mask = prompt_only_tokenized.pop("attention_mask")
        response_input_ids = response_tokenized.pop("input_ids")
        response_attention_mask = response_tokenized.pop("attention_mask")

        # Postprocess (pad, truncate)
        prompt_input_ids, prompt_attention_mask = verl_F.postprocess_data(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        response_input_ids, response_attention_mask = verl_F.postprocess_data(
            input_ids=response_input_ids,
            attention_mask=response_attention_mask,
            max_length=self.max_response_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=False,
            truncation=self.truncation,
        )

        # Concatenate prompt + response
        full_input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=-1)
        full_attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)
        full_position_ids = compute_position_id_with_mask(full_attention_mask)

        tensors = {
            'input_ids': full_input_ids.squeeze(),
            'attention_mask': full_attention_mask.squeeze(),
            'position_ids': full_position_ids.squeeze(),
            'prompts': prompt_input_ids.squeeze(),
            'responses': response_input_ids.squeeze(),
            'response_mask': response_attention_mask.squeeze(),
        }
        non_tensors = {
            **row_dict
        }

        return {**tensors, **non_tensors}
    
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]

        if self.log_prob_from_rollouts_dir:
            return self.get_item_from_rollouts_dir_source(row_dict)
        elif self.log_prob_from_chat:
            if 'suffix' in row_dict:
                return self.get_item_from_chat_with_suffix_source(row_dict)
            else:
                return self.get_item_from_chat_source(row_dict)
        
        messages = self._build_messages(row_dict)
        model_inputs = {}
        
        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
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

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else: # processor is None
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
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
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
