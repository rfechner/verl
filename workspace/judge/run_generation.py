import os
import gc
import re
import json
import torch
from tqdm import tqdm
from itertools import product
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from typing import *

from workspace.judge.taxonomy import Taxonomy
from workspace.judge.prompt_dataset import ReasoningStrategy_aime25, ReasoningStrategy_gsm8k, ReasoningType_gsm8k

# ---------------------
# Prompt template
# ---------------------
constraints_for_generation_with_reasoning_types = \
    "Please do not change the outcome of the final response, as it's already correct. Important: Stay as close to the initial ground truth as possible. Do not short-cut the thinking process of the final response. Pre-pend a #### to your final answer, such that I can parse your output."

constraints_for_generation_with_error_types = \
    "Please only lightly augment the outcome of the final response. Important: Stay as close to the initial ground truth as possible. Do not short-cut the thinking process of the final response. Pre-pend a #### to your final answer, such that I can parse your output."

def regular(question : str, ground_truth : str, taxonomy : Taxonomy, behaviour : str, constraint : str, examples : List[Tuple[str, ...]] | None, **kwargs):
    system_prompt = {
        "role": "system",
        "content": f"You're a helpful case generator. I will give you some cases and you need to imitate my case generation process." + \
                    f"You have access to a behaviour taxonomy:\n{taxonomy}"
    }
    examples = '\n'.join([f"Example: {i + 1}:\n[QUESTION]\n{q}\n[GROUND_TRUTH]\n{gt}" \
                          f"\n[TASK] Take the ground truth response and augment it, such that the behaviour: {b} is present.{a}" for i, (q, gt, b, a) in enumerate(examples)]) if examples else ""
    user_prompt = {
        'role' : "user",
        'content' : 
            f"Be sure to follow format constraints: {constraint}" + \
            f"Here are some examples for you imitate:\n" + \
            examples + \
            f"\nCase to be augmented:\n" + \
            f"[QUESTION]\n{question}\n[GROUND_TRUTH]\n{ground_truth}\n" + \
            f"[TASK] Take the ground truth response and augment it, such that the behaviour: {behaviour} is present."
    }
    chat = [system_prompt, user_prompt]
    return chat

def simple_parse_fn(outputs: list[str]) -> list[str | None]:
    """
    Returns everything after the last '####' marker in each output string.
    If '####' is not found, returns None for that entry.
    """
    ret = []
    for response in outputs:
        # Find last occurrence of '####'
        if "####" in response:
            # Split on '####' and take the last chunk
            content = response.split("####")[-1].strip()
            ret.append(content)
        else:
            ret.append(None)
    return ret

OUT_DIR = "/ptmp/rfechner/out/generated_behaviours/"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------
# Main
# ---------------------
if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    results = {}
    num_gpus = torch.cuda.device_count()

    print(f"Found {num_gpus} GPUs.")

    
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = ReasoningType_gsm8k(tokenizer=tokenizer, prompt_fn=regular, constraint=constraints_for_generation_with_reasoning_types)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )

    batch_size = 1
    # Collate function
    def collate_fn(batch):
        # flatten all behaviours across samples
        input_ids = [b for item in batch for b in item["input_ids"]]
        attention_mask = [b for item in batch for b in item["attention_mask"]]

        padded_input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='left'
        )
        padded_attention_mask = pad_sequence(
            attention_mask, batch_first=True, padding_value=0, padding_side='left'
        )

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
        }

    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size)
    categories = dataset.taxonomy.categories.keys()


    # Prepare output file path
    model_name_safe = model_name.replace('/', '--')
    out_path = os.path.join(OUT_DIR, f"{dataset.__class__.__name__}__{model_name_safe}.jsonl")

    # Open file for incremental JSONL writing
    with open(out_path, "a", encoding="utf-8") as f:
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader), start=1):
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                
                # Generate model predictions
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=2048,
                    do_sample=True
                )
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                parsed = simple_parse_fn(decoded)

                # Build the line object
                line = {
                    "batch_index": i,
                    "data": parsed
                }

                # Write as one JSONL line
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                f.flush()

    meta = {
        "dataset": dataset.__class__.__name__,
        "generated_by": model_name,
        "taxonomy": dataset.taxonomy.categories
    }

    with open(os.path.join(OUT_DIR, f'{dataset.__class__.__name__}__{model_name_safe}.json')) as file:
        json.dump(obj=meta, fp=file)