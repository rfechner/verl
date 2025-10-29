import os
import gc
import json
import torch
from itertools import product
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from workspace.judge.taxonomy import Taxonomy
from workspace.judge.prompt_dataset import PromptDataset, EIC_GSM8K
from workspace.judge.evaluator import evaluate

# ---------------------
# Prompt template
# ---------------------
regular = lambda question, ground_truth, student_answer, taxonomy: [
    {
        "role": "system",
        "content": (
            f"You're an expert in {taxonomy.field_description}. "
            f"You have access to the following valid error categories: {taxonomy.categories}. "
            f"Your task: {taxonomy.task}. "
            f"Your answer format constraints: {taxonomy.response_contraints}. "
            + (f"Example: {taxonomy.make_example()}" if taxonomy.has_example else "")
        ),
    },
    {
        "role": "user",
        "content": (
            f"[QUESTION]\n{question}\n[GROUND_TRUTH]\n{ground_truth}\n"
            f"[STUDENT_ANSWER]\n{student_answer}\n"
            "First, think step by step and format your final answer as defined previously."
        ),
    },
]

ALL_PROMPTS = {"regular": regular}
MODEL_NAMES = [
    "openai/gpt-oss-20b",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-4B",
    "meta-llama/Llama-3.2-3B-Instruct",
]


# ---------------------
# Main
# ---------------------
if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    results = {}
    num_gpus = torch.cuda.device_count()

    print(f"Found {num_gpus} GPUs.")
    
    for model_name, prompt_key in product(MODEL_NAMES, ALL_PROMPTS.keys()):
        print(f"Evaluating {model_name} with prompt '{prompt_key}'")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto"
        )

        # Prepare dataset
        dataset = EIC_GSM8K(tokenizer, ALL_PROMPTS[prompt_key])

        batch_size = 8
        # Collate function
        def collate_fn(batch):
            input_ids = [item["input_ids"].squeeze(0) for item in batch]
            attention_mask = [item["attention_mask"].squeeze(0) for item in batch]
            y_true = [item["y_true"] for item in batch]

            padded_input_ids = pad_sequence(
                input_ids, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='left'
            )
            padded_attention_mask = pad_sequence(
                attention_mask, batch_first=True, padding_value=0, padding_side='left'
            )

            return {
                "input_ids": padded_input_ids,
                "attention_mask": padded_attention_mask,
                "y_true": torch.tensor(y_true),
            }

        dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size)

        # ---------------------
        # Memory-safe evaluation
        # ---------------------
        with torch.no_grad():
            metrics = evaluate(
                model,
                tokenizer,
                dataloader,
                taxonomy=dataset.taxonomy,
                max_new_tokens=2048
            )
            
        print(torch.cuda.memory_summary())
        results[f"{model_name}_{prompt_key}"] = metrics

        # Aggressive cleanup
        del model, tokenizer, dataloader, dataset
        torch.cuda.empty_cache()
        gc.collect()

    # Save results
    with open("/u/rfechner/out.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Fin.")
