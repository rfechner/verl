# run_eval.py
import json
import torch
from itertools import product
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from workspace.judge.taxonomy import Taxonomy
from workspace.judge.prompt_dataset import PromptDataset, EIC_GSM8K
from workspace.judge.evaluator import evaluate

# Define prompt functions
regular = lambda question, ground_truth, student_answer, taxonomy: [
    {"role": "system",
     "content": f"You're an expert in {taxonomy.field_description}. "
                f"Categories: {taxonomy.categories}. "
                f"Your task: {taxonomy.task}. "
                f"Your answer format constraints: {taxonomy.response_contraints}. "
                + (f"Example: {taxonomy.make_example()}" if taxonomy.has_example else "")},
    {"role": "user",
     "content": f"[QUESTION]\n{question}\n[GROUND_TRUTH]\n{ground_truth}\n[STUDENT_ANSWER]\n{student_answer}\n" \
     "First, think step by step and format your final answer as defined previously."}
]
model_name, prompt_key = "Qwen/Qwen3-4B", "regular"
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")

    dataset = EIC_GSM8K(tokenizer, regular)
    dataloader = DataLoader(dataset, batch_size=16)