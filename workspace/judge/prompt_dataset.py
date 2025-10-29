from typing import Callable
from datasets import load_dataset
from torch.utils.data import Dataset
from workspace.judge.taxonomy import Taxonomy, EIC_Taxonomy

from abc import ABC

class PromptDataset(Dataset, ABC):
    """
        Abstract Base CLass for PromptDatasets.
    """
    def __init__(self, tokenizer, prompt_fn: Callable, data_path: str, taxonomy : Taxonomy, keymap : dict):
        self.tokenizer = tokenizer
        self.prompt_fn = prompt_fn
        self.data = load_dataset("parquet", data_files=data_path)["train"]
        self.taxonomy = taxonomy

        assert all([key in keymap.keys() for key in ["question", "ground_truth_answer", "student_answer", "y_true"]]), "Please supply correct key mapper."
        self.keymap = keymap

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        q, gt, sa = item[self.keymap.get("question")], item[self.keymap.get("ground_truth_answer")], item[self.keymap.get("student_answer")]
        y_true = item[self.keymap.get('y_true')]
        chat = self.prompt_fn(question=q, ground_truth=gt, student_answer=sa, taxonomy=self.taxonomy)
        raw_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)

        return {**model_inputs, "y_true" : y_true}

class EIC_GSM8K(PromptDataset):
    def __init__(self, tokenizer, prompt_fn : Callable):
        """
            
        """
        taxonomy = EIC_Taxonomy(
            example=None # possibly put example here
        )
        keymap = {
            "question" : "prompt",
            "ground_truth_answer" : "correct_solution",
            "student_answer" : "incorrect_solution",
            "y_true" : "correct_answer"
        }
        super().__init__(tokenizer=tokenizer, prompt_fn=prompt_fn, taxonomy=taxonomy, data_path="/u/rfechner/data/eic_gsm8k_generated/test.parquet",keymap=keymap)

