import os
import time
import gc
import json
import requests
from itertools import product
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from workspace.judge.taxonomy import Taxonomy
from workspace.judge.prompt_dataset import PromptDataset, EIC_GSM8K
from workspace.judge.evaluator import parse_outputs  # assuming this exists


# ====================================================
# OpenRouter Client
# ====================================================
class OpenRouterClient:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            # "HTTP-Referer": "https://your-app-domain.com",  # optional but good
            "X-Title": "JudgeEval",
            "Content-Type": "application/json",
        }

    def generate(self, messages, max_new_tokens=512):
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens,
        }
        try:
            response = requests.post(self.url, headers=self.headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[ERROR] API call failed for {self.model}: {e}")
            return None


# ====================================================
# Prompt template
# ====================================================
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


OUT_DIR = "/ptmp/rfechner/out/judge_api/"
os.makedirs(OUT_DIR, exist_ok=True)

ALL_PROMPTS = {"regular": regular}
MODEL_NAMES = [
    "minimax/minimax-m2:free"
]


# ====================================================
# Evaluation loop using OpenRouter
# ====================================================
def evaluate_api(client, dataset, taxonomy, max_new_tokens=512):
    y_trues, outputs = [], []
    requests_sent = 0
    for i, (messages, y_true) in enumerate(tqdm(dataset), start=1):
        output_text = client.generate(messages, max_new_tokens=max_new_tokens)
        outputs.append(output_text)
        y_trues.append(y_true)
        requests_sent += 1
        if requests_sent >= 20: # rate limit imposed by open router. note: max 50 a day...
            time.sleep(60)
            requests_sent = 0
        
        
    y_preds = parse_outputs(outputs, taxonomy)
    
    tps, p = taxonomy.grade_outputs(y_preds, y_trues)
    acc, parsed = tps / len(y_trues), p / len(y_trues)
    return {"accuracy": acc, "parse_rate": parsed}


# ====================================================
# Main
# ====================================================
if __name__ == "__main__":
    api_key = open("/u/rfechner/.cache/openrouter/key.txt").read().strip()
    results = {}

    for model_name, prompt_key in product(MODEL_NAMES, ALL_PROMPTS.keys()):
        print(f"\n=== Evaluating {model_name} with prompt '{prompt_key}' ===")

        dataset = EIC_GSM8K(tokenizer=None, prompt_fn=ALL_PROMPTS[prompt_key])
        client = OpenRouterClient(api_key, model=model_name)
        metrics = evaluate_api(client, dataset, dataset.taxonomy, max_new_tokens=2048)

        print(f"Final results for {model_name}: {metrics}")

        out_name = model_name.replace("/", "--")
        with open(os.path.join(OUT_DIR, f"{out_name}_{prompt_key}.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        gc.collect()
