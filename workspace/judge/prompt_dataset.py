from typing import Callable
from datasets import load_dataset
from torch.utils.data import Dataset
from workspace.judge.taxonomy import Taxonomy, EIC_Taxonomy, ReasoningStrategies, ReasoningTypes

from typing import *
from abc import ABC
import torch

class PromptDatasetForClassification(Dataset, ABC):
    """
        Abstract Base CLass for PromptDatasets.
    """
    def __init__(self, tokenizer, prompt_fn: Callable, data_path: str, taxonomy : Taxonomy, keymap : dict, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.prompt_fn = prompt_fn
        self.data = load_dataset("parquet", data_files=data_path)["train"]
        self.taxonomy = taxonomy
        self.keymap = keymap

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any] | Tuple[list[str], str]:
        item = self.data[idx]
        q, gt, sa = item[self.keymap.get("question")], item[self.keymap.get("ground_truth_answer")], item[self.keymap.get("student_answer")]
        y_true = item[self.keymap.get('y_true')]
        chat = self.prompt_fn(question=q, ground_truth=gt, student_answer=sa, taxonomy=self.taxonomy)
        if self.tokenizer:
            raw_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            return {**model_inputs, "y_true" : y_true}
        return chat, y_true

class PromptDatasetForGeneration(Dataset, ABC):
    def __init__(self, tokenizer, prompt_fn: Callable, data_path: str, taxonomy : Taxonomy, keymap : dict, example : str, constraint : str, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.prompt_fn = prompt_fn
        self.data = load_dataset("parquet", data_files=data_path)["train"]
        self.taxonomy = taxonomy
        self.keymap = keymap
        self.example = example
        self.constraint=constraint

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any] | list[str]:
        item = self.data[idx]
        q, gt = item[self.keymap.get("question")], item[self.keymap.get("ground_truth_answer")]

        model_input_buffer = []
        # for each category from the taxonomy, inject a behaviour.
        for behaviour in self.taxonomy.categories.keys():
            chat = self.prompt_fn(question=q, ground_truth=gt, taxonomy=self.taxonomy, behaviour=behaviour, constraint=self.constraint, examples=self.example)

            if self.tokenizer:
                raw_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
                model_input_buffer.append(model_inputs)
            else:
                model_input_buffer.append(chat)

        if self.tokenizer:
            return {
                key : [entry[key].squeeze(0) for entry in model_input_buffer] for key in model_input_buffer[0]
            }
        else:
            return model_input_buffer
    
class EIC_GSM8K(PromptDatasetForClassification):
    def __init__(self, tokenizer, prompt_fn : Callable, **kwargs):
        """
            
        """
        taxonomy = EIC_Taxonomy(
            example=None # possibly put example here
        )
        keymap = {
            "question" : "prompt",
            "ground_truth_answer" : "correct_solution",
            "student_answer" : "incorrect_solution",
            "y_true" : "error_type"
        }
        super().__init__(tokenizer=tokenizer, prompt_fn=prompt_fn, taxonomy=taxonomy, data_path="/u/rfechner/data/eic_gsm8k_generated/test.parquet",keymap=keymap, **kwargs)

class ReasoningStrategy_aime25(PromptDatasetForGeneration):
    def __init__(self, tokenizer, prompt_fn : Callable, example : str | None = None, constraint : str | None = None, generation_prompt=True, **kwargs):
        """
            GSM8K dataset augmented with Reasoning Types.
        """
        taxonomy = ReasoningStrategies(
            example=None # possibly put example here
        )
        keymap = {
            "question" : "question",
            "ground_truth_answer" : "answer"
        }
        default_question = "The sequence is 3, 6, 12, 24, 48, __, 192. What number should fill the blank?"
        default_ground_truth_answer = "Each term doubles the previous one: after 48 comes 96, then 192. The missing number is \\boxed{96}"
        prefix_agent_response = "I'm asked to augment the given ground truth response. First lets think about the behaviour i need to inject : {}."
        
        # q, gt, b, a in examples
        generation_examples = [
            (
                default_question, default_ground_truth_answer, "Pattern Recognition",
                prefix_agent_response.format('Pattern Recognition') + \
                "To augment the answer with the behaviour Pattern Recognition i should include behaviours which identify patterns among the sequence of integers, which then leads to the solution." \
                "#### I notice that each term doubles the one before it: 3 → 6 (×2), 6 → 12 (×2), etc. Following this pattern, 48 × 2 = 96, and 96 × 2 = 192. The final answer is \\boxed{96}."        
            ),
            (
                default_question, default_ground_truth_answer, "Backtracking",
                prefix_agent_response.format('Backtracking') + \
                "To augment the answer with the behaviour Backtracking i should include behaviours which backtrack on an incorrect part of a generation, then continue with the correct response." \
                "####  Let me test possible rules: if I add 3 to the first number, I'm arriving at 6 which is the next number. So maybe the pattern is that I have to add 3 every step?" \
                    "Hold on, then 6 + 3 isn't 12. My initial guess is wrong, let's backtrack. If I multiply by 2, it matches every term (3×2=6, 6×2=12, etc.)." \
                    "So 48x2 = 96. The final result is \\boxed{96}."
            ),
            (
                default_question, default_ground_truth_answer, "Verification",
                prefix_agent_response.format('Verification') + \
                "To augment the answer with the behaviour Verification i should include behaviours which try to verify steps of the calculation or the final response." \
                "#### I could try to double each number. 3x2=6, so this checks out. Next, 6x2=12. I think I'm onto something - 12x2=24, 24x2=48, 48x2=96 and 96x2=192 so my hypothesis is correct. The final answer is \\boxed{96}."
            )
        ]

        if not example and generation_prompt:
            example = generation_examples

        super().__init__(tokenizer=tokenizer, prompt_fn=prompt_fn, taxonomy=taxonomy, data_path="/u/rfechner/data/aime25_with_llm_answers/test.parquet", keymap=keymap, example= example, constraint= constraint, **kwargs)
    
class ReasoningStrategy_gsm8k(PromptDatasetForGeneration):
    def __init__(self, tokenizer, prompt_fn : Callable, example : str | None = None, constraint : str | None = None, generation_prompt=True, **kwargs):
        """
            GSM8K dataset augmented with Reasoning Types.
        """
        taxonomy = ReasoningStrategies(
            example=None # possibly put example here
        )
        keymap = {
            "question" : "prompt",
            "ground_truth_answer" : "correct_solution"
        }

        default_question = "The sequence is 3, 6, 12, 24, 48, __, 192. What number should fill the blank?"
        default_ground_truth_answer = "Each term doubles the previous one: after 48 comes 96, then 192. The missing number is \\boxed{96}"
        prefix_agent_response = "I'm asked to augment the given ground truth response. First lets think about the behaviour i need to inject : {}."
        
        # q, gt, b, a in examples
        generation_examples = [
            (
                default_question, default_ground_truth_answer, "Pattern Recognition",
                prefix_agent_response.format('Pattern Recognition') + \
                "To augment the answer with the behaviour Pattern Recognition i should include behaviours which identify patterns among the sequence of integers, which then leads to the solution." \
                "#### I notice that each term doubles the one before it: 3 → 6 (×2), 6 → 12 (×2), etc. Following this pattern, 48 × 2 = 96, and 96 × 2 = 192. The final answer is \\boxed{96}."        
             ),
            (
                default_question, default_ground_truth_answer, "Backtracking",
                prefix_agent_response.format('Backtracking') + \
                "To augment the answer with the behaviour Backtracking i should include behaviours which backtrack on an incorrect part of a generation, then continue with the correct response." \
                "####  Let me test possible rules: if I add 3 to the first number, I'm arriving at 6 which is the next number. So maybe the pattern is that I have to add 3 every step?" \
                    "Hold on, then 6 + 3 isn't 12. My initial guess is wrong, let's backtrack. If I multiply by 2, it matches every term (3×2=6, 6×2=12, etc.)." \
                    "So 48x2 = 96. The final result is \\boxed{96}."
            ),
            (
                default_question, default_ground_truth_answer, "Verification",
                prefix_agent_response.format('Verification') + \
                "To augment the answer with the behaviour Verification i should include behaviours which try to verify steps of the calculation or the final response." \
                "#### I could try to double each number. 3x2=6, so this checks out. Next, 6x2=12. I think I'm onto something - 12x2=24, 24x2=48, 48x2=96 and 96x2=192 so my hypothesis is correct. The final answer is \\boxed{96}."
            )
        ]

        if not example and generation_prompt:
            example = generation_examples

        super().__init__(tokenizer=tokenizer, prompt_fn=prompt_fn, taxonomy=taxonomy, data_path="/u/rfechner/data/eic_gsm8k_deduplicated/test.parquet", keymap=keymap, example= example, constraint= constraint, **kwargs)

class ReasoningType_gsm8k(PromptDatasetForGeneration):
    def __init__(self, tokenizer, prompt_fn : Callable, example : str | None = None, constraint : str | None = None, generation_prompt=True, **kwargs):
        """
            GSM8K dataset augmented with Reasoning Types.
        """
        taxonomy = ReasoningTypes(
            example=None # possibly put example here
        )
        keymap = {
            "question" : "prompt",
            "ground_truth_answer" : "correct_solution"
        }

        default_question = "The sequence is 3, 6, 12, 24, 48, __, 192. What number should fill the blank?"
        default_ground_truth_answer = "Each term doubles the previous one: after 48 comes 96, then 192. The missing number is \\boxed{96}"
        prefix_agent_response = "I'm asked to augment the given ground truth response. First lets think about the behaviour i need to inject : {}."
        
        # q, gt, b, a in examples
        generation_examples = [
            (
                default_question,
                default_ground_truth_answer,
                "Deductive Reasoning",
                prefix_agent_response.format("Deductive Reasoning") +
                "To augment the answer with the behaviour Deductive Reasoning I should include behaviours that start from known premises and logically derive the conclusion." \
                "#### I know that if each term is produced by applying a consistent rule, then that same rule must apply between every pair of consecutive numbers. The differences are not constant, so the rule cannot be additive. But the ratios are all 2 (6/3=2, 12/6=2, 24/12=2, 48/24=2), so by deduction the rule must be multiplication by 2. Therefore the missing number must be 48×2 = 96, and 96×2 = 192. The final answer is \\boxed{96}."
            ),
            (
                default_question,
                default_ground_truth_answer,
                "Inductive Reasoning",
                prefix_agent_response.format("Inductive Reasoning") +
                "To augment the answer with the behaviour Inductive Reasoning I should include behaviours that infer the general rule from specific observations." \
                "#### Observing the sequence term by term, I test the pattern: 3 to 6 looks like a doubling. 6 to 12 also matches doubling. Repeating the check for each pair builds evidence for the rule. Since every observed step fits the ×2 pattern, I generalize that the whole sequence is doubling. Applying it, 48×2 = 96 and 96×2 = 192. So the missing number is \\boxed{96}."
            ),
            (
                default_question,
                default_ground_truth_answer,
                "Abductive Reasoning",
                prefix_agent_response.format("Abductive Reasoning") +
                "To augment the answer with the behaviour Abductive Reasoning I should include behaviours that form the most plausible explanation for the observed pattern and check its fit." \
                "#### The rapid growth suggests a multiplicative rule. The simplest explanation is that each term doubles the previous one. Testing this hypothesis: 3→6 (fits), 6→12 (fits), 12→24 (fits), 24→48 (fits). Since this is the simplest rule consistent with the data, I infer that the next term should be 48×2 = 96, which then doubles to 192. The missing number is \\boxed{96}."
            ),
            (
                default_question,
                default_ground_truth_answer,
                "Analogical Reasoning",
                prefix_agent_response.format("Analogical Reasoning") +
                "To augment the answer with the behaviour Analogical Reasoning I should include behaviours that solve the problem by relating it to a structurally similar situation." \
                "#### This sequence grows like other geometric sequences where each term is a constant multiple of the previous one—for example, the sequence 2, 4, 8, 16,… follows a ×2 rule. By analogy, the given sequence 3, 6, 12, 24, 48 also follows a doubling pattern. So the next number should mirror that same structure: 48×2 = 96, and continuing gives 192. The missing number is \\boxed{96}."
            )
        ]


        if not example and generation_prompt:
            example = generation_examples

        super().__init__(tokenizer=tokenizer, prompt_fn=prompt_fn, taxonomy=taxonomy, data_path="/u/rfechner/data/eic_gsm8k_deduplicated/test.parquet", keymap=keymap, example= example, constraint= constraint, **kwargs)


class EIC_ErrorTypes_gsm8k(PromptDatasetForGeneration):
    def __init__(self, tokenizer, prompt_fn: Callable, example: str | None = None, constraint: str | None = None, generation_prompt=True, **kwargs):
        """
            GSM8K dataset augmented with EIC (Error Interpretation and Classification) Types.
        """
        taxonomy = EIC_Taxonomy(
            example=None  # can embed a sample later if needed
        )

        keymap = {
            "question": "prompt",
            "ground_truth_answer": "correct_solution"
        }

        default_question = "A bakery sold 48 muffins on Monday and twice as many on Tuesday. How many muffins did they sell in total?"
        default_ground_truth_answer = (
            "On Tuesday, they sold twice as many muffins as Monday: 2 × 48 = 96. "
            "So in total: 48 + 96 = 144. The bakery sold \\boxed{144} muffins."
        )

        prefix_agent_response = (
            "I'm asked to augment the given ground truth response. "
            "First, let's think about the error type I need to inject: {}."
        )

        # Example augmentations for each error type
        generation_examples = [
            (
                default_question,
                default_ground_truth_answer,
                "Calculation Error",
                prefix_agent_response.format("Calculation Error")
                + " To simulate a Calculation Error, I'll make a small arithmetic mistake during computation."
                "#### On Tuesday, they sold twice as many muffins as Monday: 2 × 48 = 94. "
                "Then, total muffins = 48 + 94 = 142. The bakery sold \\boxed{142} muffins."
            ),
            (
                default_question,
                default_ground_truth_answer,
                "Context Value Error",
                prefix_agent_response.format("Context Value Error")
                + " To simulate a Context Value Error, I'll misuse a contextual quantity from the problem."
                "#### I'm counting two more than 48 + 2 = 50 muffins on Tuesday. "
                "Then total = 48 + 50 = 98. So I conclude \\boxed{98} muffins."
            ),
            (
                default_question,
                default_ground_truth_answer,
                "Hallucination",
                prefix_agent_response.format("Hallucination")
                + " To simulate a Hallucination error, I'll add an unrelated or fictitious detail."
                "#### The bakery also sold 20 cupcakes. "
                "So total muffins + cupcakes = 48 + 96 + 20 = 164. The answer is \\boxed{164}."
            ),
            (
                default_question,
                default_ground_truth_answer,
                "Operator Error",
                prefix_agent_response.format("Operator Error")
                + " To simulate an Operator Error, I'll apply the wrong arithmetic operator."
                "#### On Tuesday: 48 ÷ 2 = 24 muffins. Total = 48 + 24 = 72. "
                "Thus, I conclude \\boxed{72} muffins."
            ),
            (
                default_question,
                default_ground_truth_answer,
                "Missing Step",
                prefix_agent_response.format("Missing Step")
                + " To simulate a Missing Step, I'll skip part of the reasoning."
                "#### On Tuesday, twice as many → 96. Therefore, total is \\boxed{96}."
            )
        ]

        if not example and generation_prompt:
            example = generation_examples

        super().__init__(
            tokenizer=tokenizer,
            prompt_fn=prompt_fn,
            taxonomy=taxonomy,
            data_path="/u/rfechner/data/eic_gsm8k_deduplicated/test.parquet",
            keymap=keymap,
            example=example,
            constraint=constraint,
            **kwargs
        )
