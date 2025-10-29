import json
import re
from abc import ABC, abstractmethod
from typing import Tuple

class Taxonomy(ABC):
    def __init__(self, categories : dict, field_descr : str, task : str, example : str | None = None):
        self.categories = categories
        self.field_description = field_descr
        self.task = task
        self.example = example
        
    @property
    def has_example(self) -> bool:
        return self.example is not None

    def make_example(self) -> str | None:
        return self.example if self.has_example else None

    @property
    @abstractmethod
    def response_contraints(self) -> str:
        pass
    
    @abstractmethod
    def parse_response(self, response: str) -> list[str] | None:
        pass

    @abstractmethod
    def grade_outputs(self, gts : list[str] | list[list[str]], preds : list[str] | list[list[str]]) -> Tuple[int, int]:
        pass

class EIC_Taxonomy(Taxonomy):
    def __init__(self, example : str):
        categories = {
            "Calculation Error": {
                "description": "Error appears during the calculation process."
            },
            "Counting Error": {
                "description": "Error occurs during the counting process."
            },
            "Context Value Error": {
                "description": "Error arises when attributes of named entities do not align with the information provided."
            },
            "Hallucination": {
                "description": "Error involves adding fictitious unrelated statements contradictory to the question."
            },
            "Unit Conversion Error": {
                "description": "Error occurs during unit conversion process."
            },
            "Operator Error": {
                "description": "Error involves a single operator being erroneously applied within the expression."
            },
            "Formula Confusion Error": {
                "description": "Error appears when applying formula in inappropriate scenario."
            },
            "Missing Step": {
                "description": "Error entails an incomplete generation of reasoning process, lacking a necessary step."
            },
            "Contradictory Step": {
                "description": "Error manifests inconsistency between preceding and subsequent reasoning steps."
            }
        }
        field_description = "Mathematical Reasoning and Error Classification"
        task = "Analyze the type of mathematical error made by the student. Place the error into one of the previously mentioned categories of errors"
        super().__init__(categories=categories, field_descr=field_description, task=task, example=example)
    
    def parse_response(self, response: str) -> list[str] | None:
        # Find all \boxed{...} matches (case-insensitive)
        matches = re.findall(r'\\boxed\{([^}]*)\}', response, flags=re.IGNORECASE)
        if not matches:
            return None

        # Take the last boxed content
        content = matches[-1].strip()

        # Remove possible \text{...} wrapping
        prediction = re.sub(r'\\text\{([^}]*)\}', r'\1', content)
        return prediction

    def grade_outputs(self, gts, preds):
        matches = [
            gt == pred for gt, pred in zip(gts, preds)
        ]
        parsed = [
            1 if pred else 0 for pred in preds
        ]
        return sum(matches), sum(parsed)

    @property
    def response_contraints(self):
        return "Flag only the first error in the students response. If the student made a Calculation Error, the final answer is: \\boxed{Calculation Error}"

      