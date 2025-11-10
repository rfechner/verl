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
    def response_contraints(self) -> str:
        return ""
    
    def parse_response(self, response: str) -> list[str] | None:
        # Find all \boxed{..} matches (case-insensitive)
        matches = re.findall(r'\\boxed\{([^}]*)\}', response, flags=re.IGNORECASE)
        if not matches:
            return None

        # Take the last boxed content
        content = matches[-1].strip()

        # Remove possible \text{..} wrapping
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

    def __repr__(self):
        return "\n\n".join([f"**{key}**:\n{value['description']}" for key, value in self.categories.items()])
    

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
        task = "Analyze the type of mathematical error made by the student."
        super().__init__(categories=categories, field_descr=field_description, task=task, example=example)
    
    @property
    def response_contraints(self):
        return "Flag only the first error in the students response. If the student made a Calculation Error, the final answer is: \\boxed{Calculation Error}"

class ReasoningTypes(Taxonomy):
    """
        This taxonomy includes Reasoning Types from neuroscience, psychology and educational sciences.
        It is neither a complete taxonomy, nor a taxonomy of solution strategies.  
    """
    def __init__(self, example : str):
        categories = {
            "Deductive Reasoning" : {
                'description' : "Deductive reasoning is a fundamental aspect of mathematics. It involves starting with a set of premises or known facts and using logical principles to reach a conclusion. For example, if we know that all squares have four sides and that a rectangle is a type of square, we can deduce that all rectangles also have four sides.",
                'source' : "https://math.libretexts.org/Courses/Coalinga_College/Math_for_Educators_(MATH_010A_and_010B_CID120)/06%3A_Mathematical_Reasoning/6.02%3A_Types_of_Reasoning",
            },
            "Inductive Reasoning" : {
                'description' : "Inductive reasoning is the process of making generalizations based on specific observations or patterns. It is used to predict outcomes or infer general rules from limited data. For example, if you observe that the sun has risen every morning for as long as you can remember, you might use inductive reasoning to conclude that the sun will rise tomorrow morning as well.",
                'source' : "https://math.libretexts.org/Courses/Coalinga_College/Math_for_Educators_(MATH_010A_and_010B_CID120)/06%3A_Mathematical_Reasoning/6.02%3A_Types_of_Reasoning",
            },
            "Abductive Reasoning" : {
                'description' : "Abductive reasoning is a form of logical inference that seeks to find the simplest and most likely explanation for a set of observations. It involves forming hypotheses to explain observed phenomena and then testing those hypotheses to see if they hold true. For example, if you come home and find your front door ajar, you might use abductive reasoning to hypothesize that someone has broken into your house.",
                'source' : "https://math.libretexts.org/Courses/Coalinga_College/Math_for_Educators_(MATH_010A_and_010B_CID120)/06%3A_Mathematical_Reasoning/6.02%3A_Types_of_Reasoning"
            },
            "Analogical Reasoning" : {
                'description' : "Analogical reasoning involves using analogies, or comparisons between two things, to draw conclusions or solve problems. It is based on the idea that if two things are similar in some respects, they are likely to be similar in other respects as well. Example: Understanding multiplication as repeated addition.",
                'source' : "https://math.libretexts.org/Courses/Coalinga_College/Math_for_Educators_(MATH_010A_and_010B_CID120)/06%3A_Mathematical_Reasoning/6.02%3A_Types_of_Reasoning"            
            }
        }
        field_description = "Mathematical Reasoning Type Classification"
        task = "Map the types of reasoning the student makes onto the given categories."
        super().__init__(categories=categories, field_descr=field_description, task=task, example=example)

class ReasoningStrategies(Taxonomy):
    """
        This taxonomy includes an informal selection of problem solving strategies. These strategies are
        applicable in any context and aren't applied conditionally on the problem type. For example if the
        task is simple, there isn't a need for Counterexample Testing, as in a harder proof-type task.
    """
    def __init__(self, example : str):
        categories = {
            "Pattern Recognition": {
                "description": "Identifying regularities, repetitions, or structures within numbers, figures, or operations. Example: A student lists the first few triangular numbers (1, 3, 6, 10, \u2026) and notices that each is obtained by adding consecutive integers, leading to the conjecture T_n = n(n+1)/2."
            },
            "Backtracking": {
                "description": "Trying partial solutions and undoing steps when they lead to contradictions or dead ends; systematic trial-and-error with revision. Example: While solving a Sudoku puzzle, a student places numbers tentatively, backtracks when a rule is violated, and explores alternate paths. In algebra, testing possible integer roots of a polynomial and rejecting invalid ones."
            },
            "Simplification": {
                "description": "Reducing a complex problem to a simpler or specific case to reveal an underlying structure. Example: To prove a formula for the sum of the first n squares, the student first tries small n (e.g., n=1,2,3) to observe and test a conjectured pattern."
            },
            "Decomposition": {
                "description": "Breaking a problem into smaller, manageable subproblems or intermediate goals. Example: To solve a multi-step geometry proof, the student identifies subgoals like proving triangles ABC and DEF are similar before tackling the final ratio result."
            },
            "Verification": {
                "description": "Checking the correctness and coherence of a solution or conjecture. Example: After solving a system of equations, the student substitutes results into the original equations to confirm validity."
            },
            "Trials": {
                "description": "Testing multiple approaches or examples to gather insight or eliminate possibilities. Example: When unsure how to find integer solutions to x^2 + y^2 = 25, a student tries values of x (e.g., 0, 3, 4) until consistent pairs (3,4), (4,3), (0,5), etc. emerge."
            },
            "Analogical Mapping": {
                "description": "Using similarities between a known problem and a new one to transfer a solution method. Example: A student recalls solving 'sum of first n odd numbers' to find a pattern for 'sum of first n even numbers.'"
            }
        }
        field_description = "Mathematical Reasoning Strategy Classification"
        task = "Map the types of reasoning streategies the student uses onto the given categories."
        super().__init__(categories=categories, field_descr=field_description, task=task, example=example)
