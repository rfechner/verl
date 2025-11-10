import os
import pandas as pd

def create_eic_gsm8k(df : pd.DataFrame):
    rows = []
    for i, row in df.iterrows():
        correct, incorrect = \
            {
                'prompt' : [{'role' : 'user', 'content' : row['prompt']}, 
                            {'role' : 'assistant', 'content' : row['correct_solution']}],
                'behaviour_type' : 'anchor',
                'source_file' : row['source_file'],
                'index' : i
            }, \
            {
                'prompt' : [{'role' : 'user', 'content' : row['prompt']}, 
                            {'role' : 'assistant', 'content' : row['incorrect_solution']}],
                'behaviour_type' : row['error_type'],
                'source_file' : row['source_file'],
                'index' : i
            }
        rows.extend([correct, incorrect])
    df = pd.DataFrame(rows)

    print(df.describe())
    print(f'Writing to ', '/u/rfechner/data/eic_gsm8k_generated/delta.jsonl')
    with open('/u/rfechner/data/eic_gsm8k_generated/delta.jsonl', 'w') as file:
        df.to_json(file, lines=True, orient='records')

def create_reasoning_strategy_gsm8k(df : pd.DataFrame):
    with open('/ptmp/rfechner/out/generated_behaviours/ReasoningStrategy_gsm8k__Qwen--Qwen3-8B.jsonl', 'r') as file:
        rt = pd.read_json(file, lines=True)

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
    df.reset_index(drop=True, inplace=True)
    rt.reset_index(drop=True, inplace=True)
    df = df.head(len(rt))
    df['augmented'] = rt['data']
    df['valid'] = df['augmented'].apply(lambda x: len(x) == len(categories))
    df = df[df['valid']]
    rows = []
    behaviours = list(categories.keys())
    for i, row in df.iterrows():
        augmented, original = \
            [{
                'prompt' : [{'role' : 'user', 'content' : row['prompt']}, 
                            {'role' : 'assistant', 'content' : row['augmented'][k]}],
                'behaviour_type' : behaviours[k],
                'index' : i
            } for k in range(len(behaviours))], \
            {
                'prompt' : [{'role' : 'user', 'content' : row['prompt']}, 
                            {'role' : 'assistant', 'content' : row['correct_solution']}],
                'behaviour_type' : 'anchor',
                'index' : i
            }
        
        rows.extend([*augmented, original])

    df = pd.DataFrame(rows)
    path = "/u/rfechner/data/incomplete_reasoning_strategies"
    os.makedirs(path, exist_ok=True)

    print(df.describe())
    print('Writing to: ', os.path.join(path, 'deltas.jsonl'))
    with open(os.path.join(path, 'deltas.jsonl'), 'w') as file:
        df.to_json(path_or_buf=file, lines=True, orient='records')

def create_reasoning_types_gsm8k(df : pd.DataFrame):
    with open('/ptmp/rfechner/out/generated_behaviours/ReasoningType_gsm8k__Qwen--Qwen3-8B.jsonl', 'r') as file:
        rt = pd.read_json(file, lines=True)

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
    df.reset_index(drop=True, inplace=True)
    rt.reset_index(drop=True, inplace=True)
    df = df.head(len(rt))
    df['augmented'] = rt['data']
    df['valid'] = df['augmented'].apply(lambda x: len(x) == len(categories))
    df = df[df['valid']]
    rows = []
    behaviours = list(categories.keys())
    for i, row in df.iterrows():
        augmented, original = \
            [{
                'prompt' : [{'role' : 'user', 'content' : row['prompt']}, 
                            {'role' : 'assistant', 'content' : row['augmented'][k]}],
                'behaviour_type' : behaviours[k],
                'index' : i
            } for k in range(len(behaviours))], \
            {
                'prompt' : [{'role' : 'user', 'content' : row['prompt']}, 
                            {'role' : 'assistant', 'content' : row['correct_solution']}],
                'behaviour_type' : 'anchor',
                'index' : i
            }
        
        rows.extend([*augmented, original])

    df = pd.DataFrame(rows)
    path = "/u/rfechner/data/incomplete_reasoning_types"
    os.makedirs(path, exist_ok=True)

    print(df.describe())
    print('Writing to: ', os.path.join(path, 'deltas.jsonl'))
    with open(os.path.join(path, 'deltas.jsonl'), 'w') as file:
        df.to_json(path_or_buf=file, lines=True, orient='records')

if __name__ == '__main__':

    with open('/u/rfechner/data/eic_gsm8k_deduplicated/test.parquet', 'rb') as file:
        df = pd.read_parquet(file)

    create_eic_gsm8k(df.copy())
    create_reasoning_strategy_gsm8k(df.copy())
    create_reasoning_types_gsm8k(df.copy())

