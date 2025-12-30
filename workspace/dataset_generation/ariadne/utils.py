import numpy as np
import pandas as pd
import os

from typing import Literal
import warnings
from functools import wraps

path = "/ptmp/rfechner/out/exp05_rollouts_qwen2.5-7b/qwen2.5_7b__gspo/val_jsonl"

def load_rollouts() -> list[pd.DataFrame]:
    checkpoints = []
    files = list(
        sorted(os.listdir(path), key=lambda x: int(x.split('_')[0]))
    )
    for file in files:
        with open(os.path.join(path, file), 'r') as jsonfile:
            cp = int(file.split('_')[0])
            df = pd.read_json(jsonfile, lines=True)
            df['checkpoint'] = [cp] * len(df)
            checkpoints.append(df)
    return checkpoints

def ignore_warnings(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper

# preprocessing functions
def prefilter(dataframe, filter_std_nonzero : bool = False, filter_overlong=False, max_resp_len=1024):
    # 1) take the first 500 * 256 answers from the dataframe as base set, as these correspond to the math500 questions.
    df = dataframe.iloc[:500 * 256]
    cond = df['output'].apply(lambda x: 'python' not in x and '\\boxed{' in x[-100:])
    df = df[cond]

    if filter_overlong:
        approx_tokens = (df['input'].apply(len) + df['output'].apply(len)) / 3
        df = df[approx_tokens < max_resp_len]
    if filter_std_nonzero:
        cond = df.groupby('input', sort=False)['score'].transform(lambda x: x.std() != 0)
        df = df[cond]

    return df

@ignore_warnings
def sample(df : pd.DataFrame, sample_qs : int = 100, resp_per_q : int = 3,
           only_filter : Literal['negative', 'positive', 'any'] = 'positive', 
           filter_std_nonzero=False,
           filter_overlong=False) -> pd.DataFrame:
    """
        Given a dataframe with N questions * K answers, we're going
        to filter for `resp_per_q` ground truth answers for `sample_qs` questions.
        Naturally, this excludes questions, for which we've sampled less than `resp_per_q` correct answers.
    """
    df = prefilter(df, filter_std_nonzero=filter_std_nonzero, filter_overlong=filter_overlong)

    # compute question groups for which we have at least `resp_per_q` correct answers
    eligible = (
        df.groupby('input', sort=False)['score']
        .transform('sum')
        .ge(resp_per_q)
    )
    
    if only_filter=='negative':
        # from those questions, filter responses based on eligibility and comparison function.
        df = df[eligible & (df['score'] <= 0)]
    elif only_filter=='positive':
        # from those questions, filter responses based on eligibility and comparison function.
        df = df[eligible & (df['score'] > 0)]
    else:
        df = df[eligible]
    
    # sample `sample_qs` questions out of those questions

    questions = (
        df['input']
        .drop_duplicates()
        .sample(n=sample_qs, random_state=0)
    ) if sample_qs > 0 else df['input'].unique() # take all questions if sample_qs == -1

    # finally, from the sampled questions, sample `resp_per_q` correct answers
    result = (
        df[df['input'].isin(questions)]
        .groupby('input', group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), resp_per_q), random_state=0))
    )
    return result

def filter_overlong_sequences(dfs: list[pd.DataFrame], max_resp_len : int = 1024) -> list[pd.DataFrame]:
    ret = []
    for df in dfs:
        approx_tokens = (df['input'].apply(len) + df['output'].apply(len)) / 3
        ret.append(df[approx_tokens < max_resp_len])
    return ret

@ignore_warnings
def sample_incorrect_answers(dfs : list[pd.DataFrame], keep_ans_per_q : int = 5, filter_std_nonzero=False) -> list[pd.DataFrame]:
    """
        Given a list of pandas DataFrames this function returns the same dataframes, but
        only questions which are negatively answered.
    """
    ret = []
    dfs = filter_overlong_sequences(dfs)
    for df in dfs:
        df = prefilter(df, filter_std_nonzero=filter_std_nonzero)
        negdf = df[df['score'] <= 0]
        sampled = (
            negdf.groupby("input", sort=False, group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), keep_ans_per_q), random_state=0))
        )
        ret.append(sampled)
    return ret