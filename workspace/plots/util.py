import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

method_color_mapper = {
    'grpo-s': 'red',
    'kl-cov': 'green',
    'grpo-passk': 'blue',
    'gspo': 'orange',
    'grpo' : 'yellow',
    'drgrpo' : 'cyan',
    'dapo' : 'pink',
    'gtpo' : 'brown',
    'clip-cov' : 'violet',
    'base' : 'black'
}

model_marker_mapper = {
    k : v for k, v in zip(['qwen2.5_7b', 'qwen2.5_1.5b', 'eurollm_9b_instruct', 'llama_3.1_8b_instruct'], \
                           ['*', '-' '+', '^', 'o'])
}

method_name_mapper = {
    'grpo-s' : "GRPO-S",
    'grpo-passk' : "Pass@k Training",
    'gspo' : "GSPO",
    'kl-cov' : "KL-Cov",
    'grpo' : "GRPO",
    'drgrpo' : "Dr.GRPO",
    'clip-cov' : "Clip-Cov",
    'gtpo' : "GTPO",
    'entropy_reg' : "Entropy Reg",
    'dapo' : "DAPO"
}
model_name_mapper = {
    'qwen2.5_7b' : "Qwen2.5-7B",
    "eurollm_9b_instruct" : "EuroLLM-9B-Instruct",
    "qwen2.5_1.5b": "Qwen2.5-1.5B",
    "llama_3.1_8b_instruct" : "Llama3.1-8B-Instruct"
}

def pass_at_k(rewards, k):
    """
    Compute pass@k for a list or array of binary rewards.
    Args:
        rewards (list or np.ndarray): Binary vector where 1 = success, 0 = failure.
        k (int): The number of samples considered for pass@k.
    Returns:
        float: Estimated pass@k value.
    """
    rewards = np.array(rewards)
    n = len(rewards)
    c = rewards.sum()
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - np.prod([(n - c - i) / (n - i) for i in range(k)]) 

def calculate_pass_at_k(data : pd.DataFrame, ks = 2**np.arange(9)):
    pass_at_values = {
        'math500' : [],
        'aime25' : [],
        'brumo25' : [],
        'cmimc25' : [],
        'hmmt25' : []
    }
    def get_key(index : int):
        """
        First 500 are math500,
        next 30 are aime25,
        next 30 are brumo25,
        next 40 are cmimc25,
        last 30 are hmmt25
        """
        if index < 500:
            return 'math500'
        elif index < 530:
            return 'aime25'
        elif index < 560:
            return 'brumo25'
        elif index < 600:
            return 'cmimc25'
        elif index < 630:
            return 'hmmt25'
        else:
            raise ValueError('Index > 630. Did validation set change?')
        
        
    for i, (name, group) in enumerate(data.groupby('input', sort=False)):
        pass_at_values[get_key(i)].append(
            [pass_at_k(rewards=group['reward'], k=k) for k in ks]
        )
    return {k : np.stack(vs) for k, vs in pass_at_values.items()}

def rollouts_from_disk(root) -> dict:
    with open(os.path.join(root, 'cache.pickle'), 'rb') as file:
        p : dict = pickle.load(file)
    return p

def qw7_rollouts(recompute=True) -> dict:
    """
    Collect rollouts for experiment 05. 
    """
    print("Reading Qwen2.5-7B rollouts")
    if not recompute:
        return rollouts_from_disk(root = "/ptmp/rfechner/out/exp05_rollouts_qwen2.5-7b")
    
    qwen_root = "/ptmp/rfechner/out/exp05_rollouts_qwen2.5-7b"
    rollout_paths = {d : os.path.join(qwen_root, d, 'val_jsonl') for d in os.listdir(qwen_root) if os.path.isdir(os.path.join(qwen_root, d))}

    passk = {}
    for method, path in rollout_paths.items():
        rollout_files = list(
                            sorted(
                                filter(lambda x: x.endswith('_rollouts.jsonl'), os.listdir(path)),
                                key=lambda x: x.split('_')[0])
                            )
        passrates = {}
        for file in tqdm(rollout_files, desc=method):
            with open(os.path.join(path, file), 'r') as jsonfile:
                checkpoint_index = file.split('_')[0]
                df = pd.read_json(jsonfile, lines=True)
                pass_at_values = calculate_pass_at_k(df, ks=2**np.arange(9))
                

                passrates[checkpoint_index] = pass_at_values
        passk[method] = passrates

    with open(os.path.join(qwen_root, 'cached.pickle'), 'wb') as file:
        pickle.dump(passk, file)

    # we have to correct for the base model archiving different pass@k values. In this analysis we'll just take the first estimate.
    base_model_mapper = {model : None for model in set([m.split('__')[0] for m in rollout_paths.keys()])}
    for method in rollout_paths.keys():
        m = method.split('__')[0]
        if base_model_mapper[m] is None:
            base_model_mapper[m] = passk[method]['0']
        
        # overwrite the base model's pass@k with the first found pass@k estimate to equalize pass@k for base model over methods.
        passk[method]['0'] = base_model_mapper[m]

    print(f"Serializing to {os.path.join(qwen_root, 'cache.pickle')}")
    with open(os.path.join(qwen_root, 'cache.pickle'), 'wb') as file:
        pickle.dump(obj=passk, file=file)

    return passk

def rollouts(recompute=True) -> dict:
    """
    Collect rollouts for experiment 13. Rollouts are split across two experiment directories, as we're including
    past rollouts for qwen2.5-7b which we don't have to re-run.
    """
    print("Reading Experiment 13 + selective Qwen2.5-7B rollouts.")
    if not recompute:
        return rollouts_from_disk(root = "/ptmp/rfechner/out/exp13_rollouts")
    exp13_root = "/ptmp/rfechner/out/exp13_rollouts/"
    qwen_root = "/ptmp/rfechner/out/exp05_rollouts_qwen2.5-7b"

    # mask for methods of interest
    qwen_method_mask = ['grpo-passk', 'gspo', 'grpo-s', 'kl-cov']

    exp13_dirs = os.listdir(exp13_root)
    exp13_dirs = list(filter(lambda x: os.path.isdir(os.path.join(exp13_root, x)), exp13_dirs))

    qwen_dirs = os.listdir(qwen_root)
    qwen_dirs = list(filter(lambda x: os.path.isdir(os.path.join(qwen_root, x)) and x.split('__')[-1] in qwen_method_mask, qwen_dirs))

    exp13_rollout_paths = {
        d : os.path.join(exp13_root, d, 'val_jsonl') for d in exp13_dirs
    }

    qwen_rollout_paths = {
        d : os.path.join(qwen_root, d, 'val_jsonl') for d in qwen_dirs
    }
    exp13_rollout_paths.update(qwen_rollout_paths)
    
    print(f"Iterating over paths:\n", "\n".join(exp13_rollout_paths.values()))
    passk = {}
    for method, path in exp13_rollout_paths.items():

        # sort rollouts files: 0_rollouts.jsonl, 10_rollouts.jsonl,..., 80_rollouts.jsonl
        rollout_files = list(
                            sorted(
                                filter(lambda x: x.endswith('_rollouts.jsonl'), os.listdir(path)),
                                key=lambda x: x.split('_')[0])
                            )
        passrates = {}
        for file in tqdm(rollout_files, desc=method):
            with open(os.path.join(path, file), 'r') as jsonfile:
                checkpoint_index = file.split('_')[0]
                df = pd.read_json(jsonfile, lines=True)
        
                # per-method, per-checkpoint get pass@1, pass@2, ..., pass@256 values.
                pass_at_values = calculate_pass_at_k(df)
                passrates[checkpoint_index] = pass_at_values
        passk[method] = passrates

    # we have to correct for the base model archiving different pass@k values. In this analysis we'll just take the first estimate.
    base_model_mapper = {model : None for model in set([m.split('__')[0] for m in exp13_rollout_paths.keys()])}
    for method in exp13_rollout_paths.keys():
        m = method.split('__')[0]
        if base_model_mapper[m] is None:
            base_model_mapper[m] = passk[method]['0']
        
        # overwrite the base model's pass@k with the first found pass@k estimate to equalize pass@k for base model over methods.
        passk[method]['0'] = base_model_mapper[m]

    print(f"Serializing to {os.path.join(exp13_root, 'cache.pickle')}")
    with open(os.path.join(exp13_root, 'cache.pickle'), 'wb') as file:
        pickle.dump(obj=passk, file=file)

    return passk


def calc_deltas(recompute=True):
    root = "/ptmp/rfechner/out/exp13_deltas/"
    if not recompute:
        return rollouts_from_disk(root=root)

    dirs = list(filter(lambda x: os.path.isdir(os.path.join(root, x)), os.listdir(root)))
    dataframes = {}
    for d in dirs:
        dataframes[d] = {}
        method_directory = os.path.join(root, d, "val_jsonl")
        sorted_rollouts = list(sorted(
            os.listdir(method_directory),
            key=lambda rollout: int(rollout.split('_')[0])
        ))
        rollout_paths = [os.path.join(method_directory, r) for r in sorted_rollouts]
        for sr, checkpoint in zip(sorted_rollouts, rollout_paths):
            checkpoint_i = sr.split('_')[0]
            with open(checkpoint, 'r') as file:
                dataframes[d][checkpoint_i] = pd.read_json(file, lines=True)

    def calculate_deltas():
        """
            Should return a dictionary ready to pre-filter.

            d:
                methodX:
                    checkpoint_0:
                        error_types: 
                            calculation_error: np.array([-0.1, -0.2, ...])
                            formula_error: ...
                            anchor: np.array([-0.01, ...])
                        reasoning_types:
                            inductive_reasoning: np.array([-0.5, -0.1, ...])
                            deductive_reasoning: ...
                            anchor: np.array([-0.01, ...])
                        reasoning_strategy: 
                            backtracking: np.array([-0.5, -0.1, ...])
                            validation: ...
                            anchor: np.array([-0.01, ...])
                    checkpoint_1:
                        error_types: 
                            calculation_error: np.array([-0.1, -0.2, ...])
                            formula_error: ...
                            anchor: np.array([-0.01, ...])
                        reasoning_types:
                            inductive_reasoning: np.array([-0.5, -0.1, ...])
                            deductive_reasoning: ...
                            anchor: np.array([-0.01, ...])
                        reasoning_strategy: 
                            backtracking: np.array([-0.5, -0.1, ...])
                            validation: ...
                            anchor: np.array([-0.01, ...])
                    ...
                ... 
        """

        def calc_delta_for_taxonomy(dataframe, taxonomy:str):
            df = dataframe[dataframe['taxonomy'] == taxonomy].copy()

            # Compute per-row means
            df['lpmean'] = df['logprobs'].apply(np.mean)
            # df['entmean'] = df['entropy'].apply(np.mean)

            behaviours = df['behaviour_type'].unique()
            taxonomy_mapper = {k : [] for k in behaviours} # includes anchor as the "ground truth behaviour"
            if taxonomy=='error_types':
                taxonomy_mapper['anchor'] = {k : [] for k in behaviours if k != 'anchor'}

            for q, per_q_df in df.groupby('index', sort=False):
                anchor = per_q_df[per_q_df['behaviour_type']=='anchor']
                augs = per_q_df[per_q_df['behaviour_type']!='anchor']
                
                # subtract others from anchor
                lpdelta =  augs['lpmean'].to_numpy() - anchor['lpmean'].to_numpy()
                # entdelta =  augs['entmean'].to_numpy() - anchor['lpmean'].to_numpy()
                for k, v in zip(augs['behaviour_type'], lpdelta):
                    taxonomy_mapper[k].append(v)
                if taxonomy=='error_types': # weird edge case.
                    for k, v in zip(augs['behaviour_type'], anchor['lpmean']):
                        taxonomy_mapper['anchor'][k].append(v)
                else:
                    taxonomy_mapper['anchor'].append(anchor['lpmean'])

            ret = {k : np.array(v) for k, v in taxonomy_mapper.items()}
            return ret
        
        ret = {}
        for method, checkpoints in tqdm(dataframes.items(), desc='Methods: '):
            ret[method] = {}
            for checkpoint, rollouts in checkpoints.items():

                error_types = calc_delta_for_taxonomy(rollouts, taxonomy='error_types')
                reasoning_types = calc_delta_for_taxonomy(rollouts, taxonomy='reasoning_types')
                reasoning_strategies = calc_delta_for_taxonomy(rollouts, taxonomy='reasoning_strategies')
                ret[method][checkpoint] = {
                    'error_types' : error_types,
                    'reasoning_types' : reasoning_types,
                    'reasoning_strategies' : reasoning_strategies
                }
        return ret
    
    out = calculate_deltas()
    print(f"Serializing to {os.path.join(root, 'cache.pickle')}")
    with open(os.path.join(root, 'cache.pickle'), 'wb') as file:
        pickle.dump(obj=out, file=file)

    return out


    
