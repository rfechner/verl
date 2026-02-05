#!/usr/bin/env python

import argparse
import os
import subprocess
import json
import pathlib
import time

from typing import Dict, List, Any
from checkpoint_utils import handle_checkpoint_validation, handle_model_method_validation
"""
    Main training run entrypoint. Purpose is to unite the export of environment
    variables for performance tuning, sampling and other critical hyperparameters,
    such that we may be sure that we're making fair comparisons.

    This script exports environment variables, which are accessed by the shell scripts which start the training. For ease of implementation
    There are some parameters which are only set INSIDE the shell scripts, such that we can, but musn't use every exported parameter. Additionally,
    there are some non-standart parameters which we have to set for some algorithms, it's better to hardcode these.
"""
def assert_num_lines_mod_ngpus_iszero(path: str, num_gpus : int = 8) -> None:
    assert path.endswith('jsonl'), "Only applicable for jsonl files"
    
    with open(path, "r") as f:
        num_lines = sum(1 for _ in f)

    if num_lines % num_gpus != 0:
        raise ValueError("Number of lines in path isn't divided evenly by number of GPUs. This will cause verl to throw later on.")
    return
    
def collect_export_vars(config: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for k, v in config.items():
        # Convert lists/dicts to JSON strings so Hydra can parse them
        if isinstance(v, (list, dict)):
            val = json.dumps(v)
        elif v is None:
            val = ""
        else:
            val = str(v)
        
        # # Quote values containing special characters or spaces
        # if any(c in val for c in [",", "\n", " "]):
        #     raise ValueError("Please try to not pass values containing special characters ',', '\\n', ' '")
        
        out.append(f"{k}={val}")
    return out

models = {
    'llama3-small' : "meta-llama/Llama-3.2-1B-Instruct",
    'llama3-medium' : "meta-llama/Llama-3.2-3B-Instruct",
    'llama3-large' : "meta-llama/Llama-3.1-8B-Instruct",
    'qwen2.5-small' : "Qwen/Qwen2.5-0.5B",
    'qwen2.5-medium' : "Qwen/Qwen2.5-1.5B",
    'qwen2.5-large' : "Qwen/Qwen2.5-7B",
    'qwen3-small' : "Qwen/Qwen3-0.6B",
    'qwen3-medium' : "Qwen/Qwen3-4B",
    'qwen3-large' : "Qwen/Qwen3-8B",
    'moxin' : "moxin-org/Moxin-7B-Instruct",
    'r1-qwen-medium' : "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    'r1-qwen-large' : "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
    'r1-llama-large' : "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    'eurollm' : "utter-project/EuroLLM-9B-Instruct",
    "olmo3" : "allenai/Olmo-3-7B-Instruct-SFT" # have to use flashinfer environment for this to work.
    # 'gpt-oss' : 'openai/gpt-oss-20b' # currently not working because of quantization errors.
}

def main():
    method_valid_choices = [
        'grpo',
        'drgrpo',
        'gspo',
        'dapo',
        'kl-cov',
        'clip-cov',
        'grpo-s',
        'gtpo',
        'entropy_reg',
        'grpo-passk', # pass@k training
        "klcov_passk" # KL-Cov and Pass@k advantage estimation
    ]

    parser = argparse.ArgumentParser(description="Entrypoint router for VERL experiments (minimal required args)")
    parser.add_argument("--method", required=True, choices=method_valid_choices, help="Which training method to use")
    parser.add_argument("--model", type=str, default='meta-llama/Llama-3.2-1B-Instruct', choices=list(models.keys()) + list(models.values()), help="Path to model or model id")
    parser.add_argument("--project-name", required=True, help="Project name for checkpoint organization")
    parser.add_argument("--identifier", default=None, help="Optional identifier appended to run name")
    parser.add_argument("--train-file", default="/u/rfechner/data/dapo17k/train.parquet")
    parser.add_argument("--cont", action="store_true", help="Continue existing checkpoint if present")
    parser.add_argument("--tp", type=int, default=4, help="Tensor model parallel size (tensor parallelism)")
    parser.add_argument("--valn", type=int, default=8, help="Number of validation samples to dump per validation step.")
    #parser.add_argument("--flashinfer", action="store_true", help="Activate flashinfer conda env instead of verl when set")
    parser.add_argument("--no-logprobs", action='store_true', default=False, help='flag: do not compute logprobs for pre-rollouts')
    parser.add_argument("--no-validation", action='store_true', default=False, help='flag: do not calculate/dump validation rollouts.')
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--savefreq", type=int, default=10)
    parser.add_argument("--testfreq", type=int, default=5)
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--cp", type=str, default=None, help="Model Checkpoint to train/eval from. Specify 'global_step_X' directory.")
    parser.add_argument("--train_batchsize", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1, help='Random seed for trainloader initialization.')

    # ======== EVAL OPTIONS ========
    parser.add_argument("--eval", action="store_true", help="Runs evaluation from given checkpoint. Needs --checkpoint to be specified")
    parser.add_argument("--cpdir", type=str, default=None, help="Model Checkpoint directory to load checkpoints from. Note: This can only be set in case we're evaluating.")
    parser.add_argument("--lpfile", type=str, default=None, help="Path to jsonl file containing chats. Note: This can only be set in case we're evaluating.")
    parser.add_argument("--lpdir", type=str, default=None, help="Path to directory to load `X_rollout.jsonl` files from to calculate logprobs.")
    parser.add_argument("--dryrun", action='store_true', default=False, help='Whether to dryrun the current experiment. This will terminate the script before delegating to sbatch.')
    parser.add_argument("--val_batchsize", type=int, default=512) # If set to none, whole batch is sent to inference engine.
    parser.add_argument("--val_data", type=str, default=None, help='Validation file. If not provided this will default to Math500 + MathArena datasets.')
    parser.add_argument("--only-first-last-checkpoint", action="store_true", default=False, help='Conditional Flag. Only allwed when chdir !=  None. Will trigger evaluation of only first (base model) and last checkpoint from the provided checkpoint directory.')
    
    # ======== TEMPORARY OPTIONS FOR TESTING ========
    parser.add_argument("--filter-solved", action="store_true", help="Whether to drop solved (solverate > 1/4) samples from the dataframe on epoch beginning.")
    parser.add_argument("--dapo-seq-reward", action="store_true", help="Whether to filter for sequence rewards instead of final reward. Currently only used acc.")
    
    args = parser.parse_args()
    
    # ======== Argument Verification ========

    if args.filter_solved:
        if not args.method == "dapo":
            raise ValueError("filter_solved currently only available on DAPO.")
        
    # paths correct?
    if args.lpfile:
        if not os.path.isabs(args.lpfile):
            raise ValueError("Please give absolute path for validation.")
        if not os.path.isfile(args.lpfile):
            raise ValueError(f"Specified lpfile isn't a file: {args.lpfile}")
        assert_num_lines_mod_ngpus_iszero(args.lpfile, args.nodes * 4)
        
    if args.lpdir:
        if not os.path.isabs(args.lpdir):
            raise ValueError("Please give absolute path for validation.")
        if not os.path.isdir(args.lpdir):
            raise ValueError(f"Specified lpdir isn't a directory: {args.lpdir}")
    
    if args.cpdir:
        if not os.path.isabs(args.cpdir):
            raise ValueError("Please give absolute path.")
        if not os.path.isdir(args.cpdir):
            raise ValueError(f"Specified cpdir isn't a directory: {args.cpdir}")
    
    if args.cp:
        if not os.path.isabs(args.cp):
            raise ValueError("Please give absolute path.")
        if not os.path.isdir(args.cp):
            raise ValueError(f"Specified cp isn't a directory: {args.cp}")
          
    # resolve alias
    if args.model in models:
        args.model = models[args.model]
        
    if args.no_logprobs:
        if args.lpfile or args.lpdir:
            raise ValueError('specified no logprobs, but gave lpfile or lpdir in args. Wrong experiment setup?')
    
    if args.eval:
        if args.cp and args.cpdir:
            raise ValueError("Cannot load from checkpoint --cp and checkpoint directory --cpdir")
        if args.no_logprobs and args.no_validation:
            raise ValueError("Both, --no-logprobs and --no-validation was set.")
    
    if args.cp:
        path = pathlib.Path(args.cp).expanduser().resolve()
        assert path.is_dir() and path.exists() and path.parts[-1].startswith('global_step'), \
            "Malformed checkpoint path."
        args.cp = path

        handle_checkpoint_validation(args.cp, args.nodes * 4) # 4 GPUs per Node have to match world_size of chkpt
        handle_model_method_validation(args.cp, args.method, args.model) # for re-loading. Make sure method/model is chosed correctly.

    if args.cpdir:
        if not args.eval:
            raise ValueError("Cannot load checkpoints from directory source if not in eval.")
        
        files = os.listdir(args.cpdir)
        checkpoint_files = list(filter(lambda x: x.startswith('global_step_'), files))
        assert len(checkpoint_files) > 0, f"No checkpoints found at directory: {args.cpdir}"

        random_checkpoint = os.path.join(args.cpdir, checkpoint_files[0])
        handle_checkpoint_validation(random_checkpoint, args.nodes * 4) # 4 GPUs per Node have to match world_size of chkpt
        handle_model_method_validation(random_checkpoint, args.method, args.model) # for re-loading. Make sure method/model is chosed correctly.
    elif args.cp:
        handle_checkpoint_validation(args.cp, args.nodes * 4) # 4 GPUs per Node have to match world_size of chkpt
        handle_model_method_validation(args.cp, args.method, args.model) # for re-loading. Make sure method/model is chosed correctly.
    else: # we're training
        if args.cpdir:
            raise ValueError("Training but checkpoint directory was set. This option is only used in evaluation. To set a checkpoint use --cp pointing towards .../global_step_X directory.")
        if args.no_validation:
            raise ValueError("Running training without validation.")
        if args.valn > 16:
            raise ValueError("Running training with too high number of validation rollouts.")

    
    # from huggingface_hub import HfFolder, login
    # from pathlib import Path

    # token_path = Path.home() / ".cache/huggingface/token"

    # # Check if a token is already stored
    # stored_token = None #HfFolder.get_token()       Have to login every session because of SLURM weirdness

    # if stored_token is None:
    #     print("Not logged in. Logging in...")
    #     token = token_path.read_text().strip()
    #     login(token=token)
    #     print("Logged into Hugging Face Hub.")
    # else:
    #     print("Already logged in.")
            
    # Build experiment name (snake_case consistency: lowercase with underscores)
    base_model_name = os.path.basename(args.model).lower().replace("-", "_")
    if args.identifier:
        expname = f"{args.identifier.lower()}__{base_model_name}__{args.method}"
    else:
        expname = f"{base_model_name}__{args.method}"

    # temporary files in raven cluster. Warning: files not accessed for ~12 weeks are deleted.
    logger_root="/ptmp/rfechner/out"
    checkpoint_dir = os.path.join(logger_root, args.project_name, expname)

    # Check checkpoint existence guard
    if os.path.isdir(checkpoint_dir) and not args.cont:
        raise SystemExit(f"{checkpoint_dir} exists and --cont not provided. Aborting to avoid overwrite.")

    # Decide entrypoint script
    this_dir = os.path.dirname(os.path.abspath(__file__))
    
    entrypoint_script = {
        'grpo' : 'grpo_entrypoint.sh',
        'gtpo' : 'grpo_entrypoint.sh',
        'grpo-passk' : 'grpo_entrypoint.sh',
        'grpo-s' : 'grpo_entrypoint.sh',
        'entropy_reg' : 'grpo_entrypoint.sh',
        'kl-cov' : 'klcov_entrypoint.sh',
        'clip-cov' : 'klcov_entrypoint.sh',
        'gspo' : 'gspo_entrypoint.sh',
        'dapo' : 'dapo_entrypoint.sh',
        'drgrpo' : 'drgrpo_entrypoint.sh',
        'klcov_passk' : 'klcov_entrypoint.sh'
    }[args.method]

    entrypoint_script = os.path.join(this_dir, entrypoint_script)
    # base hyperparams. These are the variable and "important" parameters
    config = {

        # Precomputed Q+A file for which to compute log-probs during validation
        "trainer_compute_logprob_from_file": args.lpfile if not args.no_logprobs else False, # for calculating logprobs from single file -> could be regular chat or chat_with_suffix
        "trainer_compute_logprob_from_rollout_dir" : args.lpdir if not args.no_logprobs else False, # for calculating lps from directory
        "trainer_grid_checkpoint_directory" : args.cpdir if args.eval else False, # for multiple checkpoint evaluation sweep
        "trainer_resume_mode": "resume_path" if args.cp else "auto", # for single checkpoint evaluation or continued training
        "trainer_resume_path" : args.cp,
        "trainer_skip_validation" : args.no_validation,
        "trainer_skip_logprobs" : args.no_logprobs,
        "trainer_only_first_last_checkpoint" : args.only_first_last_checkpoint,
        
        # Where to dump validation generations (placed next to checkpoints by default)
        "trainer_validation_data_dir": os.path.join(checkpoint_dir, "val_jsonl"),
        
        # Where to dump rollout generations (placed next to checkpoints by default)
        "trainer_rollout_data_dir" : os.path.join(checkpoint_dir, "train_jsonl"),

        "model_path": args.model,
        "train_files": args.train_file,
        "identifier": args.identifier or "",
        "project_name": args.project_name,
        "experiment_name" : expname,
        "checkpoint_dir": checkpoint_dir,
        "tensor_model_parallel_size": args.tp,
        "rollout_val_n" : args.valn,
        "train_batch_size": args.train_batchsize,
        "total_epochs": args.epochs,
        "max_prompt_length": 1024,
        "max_response_length": 3 * 1024,
        "learning_rate" : 0.000001, # 1e-6
        "save_freq" : args.savefreq,
        "test_freq" : args.testfreq,
        "group_n" : 8,
        "temperature" : 1.0, # training temperature == val temperature
        "top_k" : -1, # vllm rollouts
        "top_p" : 1.0, # training top-p
        "val_top_p" : 0.7,
        "gtpo" : args.method=='gtpo',
        "grpo_s" : args.method == 'gpro-s',
        "loss_mode" : args.method.replace('-', '_') if args.method in ['gspo', 'kl-cov', 'clip-cov'] else 'vanilla',
        "entropy_coeff" : 0.001 if args.method == 'entropy_reg' else 0
    }

    # Parameters which may be set or unset to trigger default behaviours
    config.update({"val_batch_size" : args.val_batchsize} if args.val_batchsize else {})
    config.update({'data_val_files' : args.val_data} if args.val_data else {})
    config.update({"algorithm_filter_solved" : args.filter_solved} if args.filter_solved else {})
    config.update({"dapo_filter_groups_metric" : "seq_reward" if args.dapo_seq_reward else 'acc'})

    # Export shared-but-fixed parameters (these are set in both entrypoint scripts)
    config.update({
        "data_seed" : args.seed,
        # Environement flags
        'VERL_FILE_LOGGER_ROOT' : logger_root,
        # algorithm and data
        "algorithm_adv_estimator": "grpo_passk" if args.method in ["grpo-passk", 'klcov_passk'] else "grpo",
        "data_truncation": "left",
        
        # actor model flags
        "actor_model_use_remove_padding": True,
        "actor_model_enable_gradient_checkpointing": True,
        "actor_model_enable_activation_offload": True,
        
        # ref / rollout flags
        "ref_log_prob_use_dynamic_bsz": True,
        "ref_fsdp_param_offload": True,
        "ref_ulysses_sequence_parallel_size": 1,
        "ref_entropy_checkpointing": True,
        "ref_fsdp_forward_prefetch": True,
        "ref_strategy": "fsdp2",
        
        # actor fsdp / dynamic flags
        "gpu_memory_utilization" : 0.8,
        "enable_chunked_prefill" : True,
        "actor_use_dynamic_bsz": True,
        "actor_fsdp_forward_prefetch": True,
        "actor_fsdp_param_offload": True,
        "actor_fsdp_optimizer_offload": True,
        "actor_ulysses_sequence_parallel_size": 1,
        "actor_strategy": "fsdp2",
        "grad_clip" : 1.0,

        # rollout shared flags
        "rollout_log_prob_use_dynamic_bsz": True,
        "rollout_val_do_sample": True,
        
        # batch size for computing log-probs on actor workers
        "trainer_compute_logprob_batch_size": 64,
        "rollout_disable_log_stats": False,
        "rollout_engine" : "vllm",
        "trainer_val_before_train": True, # updated from args.eval | False, because we always want rollouts_0 and so on.
        "trainer_only_evaluate" : args.eval,
        "trainer_n_gpus_per_node": 4,
        "trainer_nnodes": 2,
        "trainer_remove_previous_ckpt_in_save": False,
    })

    # export which conda env to activate in the entrypoint scripts
    conda_env = "flashinfer" # if args.flashinfer else "verl"
    config.update({
        "conda_env": conda_env,
    })

    export_list = collect_export_vars(config)
    sbatch_cmd = ["sbatch", f"--export={','.join(export_list)}", entrypoint_script]

    if args.dryrun:
        print("✅ Passed preliminary checks. Experiment seems well formed. Exited with --dryrun option.")
        return
    
    try:
        result = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)
        print("Job submitted:")
        print(result.stdout.strip())
        if result.stderr:
            print("sbatch stderr:")
            print(result.stderr.strip())
    except subprocess.CalledProcessError as e:
        print("sbatch failed:")
        print(e.stdout)
        print(e.stderr)
        raise


if __name__ == "__main__":
    main()
