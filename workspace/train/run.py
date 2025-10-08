#!/usr/bin/env python

import argparse
import os
import subprocess
from typing import Dict, List


def validate_file(path: str) -> str:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return path


def collect_export_vars(config: Dict[str, str]) -> List[str]:
    out: List[str] = []
    for k, v in config.items():
        if v is None:
            val = ""
        else:
            val = str(v)
        if "," in val or "\n" in val or " " in val:
            val = f"'{val}'"
        out.append(f"{k}={val}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Entrypoint router for VERL experiments (minimal required args)")

    parser.add_argument("--method", required=True, choices=["grpo", "dapo", "drgrpo"], help="Which training method to use")
    parser.add_argument("--model", type=str, default='Qwen/Qwen2.5-1.5B', help="Path to model or model id")
    parser.add_argument("--project-name", default="default", help="Project name for checkpoint organization")
    parser.add_argument("--identifier", default=None, help="Optional identifier appended to run name")
    parser.add_argument("--train-file", default="/u/rfechner/data/math/train.parquet")
    parser.add_argument("--val-file", default="/u/rfechner/data/math500/test.parquet")
    parser.add_argument("--cont", action="store_true", help="Continue existing checkpoint if present")
    parser.add_argument("--tp", type=int, default=4, help="Tensor model parallel size (tensor parallelism)")

    args = parser.parse_args()

    # Validate paths
    train_file = validate_file(args.train_file)
    val_file = validate_file(args.val_file)

    # Build experiment name (snake_case consistency: lowercase with underscores)
    base_model_name = os.path.basename(args.model).lower().replace("-", "_")
    if args.identifier:
        expname = f"{args.identifier.lower()}__{base_model_name}"
    else:
        expname = base_model_name

    checkpoint_dir = os.path.join("/u/rfechner/out", args.project_name, expname)

    # Check checkpoint existence guard
    if os.path.isdir(checkpoint_dir) and not args.cont:
        raise SystemExit(f"{checkpoint_dir} exists and --cont not provided. Aborting to avoid overwrite.")

    # Decide entrypoint script
    this_dir = os.path.dirname(os.path.abspath(__file__))
    entrypoint_script = os.path.join(this_dir, f"{args.method}_entrypoint.sh")

    # Build config with UPPERCASE env var names expected by shell scripts
    config = {
        "model_path": args.model,
        "train_files": train_file,
        "val_files": val_file,
        "project_name": args.project_name,
        "identifier": args.identifier or "",
        "checkpoint_dir": checkpoint_dir,
        "tensor_model_parallel_size": args.tp,
        "train_batch_size": 512,
        "total_epochs": 10,
        "max_prompt_length": 1024,
        "max_response_length": 3 * 1024,
        "learning_rate" : 0.000001, # 1e-6
        "save_freq" : 10,
        "test_freq" : 5,
        "group_n" : 8,
        "temperature" : 1.0, # training temperature == val temperature
        "top_k" : -1, # vllm rollouts
        "top_p" : 1.0, # training top-p
        "val_top_p" : 0.7
    }

    # Non-shared, hard-coded parameters in entrypoint scripts
    #
    # The entrypoint scripts (`dapo_entrypoint.sh` and `grpo_entrypoint.sh`) call
    # `python -m verl.trainer.main_ppo` and pass many flags. Most flags are
    # provided via the `config` dict above and therefore shared. The items
    # below are NOT exported from `run.py` and are hard-coded (or defaulted)
    # directly in the respective entrypoint script. Keep this list in sync
    # with those scripts if you change behavior.
    #
    # grpo_entrypoint.sh (hard-coded literals)
    #   actor_rollout_ref.actor.use_kl_loss=True
    #   actor_rollout_ref.actor.kl_loss_coef=0.001
    #   actor_rollout_ref.actor.kl_loss_type=low_var_kl
    #   actor_rollout_ref.actor.clip_ratio=0.2
    #
    # dapo_entrypoint.sh (hard-coded literals and bash-defaulted values)
    #   actor_rollout_ref.actor.kl_loss_coef=0.0
    #   actor_rollout_ref.actor.use_kl_loss=False
    #   actor_rollout_ref.actor.entropy_coeff=0.0
    #   actor_rollout_ref.actor.clip_ratio_low=0.2
    #   actor_rollout_ref.actor.clip_ratio_high=0.28
    #   actor_rollout_ref.actor.clip_ratio_c=10.0
    #   actor_rollout_ref.actor.loss_agg_mode='token-mean'
    #   algorithm.use_kl_in_reward=False
    #
    # dapo also provides defaults for reward-related flags using bash
    # parameter expansion. If the corresponding environment variables are
    # not exported these defaults are used by the script:
    #   reward_model.reward_manager -> default: "dapo" (via ${reward_model_reward_manager:-dapo})
    #   +reward_model.reward_kwargs.overlong_buffer_cfg.enable -> default: True
    #   +reward_model.reward_kwargs.overlong_buffer_cfg.len -> default: 512
    #   +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor -> default: -1.0
    #   +reward_model.reward_kwargs.overlong_buffer_cfg.log -> default: False
    #
    # Note: flags that compute values with shell arithmetic or reference
    # variables from `run.py` (e.g. $max_prompt_length, $max_response_length,
    # $grad_clip, $rollout_engine, etc.) are considered shared because their
    # values come from the exported `config` above.

    # Export shared-but-fixed parameters (these are set in both entrypoint scripts)
    config.update({
        # algorithm and data
        "algorithm_adv_estimator": "grpo",
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
        "rollout_val_n": 1,
        "rollout_disable_log_stats": False,
        "rollout_engine" : "vllm",
        # trainer shared flags
        "trainer_resume_mode": "auto",
        "trainer_logger": "console",
        "trainer_val_before_train": False,
        "trainer_n_gpus_per_node": 4,
        "trainer_nnodes": 2,
        "trainer_remove_previous_ckpt_in_save": False,
    })

    export_list = collect_export_vars(config)
    sbatch_cmd = ["sbatch", f"--export={','.join(export_list)}", entrypoint_script]

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
