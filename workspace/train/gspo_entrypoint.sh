#!/bin/bash
#SBATCH --job-name=gspo
#SBATCH --output=/u/rfechner/jobs/gspo_%j.out   # Standard output file (%j expands to job ID)
#SBATCH --error=/u/rfechner/jobs/gspo_%j.err    # Standard error file (%j expands to job ID)
#SBATCH --nodes=2
#SBATCH --exclusive             
#SBATCH --partition=general                                 # main partition containing GPU nodes
#SBATCH --constraint=gpu-bw                                 # high bandwidth GPUs within general partition  
#SBATCH --switches=1                                        # constraint on the topology -> we want adjacent nodes
#SBATCH --ntasks-per-node=1                                 # Run the shell script as a single task on this node
#SBATCH --cpus-per-task=72                                  # Allocate all CPUs on the node (Raven GPU nodes often have 72 cores, adjust if different)
#SBATCH --gres=gpu:a100:4                                   
#SBATCH --time=23:59:00                                     # Maximum wall clock time (e.g., 24 hours). Adjust as needed.
#SBATCH --mem=500000                                        # Request all available memory on the node. Can be specific e.g., "200G".
#SBATCH --chdir=./                                          # Set working directory to submission directory.

# vLLM + FLashinfer and multinode recommended flags
export NCCL_CUMEM_ENABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_LEVEL=SYS
export NCCL_LAUNCH_MODE=GROUP
export CUDA_DEVICE_MAX_CONNECTIONS=32
if [ "${conda_env:-}" = "flashinfer" ]; then
    export VLLM_USE_FLASHINFER=1
    export FLASHINFER_ENABLE=1
else
    export VLLM_USE_FLASHINFER=0
    export FLASHINFER_ENABLE=0
fi

# torch dynamo compilation directories. Needed for jobs surges, as SLURM+Lustre+torch dynamo doesn't like concurrent compilation.
# export TORCHINDUCTOR_CACHE_DIR=/ptmp/rfechner/.cache/torch_inductor_$SLURM_JOB_ID
# export TRITON_CACHE_DIR=/ptmp/rfechner/.cache/triton_$SLURM_JOB_ID
# export VLLM_CACHE_DIR=/ptmp/rfechner/.cache/vllm_$SLURM_JOB_ID
# export XDG_CACHE_HOME=/ptmp/rfechner/.cache/xdg_$SLURM_JOB_ID
# huggingface token for login
export HUGGINGFACE_HUB_TOKEN=$(tr -d '\n' < $HOME/.cache/huggingface/token)
echo "========================================================"
echo "Starting VERL Training Job: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated GPUs: $SLURM_JOB_GPUS"
echo "========================================================"

## Router is expected to export and validate all needed environment variables.
## This script will use the environment variables provided by the router. Do not set defaults here.

echo "Logging to ${checkpoint_dir}"
echo "Configuration:"
echo "  Identifier: $identifier"
echo "  Model: $model_path"
echo "  Project name: $project_name"
echo "  Train files: $train_files"
echo "========================================================"

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load modules and activate your Conda env
# ─────────────────────────────────────────────────────────────────────────────
echo "Loading environment modules..."
module purge
module load anaconda/3/2023.03
module load cuda/12.6
module load cudnn/9.5.1
module load nccl/2.23.4
module load gcc/10

module list

# Unset AMD GPU default flags. Required by verl.
unset ROCR_VISIBLE_DEVICES

# >>> conda initialize >>>
__conda_setup="$('/mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh" ]; then
        . "/mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh"
    else
        export PATH="/mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# Activate your environment (use exported `conda_env`, default to 'verl')
if [ -z "${conda_env:-}" ]; then
    target_conda_env="verl"
else
    target_conda_env="$conda_env"
fi
echo "Activating conda environment: $target_conda_env"
conda activate "$target_conda_env"

# ─────────────────────────────────────────────────────────────────────────────
# 2) Set up Ray cluster (only if using multiple nodes)
# ─────────────────────────────────────────────────────────────────────────────
echo "Setting up Ray cluster for multi-node training..."

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
else
    head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
export RAY_ADDRESS="$ip_head"

echo "IP Head: $ip_head"
echo "RAY_ADDRESS: $RAY_ADDRESS"

# Set number of GPUs using SLURM environment variable
num_gpus=${SLURM_GPUS_PER_NODE:-4}  # Use SLURM detection, fallback to 4

# NOTE: we have to "--export=ALL,ROCR_VISIBLE_DEVICES" to propagate set environment variables
# and importantly the unset operation on ROCR_VISIBLE_DEVICES to the worker nodes, as verl requires
# either ROCR_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES to be set.
echo "Starting HEAD at $head_node"
srun --export=ALL,ROCR_VISIBLE_DEVICES --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --dashboard-host 0.0.0.0 --dashboard-port=8265 \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "$num_gpus" --block &

# Give head node more time to initialize
sleep 30

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --export=ALL,ROCR_VISIBLE_DEVICES --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "$num_gpus" --block &
    sleep 5
done

# Give cluster time to fully connect
sleep 15

# ─────────────────────────────────────────────────────────────────────────────
# 3) Start Training
# Used most tips from: https://verl.readthedocs.io/en/latest/perf/perf_tuning.html
# ─────────────────────────────────────────────────────────────────────────────

# i have to do it this way, as exporting via python + hydra parsing is horrible.
math500=/u/rfechner/data/math500/test.parquet
aime25=/u/rfechner/data/aime25/test.parquet
brumo2025=/u/rfechner/data/brumo_2025/test.parquet
cmimc2025=/u/rfechner/data/cmimc_2025/test.parquet
hmmt2025=/u/rfechner/data/hmmt_feb_2025/test.parquet
default_val_files="['$math500', '$aime25', '$brumo2025', '$cmimc2025', '$hmmt2025']"


# GSPO params.
loss_agg_mode="seq-mean-token-mean"
reward_manager=dapo
use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0
clip_ratio_low=0.0003 # as recommended by the paper, see Sec. 5.1
clip_ratio_high=0.0004 # as recommended by the paper, see Sec. 5.1
overlong_buffer_cfg_enable=false

# delegates to the dapo trainer, which does the dynamic sampling.
python -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${algorithm_adv_estimator} \
    data.train_files="$train_files" \
    data.val_files="${data_val_files:-$default_val_files}" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation=${data_truncation} \
    data.val_batch_size=${val_batch_size} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.model.use_remove_padding=${actor_model_use_remove_padding} \
    actor_rollout_ref.model.use_fused_kernels=true \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.enable_gradient_checkpointing=${actor_model_enable_gradient_checkpointing} \
    actor_rollout_ref.model.enable_activation_offload=${actor_model_enable_activation_offload} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((1 * (max_prompt_length + max_response_length))) \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${ref_log_prob_use_dynamic_bsz} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_fsdp_param_offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ref_ulysses_sequence_parallel_size} \
    actor_rollout_ref.ref.entropy_checkpointing=${ref_entropy_checkpointing} \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=${ref_fsdp_forward_prefetch} \
    actor_rollout_ref.ref.strategy="${ref_strategy}" \
    actor_rollout_ref.actor.optim.lr=$learning_rate \
    actor_rollout_ref.actor.use_dynamic_bsz=${actor_use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((1 * (max_prompt_length + max_response_length))) \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=${actor_fsdp_forward_prefetch} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_fsdp_param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_fsdp_optimizer_offload} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.grad_clip=${grad_clip} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${actor_ulysses_sequence_parallel_size} \
    actor_rollout_ref.actor.strategy="${actor_strategy}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${rollout_log_prob_use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((1 * (max_prompt_length + max_response_length))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.enable_chunked_prefill=${enable_chunked_prefill} \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((6 * (max_prompt_length + max_response_length))) \
    actor_rollout_ref.rollout.name=${rollout_engine} \
    actor_rollout_ref.rollout.n=${group_n} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${rollout_val_do_sample} \
    actor_rollout_ref.rollout.val_kwargs.n=${rollout_val_n} \
    actor_rollout_ref.rollout.disable_log_stats=${rollout_disable_log_stats} \
    trainer.rollout_data_dir=$trainer_rollout_data_dir \
    +trainer.validation_data_dir=${trainer_validation_data_dir} \
    +trainer.compute_logprob_from_file=${trainer_compute_logprob_from_file} \
    +trainer.compute_logprob_batch_size=${trainer_compute_logprob_batch_size} \
    +trainer.compute_logprob_from_rollout_dir=${trainer_compute_logprob_from_rollout_dir} \
    +trainer.grid_checkpoint_directory=${trainer_grid_checkpoint_directory} \
    +trainer.skip_validation=$trainer_skip_validation \
    +trainer.skip_logprobs=$trainer_skip_logprobs \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    trainer.resume_mode="${trainer_resume_mode}" \
    trainer.resume_from_path="${trainer_resume_path}" \
    trainer.default_local_dir="${checkpoint_dir}" \
    trainer.project_name="$project_name" \
    trainer.logger='["console", "file"]' \
    trainer.val_before_train=${trainer_val_before_train} \
    trainer.n_gpus_per_node=${trainer_n_gpus_per_node} \
    trainer.nnodes=${trainer_nnodes} \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.val_only=${trainer_only_evaluate} \
    +trainer.remove_previous_ckpt_in_save=${trainer_remove_previous_ckpt_in_save} \
    trainer.total_epochs=$total_epochs \
    trainer.experiment_name=$experiment_name \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=false \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=2048 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
    +trainer.only_first_last_checkpoint=${trainer_only_first_last_checkpoint} \
    +reward_model.reward_kwargs.max_resp_len=3072

echo "========================================================"
echo "Training completed with exit code: $?"
echo "End time: $(date)"
echo "Training job finished."
echo "========================================================"
