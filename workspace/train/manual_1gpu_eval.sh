#!/bin/bash
# Manual single-node launcher for GRPO (4 GPUs) — run directly on a node (no sbatch)
# Hardcoded defaults taken from run.py

# vLLM + FLashinfer and multinode recommended flags
export NCCL_CUMEM_ENABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_LEVEL=SYS
export NCCL_LAUNCH_MODE=GROUP
export CUDA_DEVICE_MAX_CONNECTIONS=32
export VLLM_USE_FLASHINFER=1
export FLASHINFER_ENABLE=1

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

# Activate your environment
conda activate verl

# logging into huggingface
python -c "from huggingface_hub import login; login(token=open('$HOME/.cache/huggingface/token').read().strip())"

math500=/u/rfechner/data/math500/test.parquet
aime25=/u/rfechner/data/aime25/test.parquet
brumo2025=/u/rfechner/data/brumo_2025/test.parquet
cmimc2025=/u/rfechner/data/cmimc_2025/test.parquet
hmmt2025=/u/rfechner/data/hmmt_feb_2025/test.parquet
test_files="['$math500', '$aime25', '$brumo2025', '$cmimc2025', '$hmmt2025']"
project_name="debug-$(date +'%Y-%m-%d-%H-%M-%S')"
experiment_name="llama_3.2_1b_instruct__grpo"
#single_logprob_file="/u/rfechner/data/eic_gsm8k_generated/delta.jsonl" #"/u/rfechner/verl/workspace/chats.jsonl"
logprob_dir=false #"/ptmp/rfechner/out/manual_1gpu/llama_3.2_1b_instruct__grpo/val_jsonl/"
reasoning_strategies="/u/rfechner/data/incomplete_reasoning_strategies/llama-deltas.jsonl"

echo "Logged into huggingface"
CHECKPOINT_DIR="/ptmp/rfechner/out/${project_name}/${experiment_name}"

export VERL_FILE_LOGGER_ROOT="/ptmp/rfechner/out"
python -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.gtpo=false \
    +algorithm.gtpo-alpha=0.1 \
    data.train_files="/u/rfechner/data/math/train.parquet" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.truncation=left \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.use_fused_kernels=true \
    actor_rollout_ref.model.path="meta-llama/Llama-3.2-1B-Instruct" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((2 * (1024 + 3072))) \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.entropy_checkpointing=true \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=true \
    actor_rollout_ref.ref.strategy="fsdp2" \
    actor_rollout_ref.actor.optim.lr=0.000001 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((1 * (1024 + 3072))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((1 * (1024 + 3072))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((6 * (1024 + 3072))) \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.disable_log_stats=false \
    +trainer.validation_data_dir="${CHECKPOINT_DIR}/val_jsonl" \
    +trainer.compute_logprob_from_file=$reasoning_strategies \
    +trainer.compute_logprob_from_rollout_dir=$logprob_dir \
    +trainer.compute_logprob_batch_size=64 \
    +trainer.skip_logprobs=false \
    +trainer.skip_validation=true \
    trainer.val_only=true \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.project_name=${project_name} \
    trainer.logger='["console", "file"]' \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    +trainer.remove_previous_ckpt_in_save=false \
    trainer.total_epochs=2 \
    trainer.experiment_name=${experiment_name}