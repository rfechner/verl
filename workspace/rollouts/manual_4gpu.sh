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


# ─────────────────────────────────────────────────────────────────────────────
# 3) Start Generation
# ─────────────────────────────────────────────────────────────────────────────

data_path=$HOME/data/ariadne/prompts.parquet
save_path=$HOME/data/ariadne/outputs.parquet
model_path=Qwen/Qwen2.5-7B

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    data.batch_size=128 \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=-1 \
    rollout.top_p=0.7 \
    rollout.prompt_length=2048 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=4 \
    rollout.gpu_memory_utilization=0.3 \
    rollout.log_prob_micro_batch_size_per_gpu=8 
echo "========================================================"
echo "Training completed with exit code: $?"
echo "End time: $(date)"
echo "Training job finished."
echo "========================================================"
