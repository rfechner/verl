#!/bin/bash
#SBATCH --job-name=verl-train2                              # Descriptive job name
#SBATCH --output=/u/rfechner/jobs/slurm_train2_%j.out   # Standard output file (%j expands to job ID)
#SBATCH --error=/u/rfechner/jobs/slurm_train2_%j.err    # Standard error file (%j expands to job ID)
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

echo "Logging to ${CHECKPOINT_DIR}"

VAL_FILES=${VAL_FILES:-"[/u/rfechner/data/math500/test.parquet]"}

echo "Configuration:"
echo "  Identifier: $IDENTIFIER"
echo "  Train files: $TRAIN_FILES"
echo "  Val files: $VAL_FILES"
echo "  Model: $MODEL_PATH"
echo "  Learning rate: $LEARNING_RATE"
echo "  Entropy coefficient: $ENTROPY_COEF"
echo "  KL loss coefficient: $KL_LOSS_COEF"
echo "  Total epochs: $TOTAL_EPOCHS"
echo "  Project name: $PROJECT_NAME"
echo "  Nodes: $NNODES"
echo "  Model Tensor Parallelism: $TENSOR_PARALLEL_SIZE"
echo "  GRPO group size: $GRPO_GROUP_SIZE"

echo "  ---- FLAGS ----"
echo "  RANDOMIZE_ZERO_STD_GROUPS: $RANDOMIZE_ZERO_STD_GROUPS"
echo "  ADVANTAGE_SCHEDULE: $ADVANTAGE_SCHEDULE"
echo "  MASK_POSITIVE_ENTROPY_CHANGE: $MASK_POSITIVE_ENTROPY_CHANGE"
echo "  MASK_NEGATIVE_ENTROPY_CHANGE: $MASK_NEGATIVE_ENTROPY_CHANGE"

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

# Activate your environment
conda activate verl

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

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
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
    srun --nodes=1 --ntasks=1 -w "$node_i" \
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

python -u -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="$TRAIN_FILES" \
        data.val_files=$VAL_FILES \
        data.train_batch_size=$TRAIN_BATCH_SIZE \
        data.max_prompt_length=$MAX_PROMPT_LENGTH \
        data.max_response_length=$MAX_RESPONSE_LENGTH \
        data.truncation=left \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.path=$MODEL_PATH \
        +actor_rollout_ref.actor.adaptive_scale_by_entropy_change=${SCALE_ADVANTAGE_BY_ENTROPY_CHANGE} \
        +actor_rollout_ref.actor.mask_positive_entropy_change=${MASK_POSITIVE_ENTROPY_CHANGE} \
        +actor_rollout_ref.actor.mask_negative_entropy_change=${MASK_NEGATIVE_ENTROPY_CHANGE} \
        +trainer.randomize_zero_std_groups=${RANDOMIZE_ZERO_STD_GROUPS} \
        +trainer.advantage_schedule=${ADVANTAGE_SCHEDULE} \
        actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((1 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
        actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
        actor_rollout_ref.actor.use_kl_loss=$([ "$KL_LOSS_COEF" -gt 0 ] && echo True || echo False) \
        actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEF \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_PARALLEL_SIZE} \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((6 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
        actor_rollout_ref.rollout.n=${GRPO_GROUP_SIZE} \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((1 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) \
        actor_rollout_ref.rollout.temperature=${temperature} \
        actor_rollout_ref.rollout.top_p=${top_p} \
        actor_rollout_ref.rollout.top_k=${top_k} \
        actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
        actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
        actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.disable_log_stats=False \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.resume_mode=auto \
        trainer.default_local_dir="${CHECKPOINT_DIR}" \
        trainer.project_name="$PROJECT_NAME" \
        trainer.logger=console \
        trainer.val_before_train=False \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=$NNODES \
        trainer.save_freq=$SAVE_FREQ \
        trainer.test_freq=$TEST_FREQ \
        trainer.default_local_dir="${CHECKPOINT_DIR}" \
        trainer.remove_previous_ckpt_in_save=False \
        trainer.total_epochs=$TOTAL_EPOCHS
        

echo "========================================================"
echo "Training completed with exit code: $?"
echo "End time: $(date)"
echo "Training job finished."
echo "========================================================"
