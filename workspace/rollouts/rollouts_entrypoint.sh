#!/bin/bash
#SBATCH --job-name=rollouts
#SBATCH --output=/u/rfechner/jobs/rollouts_%j.out   # Standard output file (%j expands to job ID)
#SBATCH --error=/u/rfechner/jobs/rollouts_%j.err    # Standard error file (%j expands to job ID)
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

echo "========================================================"
echo "Starting VERL Generation Job: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated GPUs: $SLURM_JOB_GPUS"
echo "========================================================"

## Router is expected to export and validate all needed environment variables.
## This script will use the environment variables provided by the router. Do not set defaults here.

echo "Configuration:"
echo "  Identifier: $identifier"
echo "  Model: $model_path"
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
# 3) Start Generation
# ─────────────────────────────────────────────────────────────────────────────
python3 -m verl.trainer.main_generation \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=4 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=-1 \
    rollout.top_p=0.7 \
    rollout.prompt_length=2048 \
    rollout.response_length=3072 \
    rollout.tensor_model_parallel_size=4 \
    rollout.gpu_memory_utilization=0.6 \
    rollout.log_prob_micro_batch_size_per_gpu=16

echo "========================================================"
echo "Training completed with exit code: $?"
echo "End time: $(date)"
echo "Training job finished."
echo "========================================================"
