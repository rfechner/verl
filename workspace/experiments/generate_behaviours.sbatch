#!/bin/bash
#SBATCH --job-name=prompt_gen
#SBATCH --output=/u/rfechner/jobs/prompt_gen_%j.out   # Standard output file (%j expands to job ID)
#SBATCH --error=/u/rfechner/jobs/prompt_gen_%j.err    # Standard error file (%j expands to job ID)
#SBATCH --nodes=1
#SBATCH --partition=general                                 # main partition containing GPU nodes
#SBATCH --constraint=gpu                                    # high bandwidth GPUs within general partition  
#SBATCH --switches=1                                        # constraint on the topology -> we want adjacent nodes
#SBATCH --ntasks-per-node=1                                 # Run the shell script as a single task on this node
#SBATCH --cpus-per-task=64                                  # Allocate all CPUs on the node (Raven GPU nodes often have 72 cores, adjust if different)
#SBATCH --gres=gpu:a100:4                                   
#SBATCH --time=23:59:00                                     # Maximum wall clock time (e.g., 24 hours). Adjust as needed.
#SBATCH --mem=500000                                        # Request all available memory on the node. Can be specific e.g., "200G".
#SBATCH --chdir=./

echo "========================================================"
echo "Starting VERL Training Job: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated GPUs: $SLURM_JOB_GPUS"
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

echo "Activating conda environment: $target_conda_env"
conda activate verl

python verl/workspace/judge/run_generation.py