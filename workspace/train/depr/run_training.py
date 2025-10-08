import os
import subprocess
import argparse
from typing import *

def main(model: str, 
         entropy_coef: float, 
         kl_loss_coef: float, 
         project_name: str, 
         identifier: str = None, 
         mask_positive_entropy_change: bool = False, 
         mask_negative_entropy_change: bool = False,
         advantage_schedule : Union[None, str] = None,
         randomize_zero_std_groups : bool = False,
         grpo_group_size : int = 8,
         tp_size : int = 4,
         cont_flag : bool = False,
         dapo : bool = False):
    """
    Launch a SLURM training job with the specified parameters.
    """
    # Set default paths
    train_files = "/u/rfechner/data/math/train.parquet"
    
    # Convert paths to absolute and check validity
    abs_train_files = os.path.abspath(train_files)
    
    if not os.path.isfile(abs_train_files):
        raise FileNotFoundError(f"Training dataset file not found: {abs_train_files}")
    
    if (advantage_schedule and mask_negative_entropy_change) or \
        (advantage_schedule and mask_positive_entropy_change):
        raise ValueError("Cannot scale advantage AND mask advantages in the same run. Parameterization error.")
    if mask_positive_entropy_change and mask_negative_entropy_change:
        raise ValueError("Cannot mask positive and negative entropy change at the same time. Parameterization error.")

    # Build experiment name with optional identifier (mimic shell script logic)
    if identifier:
        expname = f"{model}_{identifier}_entropy_{entropy_coef}_kl_{kl_loss_coef}"
    else:
        expname = f"{model}_entropy_{entropy_coef}_kl_{kl_loss_coef}"

    ray_data_home = os.environ.get("RAY_DATA_HOME", "/u/rfechner")
    checkpoint_dir = os.path.join(ray_data_home, "out", project_name, expname)

    # Check whether the checkpoint dir already exists. If so, check whether the flag for continuous training is set.
    if os.path.isdir(checkpoint_dir) and not cont_flag:
        print(f"{checkpoint_dir} does already exist and the --cont flag to signal continuous training wasn't set. Terminating.")
        raise SystemExit(1)

    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Training files: {abs_train_files}")
    print(f"Model: {model}")
    print(f"Entropy coefficient: {entropy_coef}")
    print(f"KL loss coefficient: {kl_loss_coef}")
    print(f"Project name: {project_name}")
    
    if dapo:
        print("Running DAPO.")
        entrypoint_script = "verl/workspace/train/dapo_entrypoint.sh"
    else:
        entrypoint_script = "verl/workspace/train/train_entrypoint.sh"

    # Print configuration
    if identifier:
        print(f"Identifier: {identifier}")
    print("=" * 60)

    export_vars = [
        f"TENSOR_PARALLEL_SIZE={tp_size}",
        f"TRAIN_FILES={abs_train_files}",
        f"MODEL_PATH={model}",
        f"ENTROPY_COEF={entropy_coef}",
        f"KL_LOSS_COEF={kl_loss_coef}",
        f"PROJECT_NAME={project_name}",
        f"IDENTIFIER={identifier if identifier else ''}",
        f"MASK_POSITIVE_ENTROPY_CHANGE={'1' if mask_positive_entropy_change else '0'}",
        f"MASK_NEGATIVE_ENTROPY_CHANGE={'1' if mask_negative_entropy_change else '0'}",
        f"SCALE_ADVANTAGE_BY_ENTROPY_CHANGE={'1' if advantage_schedule else '0'}",
        f"ADVANTAGE_SCHEDULE={advantage_schedule}",
        f"RANDOMIZE_ZERO_STD_GROUPS={'1' if randomize_zero_std_groups else '0'}",
        f"GRPO_GROUP_SIZE={grpo_group_size}",
        f"CHECKPOINT_DIR={checkpoint_dir}"
    ]

    sbatch_cmd = [
        "sbatch",
        f"--export={','.join(export_vars)}",
        os.path.abspath(entrypoint_script)
    ]
    
    try:
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
        print(f"Job submitted successfully!")
        print(f"SLURM output: {result.stdout.strip()}")
        if result.stderr:
            print(f"SLURM stderr: {result.stderr.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch SLURM training job for VERL PPO training.")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B",
                       help="Model identifier/path for training (default: Qwen/Qwen2.5-7B).")
    
    # Training hyperparameters
    parser.add_argument("--entropy-coef", type=float, default=0.0,
                       help="Entropy coefficient for training (default: 0.001).")
    parser.add_argument("--kl-loss-coef", type=float, default=0.001,
                       help="KL loss coefficient (default: 0.001).")
    
    # Project configuration
    parser.add_argument("--project-name", type=str, default="entropy_training_runs",
                       help="Project name for logging and checkpointing (default: verl_training).")
    
    # Compute configuration
    # parser.add_argument("--nodes", type=int, default=2, choices=[2, 4],
    #                    help="Number of nodes to use for training (default: 2, choices: 2 or 4).")
    
    # Optional identifier for run tracking
    parser.add_argument("--identifier", type=str, default=None,
                       help="Optional string identifier to help identify the training run (e.g., 'baseline', 'high_entropy', etc.).")
        # Entropy change options
    parser.add_argument("--mask-pos", action="store_true", default=False,
                       help="Enable positive entropy change masking (default: False).")
    parser.add_argument("--mask-neg", action="store_true", default=False,
                       help="Enable negative entropy change masking (default: False).")
    parser.add_argument("--adv-schedule", type=str, default=None, choices=['linear', 'linear_to_balanced', 'balanced'],
                       help="Advantage scheduling based on predicted entropy change.")
    parser.add_argument("--rnd", action="store_true", default=False,
                       help="Whether to assign random rewards to all-zero or all-one reward groups.")
    parser.add_argument("--tp", type=int, default=4,
                       help="Tensor parallel size. Number of GPUs to split the model into. Increasing this number will reduce Maximum GPU memory utilization but slow down training.")
    parser.add_argument("--n", type=int, default=8,
                       help="GRPO group size. Increases peak memory consumption drastically.")
    parser.add_argument("--cont", action="store_true", default=False, help="Whether this training run continues a previous training run. In case this flag isn't set and all hyperparameterizations, i.e. identifier, project name, entropy, KL, ... are the same, a downstream script throws.")
    parser.add_argument("--dapo", action='store_true', default=False, help="Run the DAPO algorithm.")

    args = parser.parse_args()

    # Change to home directory for consistent paths
    os.chdir(os.getenv("HOME"))
    main(
        model=args.model,
        entropy_coef=args.entropy_coef,
        kl_loss_coef=args.kl_loss_coef,
        project_name=args.project_name,
        identifier=args.identifier,
        mask_positive_entropy_change=args.mask_pos,
        mask_negative_entropy_change=args.mask_neg,
        advantage_schedule=args.adv_schedule,
        randomize_zero_std_groups=args.rnd,
        grpo_group_size=args.n,
        tp_size=args.tp,
        cont_flag=args.cont,
        dapo=args.dapo
    )
