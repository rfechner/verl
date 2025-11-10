"""
Simple checkpoint validator for VERL rollouts.
Validates that checkpoint sharding matches the target rollout configuration.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any

def handle_model_method_validation(checkpoint_dir : Path | str, method : str, model : str):
    base_model_name = os.path.basename(model).lower().replace("-", "_")
    expname = f"{base_model_name}__{method}"
    if isinstance(checkpoint_dir, Path):
        checkpoint_dir = str(checkpoint_dir)

    assert expname in checkpoint_dir, \
        f"Trying to load \ncheckpoint: {checkpoint_dir}, \nmethod: {method},\nmodel: {model}.\n=> Failed Match. Experiment setup error."

def detect_checkpoint_sharding(checkpoint_dir: str) -> int:
    """Detect how many GPUs the checkpoint was sharded across."""
    actor_dir = os.path.join(checkpoint_dir, 'actor')
    assert os.path.isdir(actor_dir)
    files = os.listdir(actor_dir)
    if "fsdp_config.json" in files:
        with open(os.path.join(actor_dir, 'fsdp_config.json') , 'r') as jsonfile:
            world_size = int(json.load(jsonfile)['world_size'])
            return world_size
    else: # fallback detection of world size.
        for filename in os.listdir(actor_dir):
            if filename.startswith('model_world_size_') and filename.endswith('.pt'):
                parts = filename.split('_')
                if len(parts) >= 5:
                    return int(parts[3])
    raise ValueError(f"Could not detect checkpoint sharding in {actor_dir}")


def validate_checkpoint_path(checkpoint: str) -> str:
    """Validate and convert checkpoint path to absolute path."""
    if not checkpoint:
        return None
        
    abs_checkpoint = os.path.abspath(checkpoint)
    if not (os.path.basename(abs_checkpoint).startswith('global_step') and os.path.isdir(abs_checkpoint)):
        raise ValueError(
            "Please pass valid '.../global_step' directory."
        )
    return abs_checkpoint


def get_checkpoint_gpu_count(checkpoint_dir: str) -> int:
    """Get the number of GPUs that a checkpoint was trained with."""
    if not checkpoint_dir:
        return 4  # Default to 4 GPUs if no checkpoint
    
    abs_checkpoint = validate_checkpoint_path(checkpoint_dir)
    return detect_checkpoint_sharding(abs_checkpoint)


def handle_checkpoint_validation(checkpoint_dir: str, target_gpus: int):
    """
    Validate checkpoint sharding matches target configuration.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
        target_gpus: Number of target GPUs for rollouts
        
    Returns:
        str: Validated absolute checkpoint path
        
    Raises:
        ValueError: If checkpoint validation fails or sharding mismatch detected
    """
    if not checkpoint_dir:
        return None
    
    # Validate checkpoint path
    abs_checkpoint = validate_checkpoint_path(checkpoint_dir)
    
    # Get checkpoint world size
    checkpoint_gpus = detect_checkpoint_sharding(abs_checkpoint)
    
    # Validate sharding matches target - throw error if mismatch
    if checkpoint_gpus != target_gpus:
        raise ValueError(
            f"Checkpoint world size mismatch: checkpoint was saved with {checkpoint_gpus} GPUs "
            f"but target configuration requires {target_gpus} GPUs. "
            f"Please retrain the model with the correct world size ({target_gpus} GPUs) "
            f"or adjust the configuration to match the checkpoint ({checkpoint_gpus} GPUs)."
        )
    
    print(f"Checkpoint sharding validated ({checkpoint_gpus} GPUs matches target)")