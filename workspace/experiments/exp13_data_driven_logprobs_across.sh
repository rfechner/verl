#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/exp13_data_driven_across_$(date +'%Y%m%d_%H%M%S').out"

# Data-Driven Log-prob analysis.
{
    set -x

    # Qw2.5-7B
    python verl/workspace/train/run.py  --cont --project-name exp13_data_driven_across --method kl-cov     --model qwen2.5-large   --tp 4  --no-validation --eval --cp "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__kl-cov/global_step_60"     --lpfile "/u/rfechner/data/data_driven_subset/rollouts.jsonl"
    python verl/workspace/train/run.py  --cont --project-name exp13_data_driven_across --method grpo-s     --model qwen2.5-large   --tp 4  --no-validation --eval --cp "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo-s/global_step_70"     --lpfile "/u/rfechner/data/data_driven_subset/rollouts.jsonl"
    python verl/workspace/train/run.py  --cont --project-name exp13_data_driven_across --method gspo       --model qwen2.5-large   --tp 4  --no-validation --eval --cp "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__gspo/global_step_80"       --lpfile "/u/rfechner/data/data_driven_subset/rollouts.jsonl"
    python verl/workspace/train/run.py  --cont --project-name exp13_data_driven_across --method grpo-passk --model qwen2.5-large   --tp 4  --no-validation --eval --cp "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo-passk/global_step_80" --lpfile "/u/rfechner/data/data_driven_subset/rollouts.jsonl"

    # EuroLLM-9B
    python verl/workspace/train/run.py  --cont --project-name exp13_data_driven_across --method kl-cov        --model eurollm   --tp 4  --no-validation --eval --cp "/ptmp/rfechner/out/exp13_checkpoints/eurollm_9b_instruct__kl-cov/global_step_60"       --lpfile "/u/rfechner/data/data_driven_subset/rollouts.jsonl"
    python verl/workspace/train/run.py  --cont --project-name exp13_data_driven_across --method grpo-s        --model eurollm   --tp 4  --no-validation --eval --cp "/ptmp/rfechner/out/exp13_checkpoints/eurollm_9b_instruct__grpo-s/global_step_80"       --lpfile "/u/rfechner/data/data_driven_subset/rollouts.jsonl"
    python verl/workspace/train/run.py  --cont --project-name exp13_data_driven_across --method gspo          --model eurollm   --tp 4  --no-validation --eval --cp "/ptmp/rfechner/out/exp13_checkpoints/eurollm_9b_instruct__gspo/global_step_80"         --lpfile "/u/rfechner/data/data_driven_subset/rollouts.jsonl"
    python verl/workspace/train/run.py  --cont --project-name exp13_data_driven_across --method grpo-passk    --model eurollm   --tp 4  --no-validation --eval --cp "/ptmp/rfechner/out/exp13_checkpoints/eurollm_9b_instruct__grpo-passk/global_step_80"   --lpfile "/u/rfechner/data/data_driven_subset/rollouts.jsonl"

    # Llama3.1-8B
    python verl/workspace/train/run.py  --cont --project-name exp13_data_driven_across --method kl-cov        --model llama3-large   --tp 4  --no-validation --eval --cp "/ptmp/rfechner/out/exp13_checkpoints/llama_3.1_8b_instruct__kl-cov/global_step_70"        --lpfile "/u/rfechner/data/data_driven_subset/rollouts.jsonl"
    python verl/workspace/train/run.py  --cont --project-name exp13_data_driven_across --method grpo-s        --model llama3-large   --tp 4  --no-validation --eval --cp "/ptmp/rfechner/out/exp13_checkpoints/llama_3.1_8b_instruct__grpo-s/global_step_80"        --lpfile "/u/rfechner/data/data_driven_subset/rollouts.jsonl"
    python verl/workspace/train/run.py  --cont --project-name exp13_data_driven_across --method gspo          --model llama3-large   --tp 4  --no-validation --eval --cp "/ptmp/rfechner/out/exp13_checkpoints/llama_3.1_8b_instruct__gspo/global_step_80"          --lpfile "/u/rfechner/data/data_driven_subset/rollouts.jsonl"
    python verl/workspace/train/run.py  --cont --project-name exp13_data_driven_across --method grpo-passk    --model llama3-large   --tp 4  --no-validation --eval --cp "/ptmp/rfechner/out/exp13_checkpoints/llama_3.1_8b_instruct__grpo-passk/global_step_80"    --lpfile "/u/rfechner/data/data_driven_subset/rollouts.jsonl"

    # Qwen2.5-1.5B
    # KL-Cov didn't yield results on small model.
    python verl/workspace/train/run.py  --cont --project-name exp13_data_driven_across --method grpo-s        --model qwen2.5-medium   --tp 4  --no-validation --eval --cp "/ptmp/rfechner/out/exp13_checkpoints/qwen2.5_1.5b__grpo-s/global_step_80" --lpfile "/u/rfechner/data/data_driven_subset/rollouts.jsonl"
    python verl/workspace/train/run.py  --cont --project-name exp13_data_driven_across --method gspo          --model qwen2.5-medium   --tp 4  --no-validation --eval --cp "/ptmp/rfechner/out/exp13_checkpoints/qwen2.5_1.5b__gspo/global_step_80"  --lpfile "/u/rfechner/data/data_driven_subset/rollouts.jsonl"
    python verl/workspace/train/run.py  --cont --project-name exp13_data_driven_across --method grpo-passk    --model qwen2.5-medium   --tp 4  --no-validation --eval --cp "/ptmp/rfechner/out/exp13_checkpoints/qwen2.5_1.5b__grpo-passk//global_step_80" --lpfile "/u/rfechner/data/data_driven_subset/rollouts.jsonl"

    set +x
} 2>&1 | tee $LOGFILE

