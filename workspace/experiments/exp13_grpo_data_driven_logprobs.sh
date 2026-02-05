#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/exp13_data_driven_$(date +'%Y%m%d_%H%M%S').out"

# Data-Driven Log-prob analysis.
{
    set -x

    # Qw2.5-7B
    python verl/workspace/train/run.py --only-first-last-checkpoint --cont --project-name exp13_data_driven --method grpo     --model qwen2.5-large   --tp 4  --no-validation --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo"     --lpdir "/ptmp/rfechner/out/exp13_rollouts_subsampled/qwen2.5_7b__grpo/val_jsonl"

    # EuroLLM-9B
    python verl/workspace/train/run.py --only-first-last-checkpoint --cont --project-name exp13_data_driven --method grpo        --model eurollm   --tp 4  --no-validation --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/eurollm_9b_instruct__grpo"       --lpdir "/ptmp/rfechner/out/exp13_rollouts_subsampled/eurollm_9b_instruct__grpo/val_jsonl"

    # Llama3.1-8B
    python verl/workspace/train/run.py --only-first-last-checkpoint --cont --project-name exp13_data_driven --method grpo        --model llama3-large   --tp 4  --no-validation --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/llama_3.1_8b_instruct__grpo"        --lpdir "/ptmp/rfechner/out/exp13_rollouts_subsampled/llama_3.1_8b_instruct__grpo/val_jsonl"

    # Qwen2.5-1.5B
    python verl/workspace/train/run.py --only-first-last-checkpoint --cont --project-name exp13_data_driven --method grpo        --model qwen2.5-medium   --tp 4  --no-validation --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/qwen2.5_1.5b__grpo" --lpdir "/ptmp/rfechner/out/exp13_rollouts_subsampled/qwen2.5_1.5b__grpo/val_jsonl"

    set +x
} 2>&1 | tee $LOGFILE

