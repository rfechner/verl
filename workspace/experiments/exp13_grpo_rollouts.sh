#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/exp13_grpo_rollouts_$(date +'%Y%m%d_%H%M%S').out"

# Rollout script for the final experiment runs.

{
    set -x

    # EuroLLM-9B
    python verl/workspace/train/run.py --cont --project-name exp13_rollouts --method grpo --model eurollm           --tp 4  --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/eurollm_9b_instruct__grpo"
    
    # Llama3.1-8B
    python verl/workspace/train/run.py --cont --project-name exp13_rollouts --method grpo --model llama3-large      --tp 4  --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/llama_3.1_8b_instruct__grpo"
    
    # Qwen2.5-1.5B
    python verl/workspace/train/run.py --cont --project-name exp13_rollouts --method grpo --model qwen2.5-medium    --tp 4  --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/qwen2.5_1.5b__grpo"

    set +x
} 2>&1 | tee $LOGFILE

