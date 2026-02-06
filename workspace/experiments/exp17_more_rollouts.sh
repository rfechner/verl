#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/exp17_more_rollouts_$(date +'%Y%m%d_%H%M%S').out"

{
    set -x

    # Basemodel 1024 rollouts
    python verl/workspace/train/run.py --cont --project-name exp17_more_rollouts --identifier basemodel --method gspo --model qwen2.5-large   --tp 4  --valn 1024 --val_batchsize 8 --no-logprobs --eval

    python verl/workspace/train/run.py --cont --project-name exp17_more_rollouts --method klcov_passk   --model qwen2.5-large   --tp 4  --valn 1024 --val_batchsize 8 --no-logprobs --eval --cp "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__klcov_passk/global_step_60"
    python verl/workspace/train/run.py --cont --project-name exp17_more_rollouts --method kl-cov        --model qwen2.5-large   --tp 4  --valn 1024 --val_batchsize 8 --no-logprobs --eval --cp "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__kl-cov/global_step_60"
    python verl/workspace/train/run.py --cont --project-name exp17_more_rollouts --method grpo-passk    --model qwen2.5-large   --tp 4  --valn 1024 --val_batchsize 8 --no-logprobs --eval --cp "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo-passk/global_step_60"
    python verl/workspace/train/run.py --cont --project-name exp17_more_rollouts --method grpo          --model qwen2.5-large   --tp 4  --valn 1024 --val_batchsize 8 --no-logprobs --eval --cp "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo/global_step_60"

    set +x
} 2>&1 | tee $LOGFILE

