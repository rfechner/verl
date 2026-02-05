#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/exp17_rollouts_$(date +'%Y%m%d_%H%M%S').out"

{
    set -x
    python verl/workspace/train/run.py --cont --project-name exp17_rollouts --method kl-cov --model qwen2.5-large   --tp 4  --valn 256 --val_batchsize 32 --no-logprobs --eval --cp "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/seed_2__qwen2.5_7b__kl-cov/global_step_30"
    python verl/workspace/train/run.py --cont --project-name exp17_rollouts --method kl-cov --model qwen2.5-large   --tp 4  --valn 256 --val_batchsize 32 --no-logprobs --eval --cp "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/seed_3__qwen2.5_7b__kl-cov/global_step_30"
    python verl/workspace/train/run.py --cont --project-name exp17_rollouts --method klcov_passk --model qwen2.5-large   --tp 4  --valn 256 --val_batchsize 32 --no-logprobs --eval --cp "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__klcov_passk/global_step_30"
    set +x
} 2>&1 | tee $LOGFILE

