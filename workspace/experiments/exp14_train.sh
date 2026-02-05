#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/exp14_$(date +'%Y%m%d_%H%M%S').out"

# In this experiment we're training two additional setups with different random seeds

{
    set -x
    
    python verl/workspace/train/run.py --project-name exp05_train_qwen2.5-7b --method kl-cov --identifier seed_2   --seed 2 --model qwen2.5-large   --tp 4 --valn 8 --no-logprobs --cont --epochs 3 --savefreq 10 --testfreq 5
    python verl/workspace/train/run.py --project-name exp05_train_qwen2.5-7b --method kl-cov --identifier seed_3   --seed 3 --model qwen2.5-large   --tp 4 --valn 8 --no-logprobs --cont --epochs 3 --savefreq 10 --testfreq 5

    set +x
} 2>&1 | tee $LOGFILE

