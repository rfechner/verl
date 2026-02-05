#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/exp15_$(date +'%Y%m%d_%H%M%S').out"

# Combine KL-Cov, Pass@k training

{
    set -x    
    python verl/workspace/train/run.py --project-name exp05_train_qwen2.5-7b --method klcov_passk --model qwen2.5-large   --tp 4 --valn 8 --no-logprobs --cont --epochs 3 --savefreq 10 --testfreq 5
    set +x
} 2>&1 | tee $LOGFILE

