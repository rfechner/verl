#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/exp16_$(date +'%Y%m%d_%H%M%S').out"

# Verification frequency LLMJ
{
    set -x    
    python verl/workspace/rollouts/run.py --model Qwen/Qwen3-8B --data /u/rfechner/data/verified_at_k/base_klcov_64samples_chats.parquet    --out /u/rfechner/data/verified_at_k/base_klcov_64samples_chats-out.parquet
    set +x
} 2>&1 | tee $LOGFILE

