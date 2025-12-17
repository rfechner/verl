#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/exp11_$(date +'%Y%m%d_%H%M%S').out"

# I've wanted to repeat exp10 with easier data, as i think this may make it easier to augment and judge data.

{
    set -x
    python verl/workspace/train/run.py --project-name exp11_rollouts_qwen2.5-7b --method gspo    --model qwen2.5-large   --tp 4 --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__gspo" --val_data /u/rfechner/data/gsm8k/test.parquet
    set +x
} 2>&1 | tee $LOGFILE

