#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/dryrun_$(date +'%Y%m%d_%H%M%S').out"

{
    set -x
    python verl/workspace/train/run.py --project-name exp04 --method grpo --model llama3-small --eval --cpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__grpo" --no-validation --lpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__grpo/val_jsonl"
    python verl/workspace/train/run.py --project-name exp04 --method kl-cov --model llama3-small --eval --cpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__kl-cov" --no-validation --lpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__kl-cov/val_jsonl"
    set +x
} 2>&1 | tee $LOGFILE
