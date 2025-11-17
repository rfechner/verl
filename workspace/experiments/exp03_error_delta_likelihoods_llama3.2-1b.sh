#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/dryrun_$(date +'%Y%m%d_%H%M%S').out"
dataset="/u/rfechner/data/eic_gsm8k_generated/delta.jsonl"

{
    set -x
    python verl/workspace/train/run.py --project-name exp03 --method grpo --model llama3-small --eval --cpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__grpo" --no-validation --lpfile "${dataset}"
    python verl/workspace/train/run.py --project-name exp03 --method grpo-s --model llama3-small --eval --cpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__grpo-s" --no-validation --lpfile "${dataset}"
    python verl/workspace/train/run.py --project-name exp03 --method kl-cov --model llama3-small --eval --cpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__kl-cov" --no-validation --lpfile "${dataset}"
    set +x
} 2>&1 | tee $LOGFILE