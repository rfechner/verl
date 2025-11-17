#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/dryrun_$(date +'%Y%m%d_%H%M%S').out"

# In this experiment we're evaluating the likelihoods of rollouts of checkpoint i under checkpoint j for a subset of methods from the dryrun experiments.
# Hope is to find a method in which the rollouts of the final model are very unlikely under the base model. This would signal that a unique and very unlikely
# method is found.
{
    set -x
    python verl/workspace/train/run.py --project-name exp04 --method grpo --model llama3-small --eval --cpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__grpo" --no-validation --lpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__grpo/val_jsonl"
    python verl/workspace/train/run.py --project-name exp04 --method kl-cov --model llama3-small --eval --cpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__kl-cov" --no-validation --lpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__kl-cov/val_jsonl"
    set +x
} 2>&1 | tee $LOGFILE
