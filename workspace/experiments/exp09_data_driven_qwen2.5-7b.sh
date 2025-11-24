#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/exp09_$(date +'%Y%m%d_%H%M%S').out"

# In this experiment we're evaluating the likelihoods of rollouts of checkpoint i under checkpoint j for a subset of methods from the dryrun experiments.
# Hope is to find a method in which the rollouts of the final model are very unlikely under the base model. This would signal that a unique and very unlikely
# method is found.

# I wanted to repeat the experiment 04 with other model families and model size to see, whether the trends from
# the small llama models hold.
{
    set -x
    python verl/workspace/train/run.py --project-name exp09 --method grpo   --model qwen2.5-large --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo"     --no-validation --lpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo/val_jsonl"
    python verl/workspace/train/run.py --project-name exp09 --method kl-cov --model qwen2.5-large --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__kl-cov"   --no-validation --lpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__kl-cov/val_jsonl"
    python verl/workspace/train/run.py --project-name exp09 --method grpo-s --model qwen2.5-large --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo-s"   --no-validation --lpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo-s/val_jsonl"
    python verl/workspace/train/run.py --project-name exp09 --method dapo   --model qwen2.5-large --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__dapo"     --no-validation --lpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__dapo/val_jsonl"
    set +x
} 2>&1 | tee $LOGFILE

