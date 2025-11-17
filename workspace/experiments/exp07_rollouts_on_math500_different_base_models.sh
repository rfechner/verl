#!/bin/bash
cd /u/rfechner
math500_path=/u/rfechner/data/math500/test.parquet
val_files=$math500_path

# In this experiment I want to generate base rollouts from different models to postprocess further. The idea is to
# take these traces, isolate the correct traces and then per-question decide whether an augmentation (error, reasoning strategy, reasoning type) is applicable.
# from these candidates, we generate augmentations. FInally, we obtain (correct_solution, augmented_trace, base_model) triples for further experiments.
python verl/workspace/train/run.py --project-name exp07_rollouts --method grpo      --model gpt-oss            --tp 4 --flashinfer --valn 16 --val_batchsize 512 --no-logprobs --eval --val_data "$val_files"
python verl/workspace/train/run.py --project-name exp07_rollouts --method grpo      --model qwen2.5-large      --tp 4 --flashinfer --valn 16 --val_batchsize 512 --no-logprobs --eval --val_data "$val_files"
python verl/workspace/train/run.py --project-name exp07_rollouts --method grpo      --model qwen3-large        --tp 4 --flashinfer --valn 16 --val_batchsize 512 --no-logprobs --eval --val_data "$val_files"
python verl/workspace/train/run.py --project-name exp07_rollouts --method grpo      --model llama3-large       --tp 4 --flashinfer --valn 16 --val_batchsize 512 --no-logprobs --eval --val_data "$val_files"


