#!/bin/bash
cd /u/rfechner
math500_path=/u/rfechner/data/math500/test.parquet
val_files=$math500_path

python verl/workspace/train/run.py --project-name exp07_rollouts --method grpo      --model gpt-oss            --tp 4 --flashinfer --valn 16 --val_batchsize 512 --no-logprobs --eval --val_data "$val_files"
python verl/workspace/train/run.py --project-name exp07_rollouts --method grpo      --model qwen2.5-large      --tp 4 --flashinfer --valn 16 --val_batchsize 512 --no-logprobs --eval --val_data "$val_files"
python verl/workspace/train/run.py --project-name exp07_rollouts --method grpo      --model qwen3-large        --tp 4 --flashinfer --valn 16 --val_batchsize 512 --no-logprobs --eval --val_data "$val_files"
python verl/workspace/train/run.py --project-name exp07_rollouts --method grpo      --model llama3-large       --tp 4 --flashinfer --valn 16 --val_batchsize 512 --no-logprobs --eval --val_data "$val_files"


