#!/bin/bash
cd /u/rfechner

# This script trains Qwen2.5-7B models with different methods on dapo17k.
python verl/workspace/train/run.py --project-name exp05_train_qwen2.5-7b --method grpo      --model qwen2.5-large   --tp 4 --flashinfer --valn 8 --no-logprobs --cont --epochs 30 --savefreq 10 --testfreq 5
python verl/workspace/train/run.py --project-name exp05_train_qwen2.5-7b --method dapo      --model qwen2.5-large   --tp 4 --flashinfer --valn 8 --no-logprobs --cont --epochs 30 --savefreq 10 --testfreq 5
python verl/workspace/train/run.py --project-name exp05_train_qwen2.5-7b --method kl-cov    --model qwen2.5-large   --tp 4 --flashinfer --valn 8 --no-logprobs --cont --epochs 30 --savefreq 10 --testfreq 5
python verl/workspace/train/run.py --project-name exp05_train_qwen2.5-7b --method clip-cov  --model qwen2.5-large   --tp 4 --flashinfer --valn 8 --no-logprobs --cont --epochs 30 --savefreq 10 --testfreq 5
python verl/workspace/train/run.py --project-name exp05_train_qwen2.5-7b --method gtpo      --model qwen2.5-large   --tp 4 --flashinfer --valn 8 --no-logprobs --cont --epochs 30 --savefreq 10 --testfreq 5
python verl/workspace/train/run.py --project-name exp05_train_qwen2.5-7b --method grpo-s    --model qwen2.5-large   --tp 4 --flashinfer --valn 8 --no-logprobs --cont --epochs 30 --savefreq 10 --testfreq 5
