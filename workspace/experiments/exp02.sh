#!/bin/bash
cd /u/rfechner

# Full-training comparison. Do we see differences in DAPO/GRPO?
python verl/workspace/train/run.py --project-name exp02 --method grpo --model qwen2.5-large  --flashinfer --valn 1 --no-logprobs --no-rollouts --epochs 30 --savefreq 50 --testfreq 10 --cont
python verl/workspace/train/run.py --project-name exp02 --method dapo --model qwen2.5-large  --flashinfer --valn 1 --no-logprobs --no-rollouts --epochs 30 --savefreq 50 --testfreq 10 --cont