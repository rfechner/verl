#!/bin/bash
cd /u/rfechner
# This experiment run aims to train the smallest models on the full experimentation pipeline.
# Experiment settings:
# Training Dataset: Dapo17k - english, Validation Sets: Math500, MathArena Datasets
# Methods: GRPO, Dr GRPO, GSPO, DAPO
# Models: qwen3-small, llama3-small
python verl/workspace/train/run.py --project-name exp01 --method grpo    --model qwen3-small  --tp 2 --flashinfer --valn 256
python verl/workspace/train/run.py --project-name exp01 --method grpo    --model llama3-small --tp 2 --flashinfer --valn 256
python verl/workspace/train/run.py --project-name exp01 --method drgrpo  --model qwen3-small  --tp 2 --flashinfer --valn 256
python verl/workspace/train/run.py --project-name exp01 --method drgrpo  --model llama3-small --tp 2 --flashinfer --valn 256
python verl/workspace/train/run.py --project-name exp01 --method dapo    --model qwen3-small  --tp 2 --flashinfer --valn 256
python verl/workspace/train/run.py --project-name exp01 --method dapo    --model llama3-small --tp 2 --flashinfer --valn 256
python verl/workspace/train/run.py --project-name exp01 --method gspo    --model qwen3-small  --tp 2 --flashinfer --valn 256
python verl/workspace/train/run.py --project-name exp01 --method gspo    --model llama3-small --tp 2 --flashinfer --valn 256
