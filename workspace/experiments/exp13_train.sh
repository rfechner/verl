#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/exp13_$(date +'%Y%m%d_%H%M%S').out"

# Training script for the final experiment runs.
# We're running Algorithms (DAPO, KL-Cov, GRPO-S, GSPO, Pass@k-training)  x (Qwen2.5-7B, Llama3.1-8B, Olmo-3-7B-Instruct-SFT, EuroLLM-9B)
# This script trains Qwen2.5-7B models with different methods on dapo17k.

{
    set -x
    # Llama3-8B-Instruct
    python verl/workspace/train/run.py --project-name exp13_checkpoints --method gspo          --model llama3-large   --tp 4 --valn 8 --no-logprobs --cont --epochs 3 --savefreq 10 --testfreq 5
    python verl/workspace/train/run.py --project-name exp13_checkpoints --method grpo-passk    --model llama3-large   --tp 4 --valn 8 --no-logprobs --cont --epochs 3 --savefreq 10 --testfreq 5
    python verl/workspace/train/run.py --project-name exp13_checkpoints --method kl-cov        --model llama3-large   --tp 4 --valn 8 --no-logprobs --cont --epochs 3 --savefreq 10 --testfreq 5
    python verl/workspace/train/run.py --project-name exp13_checkpoints --method grpo-s        --model llama3-large   --tp 4 --valn 8 --no-logprobs --cont --epochs 3 --savefreq 10 --testfreq 5
    python verl/workspace/train/run.py --project-name exp13_checkpoints --method grpo          --model llama3-large   --tp 4 --valn 8 --no-logprobs --cont --epochs 3 --savefreq 10 --testfreq 5

    # utter-project/EuroLLM-9B-Instruct
    python verl/workspace/train/run.py --project-name exp13_checkpoints --method gspo          --model eurollm   --tp 4 --valn 8 --no-logprobs --cont --epochs 3 --savefreq 10 --testfreq 5
    python verl/workspace/train/run.py --project-name exp13_checkpoints --method grpo-passk    --model eurollm   --tp 4 --valn 8 --no-logprobs --cont --epochs 3 --savefreq 10 --testfreq 5
    python verl/workspace/train/run.py --project-name exp13_checkpoints --method kl-cov        --model eurollm   --tp 4 --valn 8 --no-logprobs --cont --epochs 3 --savefreq 10 --testfreq 5
    python verl/workspace/train/run.py --project-name exp13_checkpoints --method grpo-s        --model eurollm   --tp 4 --valn 8 --no-logprobs --cont --epochs 3 --savefreq 10 --testfreq 5
    python verl/workspace/train/run.py --project-name exp13_checkpoints --method grpo          --model eurollm   --tp 4 --valn 8 --no-logprobs --cont --epochs 3 --savefreq 10 --testfreq 5

    # Qwen/Qwen2.5-1.5B
    python verl/workspace/train/run.py --project-name exp13_checkpoints --method gspo          --model qwen2.5-medium   --tp 4 --valn 8 --no-logprobs --cont --epochs 3 --savefreq 10 --testfreq 5
    python verl/workspace/train/run.py --project-name exp13_checkpoints --method grpo-passk    --model qwen2.5-medium   --tp 4 --valn 8 --no-logprobs --cont --epochs 3 --savefreq 10 --testfreq 5
    python verl/workspace/train/run.py --project-name exp13_checkpoints --method kl-cov        --model qwen2.5-medium   --tp 4 --valn 8 --no-logprobs --cont --epochs 3 --savefreq 10 --testfreq 5
    python verl/workspace/train/run.py --project-name exp13_checkpoints --method grpo-s        --model qwen2.5-medium   --tp 4 --valn 8 --no-logprobs --cont --epochs 3 --savefreq 10 --testfreq 5
    python verl/workspace/train/run.py --project-name exp13_checkpoints --method grpo          --model qwen2.5-medium   --tp 4 --valn 8 --no-logprobs --cont --epochs 3 --savefreq 10 --testfreq 5

    # Rest of the Methods for Qwen2.5-7B which we didn't run yet.
    python verl/workspace/train/run.py --project-name exp05_train_qwen2.5-7b --method grpo-passk    --model qwen2.5-large   --tp 4 --valn 8 --no-logprobs --cont --epochs 3 --savefreq 10 --testfreq 5

    set +x
} 2>&1 | tee $LOGFILE

