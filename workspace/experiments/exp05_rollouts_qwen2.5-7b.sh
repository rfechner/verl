#!/bin/bash
cd /u/rfechner
python verl/workspace/train/run.py --project-name exp05_rollouts_qwen2.5-7b --method grpo      --model qwen2.5-large   --tp 4 --flashinfer --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo"
python verl/workspace/train/run.py --project-name exp05_rollouts_qwen2.5-7b --method dapo      --model qwen2.5-large   --tp 4 --flashinfer --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__dapo"
python verl/workspace/train/run.py --project-name exp05_rollouts_qwen2.5-7b --method kl-cov    --model qwen2.5-large   --tp 4 --flashinfer --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__kl-cov"
python verl/workspace/train/run.py --project-name exp05_rollouts_qwen2.5-7b --method clip-cov  --model qwen2.5-large   --tp 4 --flashinfer --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__clip-cov"
python verl/workspace/train/run.py --project-name exp05_rollouts_qwen2.5-7b --method gtpo      --model qwen2.5-large   --tp 4 --flashinfer --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__gtpo"
python verl/workspace/train/run.py --project-name exp05_rollouts_qwen2.5-7b --method grpo-s    --model qwen2.5-large   --tp 4 --flashinfer --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo-s"
