#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/exp13_rollouts_$(date +'%Y%m%d_%H%M%S').out"

# Rollout script for the final experiment runs.

{
    set -x

    # Most of Qw2.5-7B rollouts were already done
    # python verl/workspace/train/run.py --project-name exp05_rollouts_qwen2.5-7b --method kl-cov     --model qwen2.5-large   --tp 4  --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__kl-cov"
    # python verl/workspace/train/run.py --project-name exp05_rollouts_qwen2.5-7b --method grpo-s     --model qwen2.5-large   --tp 4  --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo-s"
    # python verl/workspace/train/run.py --project-name exp05_rollouts_qwen2.5-7b --method gspo       --model qwen2.5-large   --tp 4  --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__gspo"
    python verl/workspace/train/run.py --cont --project-name exp05_rollouts_qwen2.5-7b --method grpo-passk --model qwen2.5-large   --tp 4  --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo-passk"

    # EuroLLM-9B
    python verl/workspace/train/run.py --cont --project-name exp13_rollouts --method kl-cov        --model eurollm   --tp 4  --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/eurollm_9b_instruct__kl-cov"
    python verl/workspace/train/run.py --cont --project-name exp13_rollouts --method grpo-s        --model eurollm   --tp 4  --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/eurollm_9b_instruct__grpo-s"
    python verl/workspace/train/run.py --cont --project-name exp13_rollouts --method gspo          --model eurollm   --tp 4  --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/eurollm_9b_instruct__gspo"
    python verl/workspace/train/run.py --cont --project-name exp13_rollouts --method grpo-passk    --model eurollm   --tp 4  --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/eurollm_9b_instruct__grpo-passk"

    # Llama3.1-8B
    python verl/workspace/train/run.py --cont --project-name exp13_rollouts --method kl-cov        --model llama3-large   --tp 4  --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/llama_3.1_8b_instruct__kl-cov"
    python verl/workspace/train/run.py --cont --project-name exp13_rollouts --method grpo-s        --model llama3-large   --tp 4  --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/llama_3.1_8b_instruct__grpo-s"
    python verl/workspace/train/run.py --cont --project-name exp13_rollouts --method gspo          --model llama3-large   --tp 4  --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/llama_3.1_8b_instruct__gspo"
    python verl/workspace/train/run.py --cont --project-name exp13_rollouts --method grpo-passk    --model llama3-large   --tp 4  --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/llama_3.1_8b_instruct__grpo-passk"

    # Qwen2.5-1.5B
    # KL-Cov didn't yield results on small model.
    # python verl/workspace/train/run.py --project-name exp13_rollouts --method kl-cov        --model qwen2.5-medium   --tp 4  --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/qwen2.5_1.5b__kl-cov"
    python verl/workspace/train/run.py --cont --project-name exp13_rollouts --method grpo-s        --model qwen2.5-medium   --tp 4  --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/qwen2.5_1.5b__grpo-s"
    python verl/workspace/train/run.py --cont --project-name exp13_rollouts --method gspo          --model qwen2.5-medium   --tp 4  --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/qwen2.5_1.5b__gspo"
    python verl/workspace/train/run.py --cont --project-name exp13_rollouts --method grpo-passk    --model qwen2.5-medium   --tp 4  --valn 256 --val_batchsize 32 --no-logprobs --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/qwen2.5_1.5b__grpo-passk"

    set +x
} 2>&1 | tee $LOGFILE

