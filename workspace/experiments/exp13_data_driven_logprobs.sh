#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/exp13_data_driven_$(date +'%Y%m%d_%H%M%S').out"

# Data-Driven Log-prob analysis.
{
    set -x

    # Qw2.5-7B
    python verl/workspace/train/run.py --only-first-last-checkpoint --cont --project-name exp13_data_driven --method kl-cov     --model qwen2.5-large   --tp 4  --no-validation --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__kl-cov"     --lpdir "/ptmp/rfechner/out/exp13_rollouts_subsampled/qwen2.5_7b__kl-cov/val_jsonl"
    python verl/workspace/train/run.py --only-first-last-checkpoint --cont --project-name exp13_data_driven --method grpo-s     --model qwen2.5-large   --tp 4  --no-validation --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo-s"     --lpdir "/ptmp/rfechner/out/exp13_rollouts_subsampled/qwen2.5_7b__grpo-s/val_jsonl"
    python verl/workspace/train/run.py --only-first-last-checkpoint --cont --project-name exp13_data_driven --method gspo       --model qwen2.5-large   --tp 4  --no-validation --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__gspo"       --lpdir "/ptmp/rfechner/out/exp13_rollouts_subsampled/qwen2.5_7b__gspo/val_jsonl"
    python verl/workspace/train/run.py --only-first-last-checkpoint --cont --project-name exp13_data_driven --method grpo-passk --model qwen2.5-large   --tp 4  --no-validation --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo-passk" --lpdir "/ptmp/rfechner/out/exp13_rollouts_subsampled/qwen2.5_7b__grpo-passk/val_jsonl"

    # EuroLLM-9B
    python verl/workspace/train/run.py --only-first-last-checkpoint --cont --project-name exp13_data_driven --method kl-cov        --model eurollm   --tp 4  --no-validation --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/eurollm_9b_instruct__kl-cov"       --lpdir "/ptmp/rfechner/out/exp13_rollouts_subsampled/eurollm_9b_instruct__kl-cov/val_jsonl"
    python verl/workspace/train/run.py --only-first-last-checkpoint --cont --project-name exp13_data_driven --method grpo-s        --model eurollm   --tp 4  --no-validation --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/eurollm_9b_instruct__grpo-s"       --lpdir "/ptmp/rfechner/out/exp13_rollouts_subsampled/eurollm_9b_instruct__grpo-s/val_jsonl"
    python verl/workspace/train/run.py --only-first-last-checkpoint --cont --project-name exp13_data_driven --method gspo          --model eurollm   --tp 4  --no-validation --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/eurollm_9b_instruct__gspo"         --lpdir "/ptmp/rfechner/out/exp13_rollouts_subsampled/eurollm_9b_instruct__gspo/val_jsonl"
    python verl/workspace/train/run.py --only-first-last-checkpoint --cont --project-name exp13_data_driven --method grpo-passk    --model eurollm   --tp 4  --no-validation --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/eurollm_9b_instruct__grpo-passk"   --lpdir "/ptmp/rfechner/out/exp13_rollouts_subsampled/eurollm_9b_instruct__grpo-passk/val_jsonl"

    # Llama3.1-8B
    python verl/workspace/train/run.py --only-first-last-checkpoint --cont --project-name exp13_data_driven --method kl-cov        --model llama3-large   --tp 4  --no-validation --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/llama_3.1_8b_instruct__kl-cov"        --lpdir "/ptmp/rfechner/out/exp13_rollouts_subsampled/llama_3.1_8b_instruct__kl-cov/val_jsonl"
    python verl/workspace/train/run.py --only-first-last-checkpoint --cont --project-name exp13_data_driven --method grpo-s        --model llama3-large   --tp 4  --no-validation --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/llama_3.1_8b_instruct__grpo-s"        --lpdir "/ptmp/rfechner/out/exp13_rollouts_subsampled/llama_3.1_8b_instruct__grpo-s/val_jsonl"
    python verl/workspace/train/run.py --only-first-last-checkpoint --cont --project-name exp13_data_driven --method gspo          --model llama3-large   --tp 4  --no-validation --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/llama_3.1_8b_instruct__gspo"          --lpdir "/ptmp/rfechner/out/exp13_rollouts_subsampled/llama_3.1_8b_instruct__gspo"/val_jsonl
    python verl/workspace/train/run.py --only-first-last-checkpoint --cont --project-name exp13_data_driven --method grpo-passk    --model llama3-large   --tp 4  --no-validation --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/llama_3.1_8b_instruct__grpo-passk"    --lpdir "/ptmp/rfechner/out/exp13_rollouts_subsampled/llama_3.1_8b_instruct__grpo-passk/val_jsonl"

    # Qwen2.5-1.5B
    # KL-Cov didn't yield results on small model.
    python verl/workspace/train/run.py --only-first-last-checkpoint --cont --project-name exp13_data_driven --method grpo-s        --model qwen2.5-medium   --tp 4  --no-validation --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/qwen2.5_1.5b__grpo-s" --lpdir "/ptmp/rfechner/out/exp13_rollouts_subsampled/qwen2.5_1.5b__grpo-s/val_jsonl"
    python verl/workspace/train/run.py --only-first-last-checkpoint --cont --project-name exp13_data_driven --method gspo          --model qwen2.5-medium   --tp 4  --no-validation --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/qwen2.5_1.5b__gspo"  --lpdir "/ptmp/rfechner/out/exp13_rollouts_subsampled/qwen2.5_1.5b__gspo/val_jsonl"
    python verl/workspace/train/run.py --only-first-last-checkpoint --cont --project-name exp13_data_driven --method grpo-passk    --model qwen2.5-medium   --tp 4  --no-validation --eval --cpdir "/ptmp/rfechner/out/exp13_checkpoints/qwen2.5_1.5b__grpo-passk" --lpdir "/ptmp/rfechner/out/exp13_rollouts_subsampled/qwen2.5_1.5b__grpo-passk/val_jsonl"

    set +x
} 2>&1 | tee $LOGFILE

