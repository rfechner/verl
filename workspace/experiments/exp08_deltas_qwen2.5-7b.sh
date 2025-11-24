#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/exp08_$(date +'%Y%m%d_%H%M%S').out"
reasoning_strategies="/u/rfechner/data/incomplete_reasoning_strategies/deltas.jsonl"
error_types="/u/rfechner/data/eic_gsm8k_generated/delta.jsonl"
reasoning_types="/u/rfechner/data/incomplete_reasoning_types/deltas.jsonl"

# This experiment run is a follow up to exp06 and exp06-llama. I'd like to see similar trends with the qwen2.5 models

{
    set -x
    python verl/workspace/train/run.py --project-name exp08 --identifier reasoning_strategies         --method grpo     --model qwen2.5-large --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo"   --no-validation --lpfile "${reasoning_strategies}" --cont
    python verl/workspace/train/run.py --project-name exp08 --identifier reasoning_types              --method grpo     --model qwen2.5-large --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo"   --no-validation --lpfile "${reasoning_types}" --cont
    python verl/workspace/train/run.py --project-name exp08 --identifier error_types                  --method grpo     --model qwen2.5-large --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo"   --no-validation --lpfile "${error_types}" --cont
    python verl/workspace/train/run.py --project-name exp08 --identifier reasoning_strategies         --method kl-cov   --model qwen2.5-large --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__kl-cov" --no-validation --lpfile "${reasoning_strategies}" --cont
    python verl/workspace/train/run.py --project-name exp08 --identifier reasoning_types              --method kl-cov   --model qwen2.5-large --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__kl-cov" --no-validation --lpfile "${reasoning_types}" --cont
    python verl/workspace/train/run.py --project-name exp08 --identifier error_types                  --method kl-cov   --model qwen2.5-large --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__kl-cov" --no-validation --lpfile "${error_types}" --cont
    python verl/workspace/train/run.py --project-name exp08 --identifier reasoning_strategies         --method grpo-s   --model qwen2.5-large --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo-s" --no-validation --lpfile "${reasoning_strategies}" --cont
    python verl/workspace/train/run.py --project-name exp08 --identifier reasoning_types              --method grpo-s   --model qwen2.5-large --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo-s" --no-validation --lpfile "${reasoning_types}" --cont
    python verl/workspace/train/run.py --project-name exp08 --identifier error_types                  --method grpo-s   --model qwen2.5-large --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo-s" --no-validation --lpfile "${error_types}" --cont
    python verl/workspace/train/run.py --project-name exp08 --identifier reasoning_strategies         --method dapo     --model qwen2.5-large --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__dapo"   --no-validation --lpfile "${reasoning_strategies}" --cont
    python verl/workspace/train/run.py --project-name exp08 --identifier reasoning_types              --method dapo     --model qwen2.5-large --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__dapo"   --no-validation --lpfile "${reasoning_types}" --cont
    python verl/workspace/train/run.py --project-name exp08 --identifier error_types                  --method dapo     --model qwen2.5-large --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__dapo"   --no-validation --lpfile "${error_types}" --cont
    set +x
} 2>&1 | tee $LOGFILE