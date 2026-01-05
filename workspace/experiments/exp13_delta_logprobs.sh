#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/exp13_deltas_$(date +'%Y%m%d_%H%M%S').out"
combined="/u/rfechner/data/final_deltas/delta_prompts.jsonl" # just reasoning types, reasoning strategies and error types combined into a large jsonl file

# This is the final delta-logprob experiment. Under knowledge of protential confounds we're still going forward to collect results, noting later they're confounded. 
{
    set -x

    # Qw2.5-7B
    python verl/workspace/train/run.py --cont --project-name exp13_deltas --method kl-cov     --model qwen2.5-large   --tp 4   --no-validation --eval --lpfile  "${combined}" --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__kl-cov"
    python verl/workspace/train/run.py --cont --project-name exp13_deltas --method grpo-s     --model qwen2.5-large   --tp 4   --no-validation --eval --lpfile  "${combined}" --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo-s"
    python verl/workspace/train/run.py --cont --project-name exp13_deltas --method gspo       --model qwen2.5-large   --tp 4   --no-validation --eval --lpfile  "${combined}" --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__gspo"
    python verl/workspace/train/run.py --cont --project-name exp13_deltas --method grpo-passk --model qwen2.5-large   --tp 4   --no-validation --eval --lpfile  "${combined}" --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo-passk"

    # EuroLLM-9B
    python verl/workspace/train/run.py --cont --project-name exp13_deltas --method kl-cov        --model eurollm   --tp 4   --no-validation --eval --lpfile  "${combined}" --cpdir "/ptmp/rfechner/out/exp13_checkpoints/eurollm_9b_instruct__kl-cov"
    python verl/workspace/train/run.py --cont --project-name exp13_deltas --method grpo-s        --model eurollm   --tp 4   --no-validation --eval --lpfile  "${combined}" --cpdir "/ptmp/rfechner/out/exp13_checkpoints/eurollm_9b_instruct__grpo-s"
    python verl/workspace/train/run.py --cont --project-name exp13_deltas --method gspo          --model eurollm   --tp 4   --no-validation --eval --lpfile  "${combined}" --cpdir "/ptmp/rfechner/out/exp13_checkpoints/eurollm_9b_instruct__gspo"
    python verl/workspace/train/run.py --cont --project-name exp13_deltas --method grpo-passk    --model eurollm   --tp 4   --no-validation --eval --lpfile  "${combined}" --cpdir "/ptmp/rfechner/out/exp13_checkpoints/eurollm_9b_instruct__grpo-passk"

    # Llama3.1-8B
    python verl/workspace/train/run.py --cont --project-name exp13_deltas --method kl-cov        --model llama3-large   --tp 4   --no-validation --eval --lpfile  "${combined}" --cpdir "/ptmp/rfechner/out/exp13_checkpoints/llama_3.1_8b_instruct__kl-cov"
    python verl/workspace/train/run.py --cont --project-name exp13_deltas --method grpo-s        --model llama3-large   --tp 4   --no-validation --eval --lpfile  "${combined}" --cpdir "/ptmp/rfechner/out/exp13_checkpoints/llama_3.1_8b_instruct__grpo-s"
    python verl/workspace/train/run.py --cont --project-name exp13_deltas --method gspo          --model llama3-large   --tp 4   --no-validation --eval --lpfile  "${combined}" --cpdir "/ptmp/rfechner/out/exp13_checkpoints/llama_3.1_8b_instruct__gspo"
    python verl/workspace/train/run.py --cont --project-name exp13_deltas --method grpo-passk    --model llama3-large   --tp 4   --no-validation --eval --lpfile  "${combined}" --cpdir "/ptmp/rfechner/out/exp13_checkpoints/llama_3.1_8b_instruct__grpo-passk"

    # Qwen2.5-1.5B
    # python verl/workspace/train/run.py --project-name exp13_deltas --method kl-cov        --model qwen2.5-medium   --tp 4   --no-validation --eval --lpfile  "${combined}" --cpdir "/ptmp/rfechner/out/exp13_checkpoints/qwen2.5_1.5b__kl-cov"
    python verl/workspace/train/run.py --cont --project-name exp13_deltas --method grpo-s        --model qwen2.5-medium   --tp 4   --no-validation --eval --lpfile  "${combined}" --cpdir "/ptmp/rfechner/out/exp13_checkpoints/qwen2.5_1.5b__grpo-s"
    python verl/workspace/train/run.py --cont --project-name exp13_deltas --method gspo          --model qwen2.5-medium   --tp 4   --no-validation --eval --lpfile  "${combined}" --cpdir "/ptmp/rfechner/out/exp13_checkpoints/qwen2.5_1.5b__gspo"
    python verl/workspace/train/run.py --cont --project-name exp13_deltas --method grpo-passk    --model qwen2.5-medium   --tp 4   --no-validation --eval --lpfile  "${combined}" --cpdir "/ptmp/rfechner/out/exp13_checkpoints/qwen2.5_1.5b__grpo-passk"

    set +x
} 2>&1 | tee $LOGFILE

