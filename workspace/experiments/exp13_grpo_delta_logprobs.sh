#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/exp13_deltas_$(date +'%Y%m%d_%H%M%S').out"
combined="/u/rfechner/data/final_deltas/delta_prompts.jsonl" # just reasoning types, reasoning strategies and error types combined into a large jsonl file

# This is the final delta-logprob experiment. Under knowledge of protential confounds we're still going forward to collect results, noting later they're confounded. 
{
    set -x

    # Qw2.5-7B
    python verl/workspace/train/run.py --cont --project-name exp13_deltas --method grpo     --model qwen2.5-large   --tp 4   --no-validation --eval --lpfile  "${combined}" --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__grpo"

    # EuroLLM-9B
    python verl/workspace/train/run.py --cont --project-name exp13_deltas --method grpo        --model eurollm   --tp 4   --no-validation --eval --lpfile  "${combined}" --cpdir "/ptmp/rfechner/out/exp13_checkpoints/eurollm_9b_instruct__grpo"

    # Llama3.1-8B
    python verl/workspace/train/run.py --cont --project-name exp13_deltas --method grpo        --model llama3-large   --tp 4   --no-validation --eval --lpfile  "${combined}" --cpdir "/ptmp/rfechner/out/exp13_checkpoints/llama_3.1_8b_instruct__grpo"

    # Qwen2.5-1.5B
    python verl/workspace/train/run.py --project-name exp13_deltas --method grpo        --model qwen2.5-medium   --tp 4   --no-validation --eval --lpfile  "${combined}" --cpdir "/ptmp/rfechner/out/exp13_checkpoints/qwen2.5_1.5b__grpo"
    
    set +x
} 2>&1 | tee $LOGFILE

