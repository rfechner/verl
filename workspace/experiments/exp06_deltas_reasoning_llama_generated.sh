#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/exp06_$(date +'%Y%m%d_%H%M%S').out"
reasoning_strategies="/u/rfechner/data/incomplete_reasoning_strategies/meta-llama--Llama-3.1-8B-Instruct.jsonl-deltas.jsonl"
reasoning_types="/u/rfechner/data/incomplete_reasoning_types/meta-llama--Llama-3.1-8B-Instruct.jsonl-deltas.jsonl"

{
    set -x
    python verl/workspace/train/run.py --project-name exp06_llama_gen --identifier reasoning_strategies --method grpo --model llama3-small --eval --cpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__grpo" --no-validation --lpfile "${reasoning_strategies}"
    python verl/workspace/train/run.py --project-name exp06_llama_gen --identifier reasoning_types --method grpo --model llama3-small --eval --cpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__grpo" --no-validation --lpfile "${reasoning_types}"
    python verl/workspace/train/run.py --project-name exp06_llama_gen --identifier reasoning_strategies --method kl-cov --model llama3-small --eval --cpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__kl-cov" --no-validation --lpfile "${reasoning_strategies}"
    python verl/workspace/train/run.py --project-name exp06_llama_gen --identifier reasoning_types --method kl-cov --model llama3-small --eval --cpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__kl-cov" --no-validation --lpfile "${reasoning_types}"
    python verl/workspace/train/run.py --project-name exp06_llama_gen --identifier reasoning_strategies --method grpo-s --model llama3-small --eval --cpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__grpo-s" --no-validation --lpfile "${reasoning_strategies}"
    python verl/workspace/train/run.py --project-name exp06_llama_gen --identifier reasoning_types --method grpo-s --model llama3-small --eval --cpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__grpo-s" --no-validation --lpfile "${reasoning_types}"
    set +x
} 2>&1 | tee $LOGFILE