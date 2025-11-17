#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/exp06_$(date +'%Y%m%d_%H%M%S').out"
reasoning_strategies="/u/rfechner/data/incomplete_reasoning_strategies/deltas.jsonl"
reasoning_types="/u/rfechner/data/incomplete_reasoning_types/deltas.jsonl"
# In this experiment we look at the delta between likelihoods of ground truth and augmented answers. The ground truths were sysnthesized using GPT4 and
# are drawn from the EIC error dataset from literature. The augmentations are made using a similar prompting strategy (4-shot sampling) and collected into a dataset,
# the run script used is "generate_behaviours.sbatch". Beware: The API isn't existant, so you have to setup the generation manually in the corresonding python file.
# Further, I'm bringing the collected traces into a 'delta format' by using verl/workspace/download/create_delta_datasets.py . Again: please beware with re-starting,
# API isn't developed, you have to manually point to files to create the delta dataset from.
# Finally, we pass the delta dataset to the run script. 
{
    set -x
    python verl/workspace/train/run.py --project-name exp06 --identifier reasoning_strategies --method grpo --model llama3-small --eval --cpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__grpo" --no-validation --lpfile "${reasoning_strategies}"
    python verl/workspace/train/run.py --project-name exp06 --identifier reasoning_types --method grpo --model llama3-small --eval --cpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__grpo" --no-validation --lpfile "${reasoning_types}"
    python verl/workspace/train/run.py --project-name exp06 --identifier reasoning_strategies --method kl-cov --model llama3-small --eval --cpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__kl-cov" --no-validation --lpfile "${reasoning_strategies}"
    python verl/workspace/train/run.py --project-name exp06 --identifier reasoning_types --method kl-cov --model llama3-small --eval --cpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__kl-cov" --no-validation --lpfile "${reasoning_types}"
    python verl/workspace/train/run.py --project-name exp06 --identifier reasoning_strategies --method grpo-s --model llama3-small --eval --cpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__grpo-s" --no-validation --lpfile "${reasoning_strategies}"
    python verl/workspace/train/run.py --project-name exp06 --identifier reasoning_types --method grpo-s --model llama3-small --eval --cpdir "/ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__grpo-s" --no-validation --lpfile "${reasoning_types}"
    set +x
} 2>&1 | tee $LOGFILE