#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/eval_dryrun_$(date +'%Y%m%d_%H%M%S').out"
{
    set -x
    python verl/workspace/train/run.py --identifier eval_test_regular --project-name eval_test --method grpo --model llama3-small --eval --cp /ptmp/rfechner/out/dryrun/llama_3.2_1b_instruct__grpo/global_step_5
    set +x
} 2>&1 | tee $LOGFILE