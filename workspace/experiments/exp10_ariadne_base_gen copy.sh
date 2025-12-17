#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/exp10_$(date +'%Y%m%d_%H%M%S').out"

# In this experiment we're generating base datasets for further log-prob analysis. We're generating OOD-augmented samples and try to judge ID-samples.

{
    set -x
    python verl/workspace/rollouts/run.py --model Qwen/Qwen3-8B --data /u/rfechner/data/ariadne/ood-prompts.parquet    --out /u/rfechner/data/ariadne/ood-outputs-qw3-8b.parquet
    python verl/workspace/rollouts/run.py --model Qwen/Qwen3-8B --data /u/rfechner/data/ariadne/id-prompts.parquet     --out /u/rfechner/data/ariadne/id-outputs-qw3-8b.parquet
    set +x
} 2>&1 | tee $LOGFILE

