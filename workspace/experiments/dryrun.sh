#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/dryrun_$(date +'%Y%m%d_%H%M%S').out"
{
    set -x
    python verl/workspace/train/run.py --dryrun --project-name dryrun --method grpo --model llama3-small --epochs 2 --savefreq 5 --testfreq 1 --cont
    python verl/workspace/train/run.py --dryrun --project-name dryrun --method drgrpo --model llama3-small --epochs 2 --savefreq 5 --testfreq 1 --cont
    python verl/workspace/train/run.py --dryrun --project-name dryrun --method dapo --model llama3-small --epochs 2 --savefreq 5 --testfreq 1 --cont
    python verl/workspace/train/run.py --dryrun --project-name dryrun --method gspo --model llama3-small --epochs 2 --savefreq 5 --testfreq 1 --cont
    python verl/workspace/train/run.py --dryrun --project-name dryrun --method kl-cov --model llama3-small --epochs 2 --savefreq 5 --testfreq 1 --cont
    python verl/workspace/train/run.py --dryrun --project-name dryrun --method clip-cov --model llama3-small --epochs 2 --savefreq 5 --testfreq 1 --cont
    python verl/workspace/train/run.py --dryrun --project-name dryrun --method gtpo --model llama3-small --epochs 2 --savefreq 5 --testfreq 1 --cont
    python verl/workspace/train/run.py --dryrun --project-name dryrun --method grpo-s --model llama3-small --epochs 2 --savefreq 5 --testfreq 1 --cont
    python verl/workspace/train/run.py --dryrun --project-name dryrun --method entropy_reg --model llama3-small --epochs 2 --savefreq 5 --testfreq 1 --cont
    set +x
} 2>&1 | tee $LOGFILE