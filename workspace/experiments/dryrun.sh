#!/bin/bash
cd /u/rfechner
python verl/workspace/train/run.py --identifier dryrun --method grpo --model qwen3-small --epochs 2 --savefreq 5 --testfreq 1
python verl/workspace/train/run.py --identifier dryrun --method grpo --model llama3-small --epochs 2 --savefreq 5 --testfreq 1
python verl/workspace/train/run.py --identifier dryrun --method drgrpo --model qwen3-small --epochs 2 --savefreq 5 --testfreq 1
python verl/workspace/train/run.py --identifier dryrun --method drgrpo --model llama3-small --epochs 2 --savefreq 5 --testfreq 1
python verl/workspace/train/run.py --identifier dryrun --method dapo --model qwen3-small --epochs 2 --savefreq 5 --testfreq 1
python verl/workspace/train/run.py --identifier dryrun --method dapo --model llama3-small --epochs 2 --savefreq 5 --testfreq 1
python verl/workspace/train/run.py --identifier dryrun --method gspo --model qwen3-small --epochs 2 --savefreq 5 --testfreq 1
python verl/workspace/train/run.py --identifier dryrun --method gspo --model llama3-small --epochs 2 --savefreq 5 --testfreq 1
