#!/bin/bash
cd /u/rfechner
LOGFILE="jobs/exp11_$(date +'%Y%m%d_%H%M%S').out"
id_data="/u/rfechner/data/ariadne/id-deltas.jsonl" # dropped last 2 rows to align with number of gpus... (this sucks ass but im doing it now)
ood_data="/u/rfechner/data/ariadne/ood-deltas.jsonl" # dropped last 6 rows to align with number of gpus

# This experiment is about correlating deltas of out-of-distribution deltas and in-distribution deltas.
# The OOD samples are tuples (correct response, error augmented response), the ID samples are tuples (correct response, incorrect response with calc error)
# The ID samples are per-checkpoint, we're evaluating all checkpoints on all samples (it's just easier for me to run the experiment this way... im lazy)
# Later on, we'll calculate the log-probabilities for (gt, aug) and (gt, err) and try to correlate along the checkpoints.
# What we'd like to observe: Decreasing likelihood delta for both sets of traces across training.
{
    set -x
    python verl/workspace/train/run.py --project-name exp12 --identifier id_data    --method gspo     --model qwen2.5-large --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__gspo"   --no-validation --lpfile "${id_data}"
    python verl/workspace/train/run.py --project-name exp12 --identifier ood_data   --method gspo     --model qwen2.5-large --eval --cpdir "/ptmp/rfechner/out/exp05_train_qwen2.5-7b/qwen2.5_7b__gspo"   --no-validation --lpfile "${ood_data}"
    set +x
} 2>&1 | tee $LOGFILE