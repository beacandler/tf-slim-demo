#!/usr/bin/env bash
# Run evaluation.
CUDA_VISIBLE_DEVICES=1 \
python src/eval.py \
  --checkpoint_path=./log/train_dir \
  --eval_dir=./log/eval_dir \
  --dataset_name=flowers \
  --dataset_split_name=val \
  --num_evals=50 \
  --eval_interval_secs=10 \
