#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x /home/yqs/miniconda3/envs/yqs312/bin/python ]]; then
    PYTHON_BIN=/home/yqs/miniconda3/envs/yqs312/bin/python
  elif [[ -x /root/miniconda3/bin/python ]]; then
    PYTHON_BIN=/root/miniconda3/bin/python
  else
    PYTHON_BIN=python
  fi
fi

COMMON_ARGS=(
  --batch_size 64
  --epochs 200
  --extract_layers 7 9 11
  --prop_train_labels 0.5
  --class_split_seed 0
)

for DATASET in aid nwpu; do
  for SPLIT in random confusable; do
    "${PYTHON_BIN}" train.py \
      --dataset_name "${DATASET}" \
      --split_type "${SPLIT}" \
      --exp_name "layergcd_${DATASET}_${SPLIT}" \
      "${COMMON_ARGS[@]}"
  done
done
