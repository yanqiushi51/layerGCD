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

"${PYTHON_BIN}" diagnostic.py \
  --dataset_name nwpu \
  --split_type confusable \
  --class_split_seed 0 \
  --extract_layers 5 7 9 11 \
  --batch_size 128
