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

BASE_ARGS=(
  --dataset_name nwpu
  --split_type confusable
  --class_split_seed 0
  --prop_train_labels 0.5
  --batch_size 64
  --epochs 200
)

"${PYTHON_BIN}" train.py "${BASE_ARGS[@]}" --extract_layers 7 9 11 --exp_name "ablate_full"
"${PYTHON_BIN}" train.py "${BASE_ARGS[@]}" --extract_layers 7 9 11 --single_layer_hierarchy --exp_name "ablate_single_layer_hierarchy"
"${PYTHON_BIN}" train.py "${BASE_ARGS[@]}" --extract_layers 7 9 11 --disable_hierarchy --exp_name "ablate_no_hierarchy"
"${PYTHON_BIN}" train.py "${BASE_ARGS[@]}" --extract_layers 7 9 11 --no_prompts --exp_name "ablate_no_prompts"
"${PYTHON_BIN}" train.py "${BASE_ARGS[@]}" --extract_layers 7 9 11 --fine_prompt_only --exp_name "ablate_fine_prompt_only"
"${PYTHON_BIN}" train.py "${BASE_ARGS[@]}" --extract_layers 7 9 11 --disable_bridge --exp_name "ablate_no_bridge"
"${PYTHON_BIN}" train.py "${BASE_ARGS[@]}" --extract_layers 7 9 11 --disable_relation_relaxation --exp_name "ablate_no_relation_relaxation"
"${PYTHON_BIN}" train.py "${BASE_ARGS[@]}" --extract_layers 7 9 11 --relation_relaxation_mode coarse --exp_name "ablate_coarse_relation_relaxation"

"${PYTHON_BIN}" train.py "${BASE_ARGS[@]}" --extract_layers 11 --exp_name "ablate_layers_11"
"${PYTHON_BIN}" train.py "${BASE_ARGS[@]}" --extract_layers 9 11 --exp_name "ablate_layers_9_11"
"${PYTHON_BIN}" train.py "${BASE_ARGS[@]}" --extract_layers 7 9 11 --exp_name "ablate_layers_7_9_11"
"${PYTHON_BIN}" train.py "${BASE_ARGS[@]}" --extract_layers 5 7 9 11 --exp_name "ablate_layers_5_7_9_11"
