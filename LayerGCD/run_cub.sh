#!/bin/bash
export DATASET_DIR=/root/Cold-Discovery

python train.py \
    --dataset_name cub \
    --batch_size 64 \
    --epochs 200 \
    --extract_layers 7 9 11 \
    --exp_name layergcd_prompt_cub
