#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name 'nwpu' \
    --split_type 'random' \
    --class_split_seed 0 \
    --prop_train_labels 0.5 \
    --batch_size 128 \
    --num_workers 8 \
    --eval_funcs 'v2' \
    --device 'cuda:0' \
    --exp_name 'aptgcd_nwpu_random'
