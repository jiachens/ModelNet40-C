#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
LOG="log_txt/train_scanobjnn.txt.$(date +'%Y-%m-%d_%H-%M-%S')"
exec &> >(tee -a "${LOG}")

run="${1}"

python train.py \
    --resolution 128 \
    --views 6 \
    --weight_decay 0 \
    --batch_size 60 \
    --aug \
    --sigma 0.01 \
    --clip 0.05 \
    --ratio 0.35 \
    --learning_rate 0.001 \
    --decay_rate 0.7 \
    --size 1 \
    --learning_rate_clip 0 \
    --num_class 15 \
    --kernel_size 3 \
    --conv_stride 1 \
    --first_pool_size 0 \
    --first_pool_stride 0 \
    --max_epoch 300 \
    --record_file ../records/scanobjnn_run_${run}.csv \
    --log_dir log/scanobjnn_run_${run} \
    --train_file ../data/h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5 \
    --test_file ../data/h5_files/main_split/test_objectdataset_augmentedrot_scale75.h5
