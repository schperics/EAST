#!/usr/bin/env bash

python3 multigpu_train.py \
    --gpu_list=2 \
    --input_size=512 \
    --batch_size_per_gpu=14
    --checkpoint_path=/mnt/argman_EAST/train0/ \
    --text_scale=512 \
    --training_data_path=/mnt/icdar2017_mlt/train \
    --geometry=RBOX \
    --learning_rate=0.0001 \
    --num_readers=24 \
    --pretrained_model_path=/mnt/pretrained/resnet_v1_50.ckpt
