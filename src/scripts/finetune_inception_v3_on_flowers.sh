#!/usr/bin/env bash

# Note: donot contain 'space' in variabel define

# This script performs the following operations.
# 1. Downloads the pretrained InceptionV3 model.
# 2. Fine-tunes an InceptionV3 model on the Flowers training set.

# Usage:
# bash ./src/scripts/finetune_inception_v3_on_flowers.sh

set -e
# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=./src/models

# Downloads the pre-trained checkpoint.

if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
    mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi

if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt ]; then
    wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
    tar -xvf inception_v3_2016_08_28.tar.gz
    mv inception_v3.ckpt ${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt
    rm inception_v3_2016_08_28.tar.gz
fi

# Fine-tune only all the new layers for 1000 steps.
CUDA_VISIBLE_DEVICES=0 \
python src/train.py \
    --train_dir=./log/train_dir \
    --reset_train_dir=True \
    --dataset_name=flowers \
    --model_name=inception_v3 \
    --dataset_split_name=train \
    --checkpoint_inception=./src/models/inception_v3.ckpt \
    --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --max_number_of_steps=200000 \
    --batch_size=32 \
    --learning_rate=0.001 \
    --learning_rate_deacy_type=fixed \
    --save_interval_secs=10 \
    --save_summaries_secs=10 \
    --log_every_n_steps=10 \
    --optimizier=rmsprop \
    --weight_decay=0.00004