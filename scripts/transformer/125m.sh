#!/bin/bash

DATA_PATH=TODO
DATA_NAME="the_pile" # "books3" 

# Product should equal 0.5 million
SEQ_LEN=2048
BS=256

# Experiment details
EXP_NAME=TODO
EXP_DIR=TODO

sudo mkdir -p /${EXP_DIR}/${EXP_NAME} && sudo chmod -R 777 ${EXP_DIR}/${EXP_NAME};
cd ../..

python3 -m ttt.train \
        --mesh_dim='!-1,1,1' \
        --dtype='fp32' \
        --total_steps=4800 \
        --save_checkpoint_freq=1000 \
        --save_milestone_freq=2000 \
        --load_model_config='125m' \
        --dataset_path=${DATA_PATH} \
        --dataset_name=${DATA_NAME} \
        --seq_length=${SEQ_LEN} \
        --global_batch_size=${BS} \
        --optimizer.type='adamw' \
        --optimizer.adamw_optimizer.weight_decay=0.1 \
        --optimizer.adamw_optimizer.lr=3e-3 \
        --optimizer.adamw_optimizer.end_lr=1e-5 \
        --optimizer.adamw_optimizer.lr_warmup_steps=480 \
        --optimizer.adamw_optimizer.lr_decay_steps=4800 \
        --exp_dir=${EXP_DIR} \
        --exp_name=${EXP_NAME}