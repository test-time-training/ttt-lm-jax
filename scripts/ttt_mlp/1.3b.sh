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
        --total_steps=50000 \
        --save_checkpoint_freq=1000 \
        --save_milestone_freq=2000 \
        --load_model_config='1b-TTT' \
        --update_model_config="dict(seq_modeling_block='ttt_mlp', ttt_base_lr=0.1, ttt_base_lr_init=0.01, ttt_base_lr_warmup=5000)" \
        --dataset_path=${DATA_PATH} \
        --dataset_name=${DATA_NAME} \
        --seq_length=${SEQ_LEN} \
        --global_batch_size=${BS} \
        --optimizer.type='adamw' \
        --optimizer.adamw_optimizer.weight_decay=0.1 \
        --optimizer.adamw_optimizer.lr=1e-3 \
        --optimizer.adamw_optimizer.end_lr=1e-5 \
        --optimizer.adamw_optimizer.lr_warmup_steps=5000 \
        --optimizer.adamw_optimizer.lr_decay_steps=50000 \
        --exp_dir=${EXP_DIR} \
        --exp_name=${EXP_NAME}