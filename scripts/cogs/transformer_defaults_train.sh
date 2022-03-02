#!/usr/bin/env bash

python main.py \
--dataset cogs \
--split train \
--num_runs 1 \
--batch_size 256 \
--num_epochs 100 \
--model_type transformer \
--d_model 128 \
--nhead 4 \
--n_layers 2 \
--dim_feedforward 512 \
--dropout 0.1 \
--learning_rate 0.0001 \
--results_dir transformer \
--out_data_file train_defaults \
--out_attn_wts train_defaults_attn_maps \
--checkpoint_path ../weights/transformer/cogs/defaults_train.pt \
--checkpoint_every 4 \
--record_loss_every 20
