#!/usr/bin/env bash

python main.py \
--dataset cogs \
--split train-100 \
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
--out_data_file train_100_defaults \
--out_attn_wts train_100_defaults_attn_maps \
--checkpoint_path ../weights/transformer/cogs/defaults_train_100.pt \
--checkpoint_every 4 \
--record_loss_every 20
