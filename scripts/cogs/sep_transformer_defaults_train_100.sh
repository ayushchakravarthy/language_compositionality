#!/usr/bin/env bash

python main.py \
--pos \
--cat_xm \
--encoding_scheme absolute \
--sp_kernel 0.08 \
--dataset cogs \
--split train-100 \
--num_runs 1 \
--batch_size 32 \
--num_epochs 200 \
--model_type sep-transformer \
--d_model 256 \
--nhead 8 \
--n_layers 2 \
--dim_feedforward 512 \
--dropout 0.1 \
--learning_rate 0.00025 \
--results_dir sep-transformer \
--out_data_file train_100_defaults \
--out_attn_wts train_100_defaults_attn_maps \
--checkpoint_path ../weights/sep-transformer/cogs/defaults_train_100.pt \
--checkpoint_every 2 \
--record_loss_every 20
