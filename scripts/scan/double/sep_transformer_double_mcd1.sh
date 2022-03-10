#!/usr/bin/env bash

python main.py \
--pos \
--cat_xm \
--dataset scan \
--split mcd1 \
--num_runs 10 \
--batch_size 512 \
--num_epochs 200 \
--model_type sep-transformer \
--d_model 256 \
--nhead 8 \
--n_layers 2 \
--dim_feedforward 512 \
--dropout 0.1 \
--learning_rate 0.001 \
--results_dir sep-transformer \
--out_data_file train_double_mcd1 \
--out_attn_wts train_double_mcd1_attn_maps \
--checkpoint_path ../weights/sep-transformer/scan/double_mcd1.pt \
--checkpoint_every 4 \
--record_loss_every 20