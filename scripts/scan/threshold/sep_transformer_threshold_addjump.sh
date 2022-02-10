#!/usr/bin/env bash

python main.py \
--pos \
--sp_kernel \
--threshold 0.2 \
--dataset scan \
--split addjump \
--num_runs 1 \
--batch_size 256 \
--num_epochs 150 \
--model_type sep-transformer \
--d_model 256 \
--nhead 8 \
--n_layers 2 \
--dim_feedforward 512 \
--dropout 0.1 \
--learning_rate 0.001 \
--results_dir sep-transformer \
--out_data_file train_threshold_jump \
--out_attn_wts train_threshold_jump_attn_maps \
--checkpoint_path ../weights/sep-transformer/scan/threshold_addjump.pt \
--checkpoint_every 4 \
--record_loss_every 20