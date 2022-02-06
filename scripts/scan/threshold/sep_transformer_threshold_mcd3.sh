#!/usr/bin/env bash

python main.py \
--pos \
--sp_kernel \
--threshold 0.6 \
--dataset scan \
--split mcd3 \
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
--out_data_file train_threshold_mcd3 \
--out_attn_wts train_threshold_mcd3_attn_maps \
--checkpoint_path ../weights/sep-transformer/scan/threshold_mcd3.pt \
--checkpoint_every 4 \
--record_loss_every 20