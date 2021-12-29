#!/usr/bin/env bash

python main.py \
--dataset scan \
--split mcd2 \
--num_runs 1 \
--batch_size 256 \
--num_epochs 100 \
--model_type transformer \
--d_model 512 \
--nhead 8 \
--n_layers 6 \
--dim_feedforward 2048 \
--dropout 0.1 \
--learning_rate 0.00004 \
--results_dir transformer \
--out_data_file train_defaults_mcd2 \
--out_attn_wts train_defaults_mcd2_attn_maps \
--checkpoint_path ../weights/transformer/scan/defaults_mcd2.pt \
--checkpoint_every 4 \
--record_loss_every 20