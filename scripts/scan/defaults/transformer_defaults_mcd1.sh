#!/usr/bin/env bash

python main.py \
--dataset scan \
--split mcd1 \
--num_runs 10 \
--batch_size 512 \
--num_epochs 200 \
--model_type transformer \
--d_model 256 \
--nhead 8 \
--n_layers 2 \
--dim_feedforward 512 \
--dropout 0.1 \
--learning_rate 0.001 \
--results_dir transformer \
--out_data_file train_defaults_mcd1 \
--out_attn_wts train_defaults_mcd1_attn_maps \
--checkpoint_path ../weights/transformer/scan/defaults_mcd1.pt \
--checkpoint_every 4 \
--record_loss_every 20