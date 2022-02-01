#!/usr/bin/env bash

python3 main.py \
--dataset pcfg-set \
--num_runs 1 \
--batch_size 256 \
--num_epochs 100 \
--model_type transformer \
--d_model 128 \
--nhead 8 \
--n_layers 2 \
--dim_feedforward 256 \
--dropout 0.1 \
--learning_rate 0.0001 \
--results_dir transformer \
--out_data_file train_defaults \
--out_attn_wts train_defaults_attn_maps \
--checkpoint_path ../weights/transformer/pcfgset/defaults.pt \
--checkpoint_every 4 \
--record_loss_every 20
