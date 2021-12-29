#!/usr/bin/env bash

python main.py \
--dataset scan \
--split mcd3 \
--num_runs 1 \
--batch_size 256 \
--num_epochs 150 \
--model_type language_parser \
--d_model 256 \
--nhead 8 \
--n_layers 6 \
--dim_feedforward 1024 \
--dropout 0.1 \
--learning_rate 0.00004 \
--results_dir language_parser \
--out_data_file train_defaults_mcd3 \
--out_attn_wts train_defaults_mcd3_attn_maps \
--checkpoint_path ../weights/language_parser/scan/defaults_mcd3.pt \
--checkpoint_every 4 \
--record_loss_every 20
