#!/usr/bin/env bash

python main.py \
--dataset scan \
--split mcd2 \
--num_runs 1 \
--batch_size 256 \
--num_epochs 150 \
--model_type language_parser \
--d_model 128 \
--nhead 8 \
--n_layers 1 \
--dim_feedforward 256 \
--dropout 0.1 \
--learning_rate 0.0005 \
--results_dir language_parser \
--out_data_file train_defaults_mcd2 \
--out_attn_wts train_defaults_mcd2_attn_maps \
--checkpoint_path ../weights/language_parser/scan/defaults_mcd2.pt \
--checkpoint_every 4 \
--record_loss_every 20
