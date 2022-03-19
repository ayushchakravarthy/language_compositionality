#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name="sep-transformer"
#SBATCH --time=2:00:00
#SBATCH --output="depth2_defaults_addjump.txt"

python main.py \
--pos \
--cat_xm \
--encoding_scheme absolute \
--dataset scan \
--split addjump \
--depth 2 \
--num_runs 10 \
--batch_size 1024 \
--num_epochs 400 \
--model_type sep-transformer \
--d_model 256 \
--nhead 8 \
--n_layers 2 \
--dim_feedforward 512 \
--dropout 0.1 \
--learning_rate 0.00025 \
--results_dir sep-transformer \
--out_data_file train_double_jump \
--out_attn_wts train_double_jump_attn_maps \
--checkpoint_path ../weights/sep-transformer/scan/double_addjump.pt \
--checkpoint_every 4 \
--record_loss_every 20