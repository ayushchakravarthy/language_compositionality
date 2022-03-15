#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --job-name="sep-transformer"
#SBATCH --time=5:00:00
#SBATCH --output="depth2_defaults_mcd3.txt"

source /home/akchak/.bashrc
conda init
conda activate test

python main.py \
--pos \
--cat_xm \
--sp_kernel \
--threshold 0.08 \
--dataset scan \
--split mcd3 \
--depth 2 \
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
--out_data_file train_defaults_mcd3 \
--out_attn_wts train_defaults_mcd3_attn_maps \
--checkpoint_path ../weights/sep-transformer/scan/defaults_mcd3.pt \
--checkpoint_every 4 \
--record_loss_every 20