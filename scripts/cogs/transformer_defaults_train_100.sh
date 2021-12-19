#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ucdavis
#SBATCH --mem=1G
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --output=logs/cogs/transformer_def_train_100.txt

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /home/akchak/.bashrc
source /home/akchak/.bashrc
conda activate lp


python main.py \
--dataset cogs \
--split train-100 \
--num_runs 1 \
--batch_size 256 \
--num_epochs 500 \
--model_type transformer \
--d_model 512 \
--nhead 8 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--dim_feedforward 2048 \
--dropout 0.1 \
--learning_rate 0.00004 \
--results_dir transformer \
--out_data_file train_100_defaults \
--out_attn_wts train_100_defaults_attn_maps \
--checkpoint_path ../weights/transformer/cogs/defaults_train_100.pt \
--checkpoint_every 1 \
--record_loss_every 10
