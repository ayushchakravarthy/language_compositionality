#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ucdavis
#SBATCH --mem=1G
#SBATCH --time=200:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --output=logs/scan/transformer_def_add_jump.txt

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /home/akchak/.bashrc
source /home/tqhe/.bashrc
conda activate lp

python main.py \
--dataset scan \
--split addjump \
--num_runs 3 \
--batch_size 128 \
--num_epochs 100 \
--model_type transformer \
--d_model 400 \
--nhead 8 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--dim_feedforward 2048 \
--dropout 0.1 \
--learning_rate 0.00004 \
--results_dir transformer \
--out_data_file train_defaults_jump \
--out_attn_wts train_defaults_jump_attn_maps \
--checkpoint_path ../weights/transformer/scan/defaults_addjump.pt \
--checkpoint_every 4 \
--record_loss_every 20