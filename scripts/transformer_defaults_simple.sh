#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ucdavis
#SBATCH --mem=1G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --output=logs/transformer_def_simple_20_runs.txt

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /home/akchak/.bashrc
source /home/tqhe/.bashrc
conda activate lp

python main.py \
--split simple \
--num_runs 20 \
--batch_size 32 \
--num_epochs 100 \
--model_type transformer \
--d_model 12 \
--nhead 2 \
--num_encoder_layers 2 \
--num_decoder_layers 2 \
--dim_feedforward 20 \
--dropout 0.1 \
--learning_rate 0.001 \
--results_dir transformer \
--out_data_file train_defaults_simple \
--out_attn_wts train_defaults_simple_attn_maps \
--checkpoint_path weights/transformer/defaults_simple.pt \
--checkpoint_every 4 \
--record_loss_every 20
