#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ucdavis
#SBATCH --mem=1G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --output=logs/language_parser_def_add_jump_20_runs.txt

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /home/akchak/.bashrc
source /home/tqhe/.bashrc
conda activate lp

python main.py \
--split addjump \
--num_runs 20 \
--batch_size 32 \
--num_epochs 1000 \
--model_type language_parser \
--d_model 6 \
--nhead 2 \
--ffn_exp 3 \
--num_parts 16 \
--num_decoder_layers 2 \
--dim_feedforward 20 \
--dropout 0.1 \
--learning_rate 0.001 \
--results_dir language_parser \
--out_data_file train_defaults_jump \
--out_attn_wts train_defaults_jump_attn_maps \
--checkpoint_path weights/language_parser/defaults_addjump.pt \
--checkpoint_every 2 \
--record_loss_every 20
