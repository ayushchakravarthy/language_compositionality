#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ucdavis
#SBATCH --mem=1G
#SBATCH --time=200:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --output=logs/scan/language_parser_def_add_jump.txt

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
--batch_size 64 \
--num_epochs 100 \
--model_type language_parser \
--d_model 256 \
--nhead 8 \
--dim_feedforward 2048 \
--dropout 0.1 \
--learning_rate 0.0001 \
--results_dir language_parser \
--out_data_file train_defaults_jump \
--out_attn_wts train_defaults_jump_attn_maps \
--checkpoint_path weights/language_parser/scan/defaults_addjump.pt \
--checkpoint_every 4 \
--record_loss_every 20
