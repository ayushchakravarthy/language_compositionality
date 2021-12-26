#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ucdavis
#SBATCH --mem=1G
#SBATCH --time=50:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --output=logs/scan/language_parser_def_mcd1.txt

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /home/akchak/.bashrc
source /home/tqhe/.bashrc
conda activate lp

python main.py \
--dataset scan \
--split mcd1 \
--num_runs 1 \
--batch_size 32 \
--num_epochs 100 \
--model_type language_parser \
--d_model 6 \
--nhead 2 \
--ffn_exp 3 \
--num_parts 16 \
--n_layers 2 \
--dim_feedforward 20 \
--dropout 0.1 \
--learning_rate 0.001 \
--results_dir language_parser \
--out_data_file train_defaults_mcd1 \
--out_attn_wts train_defaults_mcd1_attn_maps \
--checkpoint_path ../weights/language_parser/scan/defaults_mcd1.pt \
--checkpoint_every 4 \
--record_loss_every 20