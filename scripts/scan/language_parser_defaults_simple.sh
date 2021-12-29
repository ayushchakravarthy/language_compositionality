#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ucdavis
#SBATCH --mem=1G
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --output=logs/scan/language_parser_def_simple.txt

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /home/akchak/.bashrc
source /home/tqhe/.bashrc
conda activate lp

python main.py \
--dataset scan \
--split simple \
--num_runs 1 \
--batch_size 256 \
--num_epochs 150 \
--model_type language_parser \
--d_model 256 \
--n_layers 6 \
--dim_feedforward 1024 \
--dropout 0.1 \
--learning_rate 0.00004 \
--results_dir language_parser \
--out_data_file train_defaults_simple \
--out_attn_wts train_defaults_simple_attn_maps \
--checkpoint_path ../weights/language_parser/scan/defaults_simple.pt \
--checkpoint_every 4 \
--record_loss_every 20
